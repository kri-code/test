import transformers
import torch
from datasets import load_dataset
import json
import os
import sys
from typing import List
 
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
 
import fire

device = "cuda" if torch.cuda.is_available() else "cpu"
device

def generate_dataset():
    # load dataset
    dataset = load_dataset("Amod/mental_health_counseling_conversations")

    # convert dataset to the format in the original Alpaca repository
    dataset_data = [
        {
            "instruction": "Give advice or suggestions in response to a mental health-related question",
            "input": dataset["train"]["Context"][i],
            "output": dataset["train"]["Response"][i]
        }
        for i, e in enumerate(dataset["train"])
    ]

    # convert into a JSON file to use it for training the model later on
    with open("alpaca-mental-health-dataset.json", "w") as f:
        json.dump(dataset_data, f)
        
generate_dataset()

# load alpaca model
alpaca_model = transformers.AutoModelForCausalLM.from_pretrained("../alpaca_weights", device_map="auto")
alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained("../alpaca_weights")


# sets the pad_token_id to 0 to represent unknown tokens and sets the padding_side to "left" to pad sequences on the left side.
alpaca_tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
alpaca_tokenizer.padding_side = "left"

# load data (json file)
data = load_dataset("json", data_files="alpaca-mental-health-dataset.json")

# create prompts from the loaded dataset and tokenize them
def generate_prompt(data_point):
    """
    This function takes a data point from the dataset and generates a prompt by combining the instruction, input, and output values.
    """
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""

def tokenize(prompt, add_eos_token=True):
    """
    This function takes the generated prompt and tokenizes it using the tokenizer defined earlier. 
    It also adds an end-of-sequence token to the input sequence and sets the label to be the same as the input sequence.
    """
    result = alpaca_tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != alpaca_tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(alpaca_tokenizer.eos_token_id)
        result["attention_mask"].append(1)
 
    result["labels"] = result["input_ids"].copy()
 
    return result
    
def generate_and_tokenize_prompt(data_point):
    """
    This function combines the first two functions to generate and tokenize the prompt in one step.
    """
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt

# apply generate_and_tokenize_prompt() function to every example in dataset
train_data = (
    data["train"].map(generate_and_tokenize_prompt)
)


 
# parameters which are mostly derived from the fine-tuning script in the original repository:
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
 
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 300
OUTPUT_DIR = "experiments"

# prepare the model for training:
"""
initialize and prepare the model for training with the LORA algorithm, which is a form of quantization that can 
reduce model size and memory usage without significant loss in accuracy.
"""
model = prepare_model_for_int8_training(alpaca_model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

# TrainingArguments object which specifies various settings and hyperparameters for training the model
training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard"
)

# creates batches of input/output sequences for sequence-to-sequence models.
data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    args=training_arguments,
    data_collator=data_collator
)
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))
 
model = torch.compile(model)
 
trainer.train()
model.save_pretrained(OUTPUT_DIR)
