import torch
import transformers
import os
from datasets import load_dataset
from typing import List, Callable, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    StoppingCriteriaList,
    StoppingCriteria,
    AutoConfig,
    pipeline,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
    AutoModel,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer



class dolly_finetune:
    def __init__(self,
                 model_name: "dolly-v2-3b",
                 device_id: int = 1,
                 batch_size: int = 8,
                 learning_rate: float = 2e-4,
                 logging_steps: int = 50,
                 max_steps: int = 1000):
        self.model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", device_map="auto", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left")
                 
                 
        dataset = load_dataset("Amod/mental_health_counseling_conversations")
        dataset = dataset["train"]
        print(dataset[1])

        dataset = self.tokenizer(dataset["Context"], truncation=True, padding=True)
        self.dataset = self.tokenizer(dataset["Response"], truncation=True, padding=True)

        out_dir = f"./medicine_results/.model_name"
        out_logs = f"./medicine_results/.logs"
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_logs, exist_ok=True)

        self.training_args = TrainingArguments(
                    output_dir=out_dir,
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=4,
                    learning_rate=learning_rate,
                    logging_steps=logging_steps,
                    max_steps=max_steps,
                    logging_dir=out_logs,  # Directory for storing logs
                    save_strategy="steps",  # Save the model checkpoint every logging step
                    save_steps=50,  # Save checkpoints every 50 steps
                    evaluation_strategy="steps",  # Evaluate the model every logging step
                    eval_steps=50,  # Evaluate and save checkpoints every 50 steps
                    )


        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        
        self.trainer = Trainer(
            model=self.model,
            data_collator=data_collator,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
            args=self.training_args,  # HF Trainer arguments
        )
    
   
        
    def train(self):
        self.trainer.train()


d = dolly_finetune(model_name="dolly-v2-3b")
d.train()
        
        
        
