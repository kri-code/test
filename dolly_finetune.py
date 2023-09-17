import torch
import transformers
import os
from datasets import load_dataset



class dolly_finetune:
    def __init__(self,
                 model_name: "dolly-v2-7b",
                 device_id: int = 1,
                 batch_size: int = 1,
                 learning_rate: float = 2e-4,
                 logging_steps: int = 50,
                 max_steps: int = 1000):
                 
                 
        dataset = load_dataset("Amod/mental_health_counseling_conversations")
        dataset = dataset["train"].train_test_split(train_size=0.8)

        self.train_dataset = dataset["train"]
        self.test_dataset = dataset["test"]

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
                    do_eval=True  # Perform evaluation at the end of training
                    )
        self.model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", device_map="auto", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left")
        
        self.trainer = SFTTrainer(
            model=self.base_model,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            formatting_func=self.formatting_func,
            max_seq_length=1048,
            tokenizer=self.tokenizer,
            args=self.training_args,  # HF Trainer arguments
        )
    
    @staticmethod
    def formatting_func(example) -> List[str]:
        return [f"### Question: {example['Context']}\n ### Answer: {example['Response']}"]
        
    def train(self):
        self.trainer.train()


d = dolly_finetune()
d.train()
        
        
        
