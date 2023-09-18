import transformers
import torch
from transformers import pipeline

alpaca_model = transformers.AutoModelForCausalLM.from_pretrained("../alpaca_weights")
alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained("../alpaca_weights")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pipeline = pipeline(task= "text-generation", model=alpaca_model, tokenizer= alpaca_tokenizer, max_new_tokens=50,device=device, do_sample=False)

dialog = "Tell me about elephants"

generate_text = pipeline(dialog)

print(generate_text[0]["generated_text"])
