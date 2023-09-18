import transformers
import torch
from transformers import pipeline

alpaca_model = transformers.AutoModelForCausalLM.from_pretrained("../alpaca_weights")
alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained("../alpaca_weights")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pipeline = pipeline(model=alpaca_model, torch_dtype=torch.bfloat16, trust_remote_code=True, device=device)

dialog = "Tell me about alpacas"

generate_text = pipeline(dialog)

print(generate_text[0]["generated_text"])
