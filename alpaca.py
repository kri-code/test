import transformers
import torch
from transformers import pipeline


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pipeline = pipeline(model="../alpaca_weights", torch_dtype=torch.bfloat16, trust_remote_code=True, device=device)

dialog = "Tell me about alpacas"

generate_text = pipeline(dialog)

print(generate_text[0]["generated_text"])
