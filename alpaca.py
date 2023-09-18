import transformers
import torch
from transformers import pipeline

alpaca_model = transformers.AutoModelForCausalLM.from_pretrained("../alpaca_weights")
alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained("../alpaca_weights")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pipeline = pipeline(task='question-answering', model=alpaca_model, tokenizer= alpaca_tokenizer, device=device, max_length=512, do_sample=False)

dialog = "Tell me about alpacas"

generate_text = pipeline(dialog)

print(generate_text[0]["answer"])
