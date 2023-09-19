import transformers
import torch
from transformers import pipeline

alpaca_model = transformers.AutoModelForCausalLM.from_pretrained("../alpaca_weights")
alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained("../alpaca_weights")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#pipeline = pipeline(task= "text-generation", model=alpaca_model, tokenizer= alpaca_tokenizer, max_new_tokens=100,device=device, do_sample=False)

input_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\r\n\r\n"
            "### Instruction:\r\nTell me about Alpacas.\r\n\r\n### Response:"
        )
inputs = alpaca_tokenizer(input_text, return_tensors="pt")
out = alpaca_model.generate(inputs=inputs.input_ids, max_new_tokens=100)
output_text = alpaca_tokenizer.batch_decode(out, skip_special_tokens=True)[0]
output_text = output_text[len(input_text) :]
print(f"Input: {input_text}\nCompletion: {output_text}")

