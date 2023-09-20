import transformers
import torch
from transformers import pipeline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

alpaca_model = transformers.AutoModelForCausalLM.from_pretrained("../alpaca_weights", device_map="auto")
alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained("../alpaca_weights")

#pipeline = pipeline(task= "text-generation", model=alpaca_model, tokenizer= alpaca_tokenizer, max_new_tokens=100,device=device, do_sample=False)

input_text = (
            "Below is an instruction that describes a task. "
            "Give advice or suggestions in response to a mental health-related question\r\n\r\n"
            "### Instruction:\r\n I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I'm worthless and how I shouldn't be here. I've never tried or contemplated suicide. I've always wanted to fix my issues, but I never get around to it. How can I change my feeling of being worthless to everyone? \r\n\r\n### Response:"
        )
inputs = alpaca_tokenizer(input_text, return_tensors="pt")
out = alpaca_model.generate(inputs=inputs.input_ids, max_new_tokens=256)
output_text = alpaca_tokenizer.batch_decode(out, skip_special_tokens=True)[0]
output_text = output_text[len(input_text) :]
print(f"Input: {input_text}\nCompletion: {output_text}")

