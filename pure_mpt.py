# -*- coding: utf-8 -*-
"""mpt.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zHG7H2z-HhSqySe5gSoShRqSlIqfBVe-
"""
import torch
import transformers
import einops
from transformers import StoppingCriteria, StoppingCriteriaList

name = 'mosaicml/mpt-7b-chat'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.init_device = 'cuda:0' # For fast initialization directly on GPU!

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

import torch
from transformers import pipeline

stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

# Define a custom stopping criteria
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


stopping_criteria = StoppingCriteriaList([StopOnTokens()])

pipe = pipeline(task = 'text-generation',
                model=model,
                return_full_text=True,
                tokenizer=tokenizer,
                device='cuda:0',
                stopping_criteria=stopping_criteria,  # without this model will ramble
                temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                top_p=0.15,  # select from top tokens whose probability add up to 15%
                top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
                max_new_tokens=100,  # mex number of tokens to generate in the output
                repetition_penalty=1.1  # without this output begins repeating
                )


with open('dataset.txt', 'r') as fp:
    dataset = [l.strip() for l in fp.readlines()]

res = []
for inst in dataset:
  print(inst)
  generate_text = pipe(inst)
  print("---------------------------------------------")
  print(generate_text[0]["generated_text"])
  res.append(generate_text[0]["generated_text"])
  #a = pipe(inst)
  #res.append(a[0]["generated_text"].strip())
with open('pure_mptChat.txt', 'w') as fp:
  for r in res:
    fp.write(r + "\n")