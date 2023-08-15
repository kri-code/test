# -*- coding: utf-8 -*-
"""mpt.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zHG7H2z-HhSqySe5gSoShRqSlIqfBVe-
"""

import torch
import transformers
import einops

name = 'mosaicml/mpt-7b-chat'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'triton'
config.init_device = 'cuda:0' # For fast initialization directly on GPU!

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

import torch
from transformers import pipeline

pipe = pipeline('text-generation',
                model=model,
                tokenizer=tokenizer,
                device='cuda:0',
                stopping_criteria=stopping_criteria,  # without this model will ramble
                temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                top_p=0.15,  # select from top tokens whose probability add up to 15%
                top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
                max_new_tokens=100,  # mex number of tokens to generate in the output
                repetition_penalty=1.1  # without this output begins repeating
                )

with open(f'dataset.txt', 'r') as fp:
            dataset = [l.strip() for l in fp.readlines()]

res = []
for inst in dataset:
  print(inst)
  a = pipe(inst)
  res.append(a[0]["generated_text"].strip())
with open(f'ncm_mptChat.txt', 'wb') as fp:
  for r in res:
    fp.write(r.encode("utf-8") + "\n".encode("utf-8"))
