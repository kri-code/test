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

# we create a list of stopping criteria
stop_token_ids = [
    tokenizer.convert_tokens_to_ids(x) for x in [
        ['Human', ':'], ['AI', ':']
    ]
]

stop_token_ids = [torch.LongTensor(x).to('cuda:0') for x in stop_token_ids]

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
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
                max_new_tokens=70,  # mex number of tokens to generate in the output
                repetition_penalty=1.1  # without this output begins repeating
                )

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# template for an instruction with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}"
)

llm = HuggingFacePipeline(pipeline=pipe)

llm_chain = LLMChain(llm=llm, prompt=prompt)


from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# initializing the conversational memory
memory = ConversationBufferWindowMemory(
    memory_key="history",  # important to align with agent prompt (below)
    k=5,
    return_only_outputs=True
)

from langchain.chains import ConversationChain

# initialize the agent
chat = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# The default prompt template will cause the model to return longer text --> modify it to be more concise
chat.prompt.template = \
"""The following is a friendly conversation between a human and an AI. The AI is conversational but concise in its responses without rambling. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:"""

def chat_trim(chat_chain, query):
    # create response
    chat_chain.predict(input=query)
    # check for double newlines (also happens often)
    chat.memory.chat_memory.messages[-1].content = chat.memory.chat_memory.messages[-1].content.split('\n\n')[0]
    # strip any whitespace
    chat.memory.chat_memory.messages[-1].content = chat.memory.chat_memory.messages[-1].content.strip()
    # check for stop text at end of output
    for stop_text in ['Human:', 'AI:', '[]']:
        chat.memory.chat_memory.messages[-1].content = chat.memory.chat_memory.messages[-1].content.removesuffix(stop_text)
    # strip again
    chat.memory.chat_memory.messages[-1].content = chat.memory.chat_memory.messages[-1].content.strip()
    # return final response
    return chat_chain.memory.chat_memory.messages[-1].content

print(chat_trim(chat, 'what is 2 + 2?'))


"""
with open('dataset.txt', 'r') as fp:
    dataset = [l.strip() for l in fp.readlines()]

res = []
for inst in dataset:
  print(inst)
  ans = chat_trim(chat, inst)
  print("ANSW:", ans)
  res.append(ans)
  #a = pipe(inst)
  #res.append(a[0]["generated_text"].strip())
with open('ncm_mptChat.txt', 'w') as fp:
  for r in res:
    fp.write(r + "\n")"""
