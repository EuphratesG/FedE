#之前model forward的时候越界就是32000 padding的问题

import os
from openai import OpenAI
import requests
import time
import json
import time
from transformers import LlamaForCausalLM,LlamaModel,AutoTokenizer,AutoModelForCausalLM
import torch


#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
gpu_num = "7"
gpu = torch.device('cuda:' + gpu_num)
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").half()
model = model.to(gpu)
print(model)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token


text = "Hello world"
input_ids = tokenizer.encode("human", return_tensors="pt").to(gpu)

# Traditional way of generating text
outputs = model.generate(input_ids)
print("\ngenerate + input_ids:", tokenizer.decode(outputs[0], skip_special_tokens=True))


# # 从文件加载张量
# loaded_entity_embeddings = torch.load('entity_embeddings.pt').unsqueeze(1)
# loaded_entity_embeddings = loaded_entity_embeddings[:2]  # 切片操作，获取第一个维度的前两个元素
# print(loaded_entity_embeddings.shape)
# From inputs_embeds -- exact same output if you also pass `input_ids`. If you don't
# pass `input_ids`, you will get the same generated content but without the prompt
inputs_embeds = model.model.embed_tokens(input_ids).to(gpu)
print(inputs_embeds.shape)
outputs = model.generate(inputs_embeds=inputs_embeds)
print("\ngenerate + inputs_embeds:", tokenizer.decode(outputs[0], skip_special_tokens=True))