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
gpu_num = "6"
gpu = torch.device('cuda:' + gpu_num)
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").half()
model = model.to(gpu)
print(model)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token

# 读取文本文件
file_path = '/home/yvhe/FedE/umls/entity2text.txt' 
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
entity_names = [line.strip().split(None, 1)[1] for line in lines]
print(entity_names)



entity_names_with_special_tokens = [name for name in entity_names]
# 编码实体名称
encoded_inputs = tokenizer(entity_names_with_special_tokens,padding=True, return_tensors="pt").to(gpu)
# [    1,   518,  3919, 11937, 29918, 25826, 29962, 26808,   487,  4925,
#           5960,   749,   470,  4768,  6352,   293,   626,   457,   518,  3919,
#          11937, 29918, 11794, 29962]
print(encoded_inputs)
print(encoded_inputs['input_ids'].shape)

# Traditional way of generating text
# outputs = model.generate(input_ids = encoded_inputs['input_ids'])
# print("\ngenerate + input_ids:", tokenizer.decode(outputs[0], skip_special_tokens=True))



inputs_embeds = model.model.embed_tokens(encoded_inputs['input_ids']).to(gpu)

attention_mask = encoded_inputs["attention_mask"]
print(attention_mask.unsqueeze(-1))
# 对每个实体的隐藏状态进行平均池化，排除填充的 token
masked_inputs_embeds = inputs_embeds * attention_mask.unsqueeze(-1)  # 将填充位置的隐藏状态置为0
masked_inputs_embeds = masked_inputs_embeds.to('cpu')
print(masked_inputs_embeds)
print(masked_inputs_embeds.shape)
# 存储张量到文件
torch.save(masked_inputs_embeds, '510entity_embeddings.pt')
# outputs = model.generate(inputs_embeds=inputs_embeds)
# print("\ngenerate + inputs_embeds:", tokenizer.decode(outputs[0], skip_special_tokens=True))
