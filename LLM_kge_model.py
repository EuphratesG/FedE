import torch
import torch.nn as nn
from transformers import LlamaForCausalLM,LlamaModel,AutoTokenizer,AutoModelForCausalLM
import re


class LLMKGEModel(nn.Module):
    def __init__(self, args, LLM_model_name):
        super(LLMKGEModel, self).__init__()
        self.gpu=args.gpu
        self.model = AutoModelForCausalLM.from_pretrained(LLM_model_name).half()
        self.model = self.model.to(args.gpu)
        self.model_name = LLM_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.gamma = nn.Parameter(
            torch.Tensor([args.gamma]),
            requires_grad=False
        )
        # 读取文本文件
        file_path = 'FB12k237/entityDescriptionLong.txt'  # 替换为你的文件路径
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        self.entity_description = [line.strip().split('\t')[1] for line in lines]

        file_path = 'FB12k237/entityDescription.txt'  # 替换为你的文件路径
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        self.entity_text = [line.strip().split(', ')[1] for line in lines]

        # 读取文本文件
        file_path = 'FB12k237/relationDescription.txt'  # 替换为你的文件路径
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        self.relation_text = [line.strip().split(', ')[1] for line in lines]

#forward就是实体输入参数之后得到一个返回值score
    def forward(self, sample, neg=True):
        if not neg:#无负例情况
            head = [self.entity_text[index[0]] for index in sample]

            relation = [self.relation_text[index[1]] for index in sample]

            tail = [self.entity_text[index[2]] for index in sample]
        else:# 这里是默认，有负例情况         
            head_part, tail_part = sample
            batch_size = head_part.shape[0] # [512,3]

            head = [self.entity_text[index[0]] for index in head_part]
            head_description = [self.entity_description[index[0]] for index in head_part]
            #print("\n")
            #print(head.shape)
            relation = [self.relation_text[index[1]] for index in head_part]

            #LLM是不训练的因此不要下面的了 
            if tail_part == None:
                tail = self.entity_text #[14541]
                tail_description = self.entity_description
            
        model_func = {
            'Llama2': self.Llama2
        }

        score = model_func['Llama2'](head, head_description, relation, tail, tail_description)
        
        return score
    

    def Llama2(self, head, head_description, relation, tail, tail_description):
        # print(len(head)) 16
        # print(head)
        # print(len(relation)) 16
        # print(len(tail)) 14541
        # 初始化一个空的大小为[16, 14541]的列表
        score_list = [[] for _ in range(len(head))]
        #prompt = f"Please give me the likelihood of the triple \"Person:{head}, Relation:{relation}, Entity:{tail}\" belonging to the knowledge graph."
        for i in range(len(head)):
            for j in range(len(tail)):
                prompt = f"<s>[INST] <<SYS>>\
You are a well-pretrained LLM, possesses a strong ability to analyze the relation between entities.\
<</SYS>> \
I will describe a triplet in the following format: Entity A: xxx, Description: xxx;\n Entity B: xxx, Description: xxx;\n Relation: xxx.\n xxx will be the corresponding texts. \
I want you to give me a numerical value expressing the likelihood that entity A and Entity B establish a directional logical relationship through an intermediary relation to form a plausible triple in a knowledge graph.\n \
The triplet is as follows :\n \
\"Entity A: {head[i]}, Description: {head_description[i]};\n Entity B: {tail[j]}, Description: {tail_description[j]};\n Relation: {relation[i]}.\n\"  You should divide the scale from 0 to 1 into five parts and provide scores with finer granularity, like: \n \
0.0000 - 0.1999: Extremely unlikely to occur, almost impossible.\n \
0.2000 - 0.3999: Very unlikely to occur, with a small chance of happening.\n \
0.4000 - 0.5999: Uncertain, neither likely nor unlikely.\n \
0.6000 - 0.7999: Likely to occur, with a fair chance of happening.\n \
0.8000 - 1.0000: Highly likely to occur, with a strong probability of happening.\n \
Present your answer using this format: \"value of a plausible triple's likelihood = 0.XXXX\" where 0.XXXX is a value between 0 and 1. \
Ensure that the numerical answer you provide has the same precision as the example format. Emphasizing, the factor you refer to must be the relation I provided. If in your view, there is almost no relation between Entity A and Entity B as provided, you may give a low score near 0. If you believe there is a strong logical relationship from Entity A to Entity B, you may give a high score near 1. [/INST]"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.gpu)
                #inputs = inputs.half()
                tokens = self.model.generate(inputs["input_ids"], max_length=4096, do_sample=True, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id, temperature=0.9)
                generated_text = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
                print(generated_text)
                matches = re.findall(r'\d+\.\d+', generated_text)

                score = float(matches[0])
                #print(score)
                print('\n')
                score_list[i].append(score)
        return score_list
