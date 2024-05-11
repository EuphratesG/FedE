import torch
import torch.nn as nn
from transformers import LlamaForCausalLM,LlamaModel,AutoTokenizer,AutoModelForCausalLM
import re
import together
import os
import anthropic
import time
import multiprocessing
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
import concurrent.futures
from openai import OpenAI


class LLMKGEModel():#nn.Module):
    def __init__(self, args, LLM_model_name):
        super(LLMKGEModel, self).__init__()
        #multiprocessing.set_start_method('spawn')
        self.gpu=args.gpu
        if(args.LLMModel=="llama2"):
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").half()
            self.model = self.model.to(args.gpu)
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        elif(args.LLMModel=="together_ai") or (args.LLMModel=="single_test"):
            self.together_ai_model = args.together_ai_model

        self.LLMModel = args.LLMModel
        self.num_triplets = args.num_triplets
        self.num_threads = args.num_threads        

        # # 读取文本文件
        # file_path = '/home/yvhe/FedE/fb15k-237/entity2textlong_sample.txt'  # 替换为你的文件路径
        # with open(file_path, 'r', encoding='utf-8') as file:
        #     lines = file.readlines()
        # self.entity_description = [line.strip().split(None, 1)[1] for line in lines]

        # file_path = '/home/yvhe/FedE/fb15k-237/entity2text_sample.txt'  # 替换为你的文件路径
        # with open(file_path, 'r', encoding='utf-8') as file:
        #     lines = file.readlines()
        # self.entity_text = [line.strip().split(None, 1)[1] for line in lines]

        # # 读取文本文件
        # file_path = '/home/yvhe/FedE/fb15k-237/relation2text.txt'  # 替换为你的文件路径
        # with open(file_path, 'r', encoding='utf-8') as file:
        #     lines = file.readlines()
        # self.relation_text = [line.strip().split(None, 1)[1] for line in lines]

        # 读取文本文件
        file_path = '/home/yvhe/FedE/umls/entity2textlong.txt'  # 替换为你的文件路径
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        self.entity_description = [line.strip().split(None, 1)[1] for line in lines]
        # 读取文本文件
        file_path = '/home/yvhe/FedE/umls/entity2text.txt'  # 替换为你的文件路径
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        self.entity_text = [line.strip().split(None, 1)[1] for line in lines]
        # 读取文本文件
        file_path = '/home/yvhe/FedE/umls/relation2text.txt'  # 替换为你的文件路径
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        self.relation_text = [line.strip().split(None, 1)[1] for line in lines]







    def forward_PRAUC(self, sample, one2n, all_neighbors):
        head_part = sample
        batch_size = head_part.shape[0] # [512,3]
        head = [self.entity_text[index[0]] for index in head_part]
        head_description = [self.entity_description[index[0]] for index in head_part]
        #print("\n")
        #print(head.shape)
        relation = [self.relation_text[index[1]] for index in head_part]
        # print(relation[36])
        # print(relation[36])
        # print(relation[20])
        tail = [self.entity_text[index[2]] for index in head_part]
        tail_description = [self.entity_description[index[2]] for index in head_part]
        # 使用字典将每个 set 中的索引替换为对应的值
        one2n_text = [{self.entity_text[idx] for idx in s} for s in one2n]
        # 使用字典将每个子列表中的索引替换为对应的值
        all_neighbors_text = [[(self.relation_text[index1], self.entity_text[index2]) for index1, index2 in sublist] for sublist in all_neighbors]


        model_func = {
            'together_ai': self.together_ai_PRAUC
        }

        score = model_func[self.LLMModel](head, head_description, relation, tail, tail_description, one2n_text, all_neighbors_text)
        
        return score



    def together_ai_PRAUC(self, head, head_description, relation, tail, tail_description, one2n_text, all_neighbors_text):
        num_threads = self.num_threads  # 根据需要调整线程数量
        chunk_size = len(head) // num_threads
        remainder = len(head) % num_threads

        # 提交任务给线程池
        results = []
        start_index = 0
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(num_threads):
                # 计算当前线程的结束索引
                end_index = start_index + chunk_size
                if remainder > 0:
                    end_index += 1
                    remainder -= 1

                # 提交任务
                result = executor.submit(self.process_triplet_together_ai_PRAUC, head, head_description, tail, tail_description, relation, start_index, end_index, one2n_text, all_neighbors_text)
                results.append((start_index, result))  # 保存起始索引和结果的对应关系

                # 更新下一个线程的开始索引
                start_index = end_index

            executor.shutdown(wait=True)

        # 根据线程启动的顺序获取结果
        score_list = []
        for start_index, future in sorted(results, key=lambda x: x[0]):
            chunk_score = future.result()
            score_list.extend(chunk_score)

        return score_list


    def process_triplet_together_ai_PRAUC(self, head, head_description, tail, tail_description, relation, start_index, end_index, one2n_text, all_neighbors_text):
        print(f"Thread {threading.current_thread().name} is working")
        chunk_score = []
        TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

        client = OpenAI(
        api_key=TOGETHER_API_KEY,
        base_url='https://api.together.xyz/v1',
        )
        for i in range(start_index, end_index):
            row_scores = []
            prompt_to_add = ""
            if len(one2n_text[i]) >0:
                for k in range(len(one2n_text[i])):
                    #print(len(one2n_text[i]))
                    one2n_text_list = list(one2n_text[i])
                    prompt_to_add += f"correct triplet : Subject Entity A: {head[i]} ; \
Object Entity B: {one2n_text_list[k]} ; \
Predicate Relation: {relation[i]}.\n"
            if len(all_neighbors_text[i]) > 0:
                for k in range(len(all_neighbors_text[i])):
                    prompt_to_add += f"correct triplet : Subject Entity A: {head[i]} ; \
Object Entity B: {all_neighbors_text[i][k][1]} ; \
Predicate Relation: {all_neighbors_text[i][k][0]}.\n"    
                             
            retry = True  # 是否重试的标志
            while retry:  # 循环重试直到请求成功
                try:
                    start_time = time.time()

                    # 初始化 prompt
                    prompt = f"I would like you to handle this task for me : For the triplet I will provide shortly, estimate the likelihood that the subject entity A and the object entity B are connected by a predicate relation, forming a plausible true triplet in a knowledge graph. \
I will also provide some correct triplets from the real knowledge graph as a judgement factor for your score. These correct triplets, sharing the same Subject Entity A as the triplet you need to score, can assist your judgment through this proximity relationship. \
I will give you the triplet in this following format: \n\"target triplet : Subject Entity A: xxx ;\nObject Entity B: xxx ;\nPredicate Relation: xxx.\"\n\
And the correct triplets will be provided like this: \"correct triplet : Subject Entity A: xxx ; Object Entity B: xxx ; Predicate Relation: xxx.\"\n\
xxx will be the corresponding texts. \
Now the triplet is as follows :\n\n"
                    # The entities may consist of one or more words, and you should avoid confusing similar entities like \"human\" and \"human caused phenomenon or process\".\
                    prompt += f"\"target triplet: Subject Entity A: {head[i]} ;\n\
Object Entity B: {tail[i]} ;\n\
Predicate Relation: {relation[i]}.\n\n"

                    prompt += prompt_to_add
                    # 添加后续指示
                    prompt += f"\"\n\nYou should divide the scale from 0 to 1 into five parts and provide scores with fine granularity, like:\n\
0.0000 - 0.1999: Extremely unlikely to occur, almost impossible.\n\
0.2000 - 0.3999: Very unlikely to occur, with a small chance of happening.\n\
0.4000 - 0.5999: Uncertain, neither likely nor unlikely.\n\
0.6000 - 0.7999: Likely to occur, with a fair chance of happening.\n\
0.8000 - 1.0000: Highly likely to occur, with a strong probability of happening.\n\
Present your answer in the format: 'value of the target triplet's likelihood = 0.XXXX, Object Entity: XXXX' where 0.XXXX is a value between 0 and 1 with four decimal places, and 'Object Entity: XXXX' represents the Object Entity B provided.\n\
Ensure that your response strictly adheres to the format provided above: 'value of the target triplet's likelihood = 0.XXXX, Object Entity: XXXX'. Only one row, no additional explanation."


                    print(prompt)
                    # # 打开文件并写入内容
                    # with open("测试prompt.txt", "w") as file:
                    #     file.write(prompt)
                    while True:
                        chat_completion = client.chat.completions.create(
                        messages=[
                            {
                            "role": "system",
                            "content": "You are a well-pretrained LLM, possesses a strong ability to analyze the relation between entities.",
                            },
                            {
                            "role": "user",
                            "content": prompt
                            }
                        ],
                        model=self.together_ai_model
                        )

                        # parse the completion then print the whole output
                        generatedText = chat_completion.choices[0].message.content
                        print(generatedText)
                        #print('-----------------------------------------------------------------------------------------')
                        # 使用正则表达式在文本中搜索包含特定句子的部分
                        likelihoods_str = re.findall(r'likelihood = (\d+\.\d+)', generatedText)
                        #likelihoods_str += re.findall(r'is (\d+\.\d+)', generatedText)
                        score = [float(likelihood) for likelihood in likelihoods_str]
                        print(f"当前i为{i},有{len(score)}个数据")
                        if len(score) > 1:
                            score = [score[0]]
                            break
                        elif len(score) == 1:
                            break
                    print(score)
                    print('\n')
                    end_time = time.time()
                    iteration_time = end_time - start_time
                    print("循环执行时间:", iteration_time, "秒")
                    row_scores.extend(score)
                    retry = False  # 成功获取响应后退出重试循环
                except Exception as e:#requests.exceptions.HTTPError as e:
                    print(f"Thread {threading.current_thread().name} is working")
                    print(e)
                    continue
            chunk_score.extend(row_scores)
        print("hhhhhhhhhhhhhhhhhh")
        print(len(chunk_score))
        return chunk_score




#forward就是实体输入参数之后得到一个返回值score
    def forward(self, sample, one2n, all_neighbors, h2rt_all,neg=True):
        #print(len(one2n))
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
            # print(relation[36])
            # print(relation[36])
            # print(relation[20])
            #LLM是不训练的因此不要下面的了 
            if tail_part == None:
                tail = self.entity_text #[14541]
                tail_description = self.entity_description
            # 使用字典将每个 set 中的索引替换为对应的值
            one2n_text = [{self.entity_text[idx] for idx in s} for s in one2n]
            # 使用字典将每个子列表中的索引替换为对应的值
            all_neighbors_text = [[(self.relation_text[index1], self.entity_text[index2]) for index1, index2 in sublist] for sublist in all_neighbors]
            h2rt_all_text = {key: [(self.relation_text[index1], self.entity_text[index2]) for index1, index2 in value] for key, value in h2rt_all.items()}
            #print(all_neighbors_text)

            #print(len(one2n_text)) 650个集合
            # print(tail[57])
            # print(tail[57])
            # print(tail[57])
        model_func = {
            'llama2': self.llama2,
            'together_ai': self.together_ai,
            'claude': self.claude,
            'single_test': self.single_test
        }

        score = model_func[self.LLMModel](head, head_description, relation, tail, tail_description, one2n_text, all_neighbors_text,h2rt_all_text)
        
        return score











    def single_test(self, head, head_description, relation, tail, tail_description, one2n_text, all_neighbors_text):
        together.api_key = os.environ.get("TOGETHER_API_KEY") # Replace with your Together API Key
        num_triplets_this = self.num_triplets
        retry = True  # 是否重试的标志
        while retry:  # 循环重试直到请求成功
            start_time = time.time()
            # 初始化 prompt
            prompt = f"<s>[INST] <<SYS>>You are a well-pretrained LLM, possesses a strong ability to analyze the relation between entities.<</SYS>>\n \
I would like you to handle this task for me : For each triplet I give you, provide me a numerical value, expressing the likelihood that subject entity A and object entity B are linked by a predicate relation to form a plausible triplet in some knowledge graphs.\
I will give you {num_triplets_this} triplets in the following format: \"triplet 1 : Subject Entity A: xxx, Description: xxx;\nObject Entity B: xxx, Description: xxx;\nPredicate Relation: xxx.\"\nxxx will be the corresponding texts.\
The Subject Entity A and the Predicate Relation are fixed, while the Object Entity B varies with the triplet index. Therefore, I will only provide the description of Subject Entity A in the first triplet.\n \
The entities may consist of one or more words, and you should avoid confusing similar entities like \"human\" and \"human caused phenomenon or process\".\
Now the triplets are as follows :\n\n"

            # 循环生成三元组
            for k in range(num_triplets_this):
                if k==0 :
                    prompt += f"triplet {k+1} : Subject Entity A: {tail[57]}, Description: {tail_description[57]}\n\
Object Entity B: {tail[46]}, Description: {tail_description[46]}\n\
Predicate Relation: {relation[1]}.\n\n"
                else :
                    prompt += f"triplet {k+1} : Subject Entity A: {tail[57]}, Description: Refer to triplet 1\n\
Object Entity B: {tail[46+k]}, Description: {tail_description[46+k]}\n\
Predicate Relation: {relation[1]}.\n\n"

            #这里别漏了
            #j+=num_triplets
            # 添加后续指示
            prompt += f"You should divide the scale from 0 to 1 into five parts and provide scores with fine granularity, like:\n\
0.0000 - 0.1999: Extremely unlikely to occur, almost impossible.\n \
0.2000 - 0.3999: Very unlikely to occur, with a small chance of happening.\n \
0.4000 - 0.5999: Uncertain, neither likely nor unlikely.\n \
0.6000 - 0.7999: Likely to occur, with a fair chance of happening.\n \
0.8000 - 1.0000: Highly likely to occur, with a strong probability of happening.\n \
Emphasizing, the factor you refer to must be the predicate relation I provided, and the relation between the two entities is an unidirectional predicate, with subject entity A pointing to object entity B.\
Present your answer using this format: \"value of the 1st(2nd and so on, until reach {num_triplets_this}) triplet's likelihood = 0.XXXX, Object Entity:XXXX \" where 0.XXXX is a value between 0 and 1 with four decimal places, and Object Entity :XXXX means the Object Entity B I give you.\n\
Ensure that your response for each corresponding triplet strictly adheres to the format provided above, and also ensure that the number of answers matches the quantity I requested, one answer per row. No extra explanation about your answers. [/INST]"



            print(prompt)
            # # 打开文件并写入内容
            # with open("测试prompt.txt", "w") as file:
            #     file.write(prompt)
            while True:
                output = together.Complete.create(
                    prompt=prompt,
                    model=self.together_ai_model,
                    max_tokens=4096,
                    temperature=0.8,
                )

                # parse the completion then print the whole output
                generatedText = output['output']['choices'][0]['text']
                print(generatedText)
                #print('-----------------------------------------------------------------------------------------')
                # 使用正则表达式在文本中搜索包含特定句子的部分
                likelihoods_str = re.findall(r'likelihood = (\d+\.\d+)', generatedText)
                score = [float(likelihood) for likelihood in likelihoods_str]
                #print(f"当前i为{i},j为{j},有{len(score)}个数据")
                if len(score) == num_triplets_this:
                    break
            print(score)
            print('\n')
            end_time = time.time()
            iteration_time = end_time - start_time
            print("循环执行时间:", iteration_time, "秒")
            retry = False  # 成功获取响应后退出重试循环
        # except Exception as e:#requests.exceptions.HTTPError as e:
        #     print(f"Thread {threading.current_thread().name} is working")
        #     print(e)
        #     continue








    def together_ai(self, head, head_description, relation, tail, tail_description, one2n_text, all_neighbors_text,h2rt_all_text):
        num_threads = self.num_threads  # 根据需要调整线程数量
        chunk_size = len(head) // num_threads
        remainder = len(head) % num_threads

        # 提交任务给线程池
        results = []
        start_index = 0
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(num_threads):
                # 计算当前线程的结束索引
                end_index = start_index + chunk_size
                if remainder > 0:
                    end_index += 1
                    remainder -= 1

                # 提交任务
                result = executor.submit(self.process_triplet_together_ai, head, head_description, tail, tail_description, relation, start_index, end_index, one2n_text, all_neighbors_text,h2rt_all_text)
                results.append((start_index, result))  # 保存起始索引和结果的对应关系

                # 更新下一个线程的开始索引
                start_index = end_index

            executor.shutdown(wait=True)

        # 根据线程启动的顺序获取结果
        score_list = []
        for start_index, future in sorted(results, key=lambda x: x[0]):
            chunk_score = future.result()
            for row_score in chunk_score:
                score_list.append(row_score)

        return score_list


# 这是用openAI api的版本 每次三个实体
#     def process_triplet_together_ai(self, head, head_description, tail, tail_description, relation, start_index, end_index):
#         print(f"Thread {threading.current_thread().name} is working")
#         chunk_score = []
#         TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

#         client = OpenAI(
#         api_key=TOGETHER_API_KEY,
#         base_url='https://api.together.xyz/v1',
#         )
#         for i in range(start_index, end_index):
#             row_scores = []
#             for j in range(0, len(tail),self.num_triplets):
#                 num_triplets_this = min(self.num_triplets, len(tail) - j)
#                 retry = True  # 是否重试的标志
#                 while retry:  # 循环重试直到请求成功
#                     try:
#                         start_time = time.time()

#                         # 初始化 prompt
#                         prompt = f"<s>[INST] <<SYS>>You are a well-pretrained LLM, possesses a strong ability to analyze the relation between entities.<</SYS>>\n \
# I would like you to handle this task for me : For each triplet I will provide shortly, estimate the likelihood that the subject entity A and the object entity B are connected by a predicate relation, forming a plausible true triplet in some knowledge graph.\
# I will give you {num_triplets_this} triplets in the following format: \"triplet 1 : Subject Entity A: xxx, Description: xxx;\nObject Entity B: xxx, Description: xxx;\nPredicate Relation: xxx.\"\nxxx will be the corresponding texts.\
# The Subject Entity A and the Predicate Relation are fixed, while the Object Entity B varies with the triplet index. Therefore, I will only provide the description of Subject Entity A in the first triplet.\n \
# The entities may consist of one or more words, and you should avoid confusing similar entities like \"human\" and \"human caused phenomenon or process\".\
# Now the triplets are as follows :\n\n"

#                         # 循环生成三元组
#                         for k in range(num_triplets_this):
#                             if k==0 :
#                                 prompt += f"triplet {k+1} : Subject Entity A: {head[i]}, Description: {head_description[i]}\n\
# Object Entity B: {tail[j+k]}, Description: {tail_description[j+k]}\n\
# Predicate Relation: {relation[i]}.\n\n"
#                             else :
#                                 prompt += f"triplet {k+1} : Subject Entity A: {head[i]}, Description: Refer to triplet 1\n\
# Object Entity B: {tail[j+k]}, Description: {tail_description[j+k]}\n\
# Predicate Relation: {relation[i]}.\n\n"

#                         #这里别漏了
#                         #j+=num_triplets
#                         # 添加后续指示
#                         prompt += f"You should divide the scale from 0 to 1 into five parts and provide scores with fine granularity, like:\n\
# 0.0000 - 0.1999: Extremely unlikely to occur, almost impossible.\n \
# 0.2000 - 0.3999: Very unlikely to occur, with a small chance of happening.\n \
# 0.4000 - 0.5999: Uncertain, neither likely nor unlikely.\n \
# 0.6000 - 0.7999: Likely to occur, with a fair chance of happening.\n \
# 0.8000 - 1.0000: Highly likely to occur, with a strong probability of happening.\n \
# Emphasizing, the factor you refer to must be the predicate relation I provided, and the relation between the two entities is an unidirectional predicate, with subject entity A pointing to object entity B.\
# Present your answer using this format: \"value of the 1st(2nd and so on, until reach {num_triplets_this}) triplet's likelihood = 0.XXXX, Object Entity:XXXX \" where 0.XXXX is a value between 0 and 1 with four decimal places, and Object Entity :XXXX means the Object Entity B I give you.\n\
# Ensure that your response for each corresponding triplet strictly adheres to the format provided above, and also ensure that the number of answers matches the quantity I requested, one answer per row. No extra explanation about your answers. [/INST]"



#                         #print(prompt)
#                         # # 打开文件并写入内容
#                         # with open("测试prompt.txt", "w") as file:
#                         #     file.write(prompt)
#                         while True:
#                             chat_completion = client.chat.completions.create(
#                             messages=[
#                                 {
#                                 "role": "system",
#                                 "content": "You are a well-pretrained LLM, possesses a strong ability to analyze the relation between entities.",
#                                 },
#                                 {
#                                 "role": "user",
#                                 "content": prompt
#                                 }
#                             ],
#                             model="google/gemma-7b-it"
#                             )

#                             # parse the completion then print the whole output
#                             generatedText = chat_completion.choices[0].message.content
#                             print(generatedText)
#                             #print('-----------------------------------------------------------------------------------------')
#                             # 使用正则表达式在文本中搜索包含特定句子的部分
#                             likelihoods_str = re.findall(r'likelihood = (\d+\.\d+)', generatedText)
#                             score = [float(likelihood) for likelihood in likelihoods_str]
#                             print(f"当前i为{i},j为{j},有{len(score)}个数据")
#                             if len(score) == num_triplets_this:
#                                 break
#                         print(score)
#                         print('\n')
#                         end_time = time.time()
#                         iteration_time = end_time - start_time
#                         print("循环执行时间:", iteration_time, "秒")
#                         row_scores.extend(score)
#                         retry = False  # 成功获取响应后退出重试循环
#                     except Exception as e:#requests.exceptions.HTTPError as e:
#                         print(f"Thread {threading.current_thread().name} is working")
#                         print(e)
#                         continue
#             chunk_score.append(row_scores)
#         print("hhhhhhhhhhhhhhhhhh")
#         print(len(chunk_score))
#         return chunk_score




# # 这是用together api的版本 先改成每次一个实体试试
#     def process_triplet_together_ai(self, head, head_description, tail, tail_description, relation, start_index, end_index):
#         print(f"Thread {threading.current_thread().name} is working")
#         chunk_score = []
#         together.api_key = os.environ.get("TOGETHER_API_KEY") # Replace with your Together API Key
#         for i in range(start_index, end_index):
#             row_scores = []
#             for j in range(0, len(tail)):
#                 #num_triplets_this = min(self.num_triplets, len(tail) - j)
#                 retry = True  # 是否重试的标志
#                 while retry:  # 循环重试直到请求成功
#                     try:
#                         start_time = time.time()

#                         # 初始化 prompt
#                         prompt = f"<s>[INST] <<SYS>>You are a well-pretrained LLM, possesses a strong ability to analyze the relation between entities.<</SYS>>\n \
# I would like you to handle this task for me : For the triplet I will provide shortly, estimate the likelihood that the subject entity A and the object entity B are connected by a predicate relation, forming a plausible true triplet in some knowledge graph.\
# I will give you the triplet in the following format: \"triplet : \nSubject Entity A: xxx, Description: xxx;\nObject Entity B: xxx, Description: xxx;\nPredicate Relation: xxx.\"\nxxx will be the corresponding texts.\
# The entities may consist of one or more words, and you should avoid confusing similar entities like \"human\" and \"human caused phenomenon or process\".\
# Now the triplet is as follows :\n\n"
#                         # 循环生成三元组
#                         prompt += f"\"triplet : \nSubject Entity A: {head[i]}, Description: {head_description[i]};\n\
# Object Entity B: {tail[j]}, Description: {tail_description[j]};\n\
# Predicate Relation: {relation[i]}.\"\n\n"

#                         # 添加后续指示
#                         prompt += f"Provide the score with fine granularity on a scale from 0 to 1, divided into five parts:\n\
# 0.0000 - 0.1999: Extremely unlikely to occur, almost impossible.\n \
# 0.2000 - 0.3999: Very unlikely to occur, with a small chance of happening.\n \
# 0.4000 - 0.5999: Uncertain, neither likely nor unlikely.\n \
# 0.6000 - 0.7999: Likely to occur, with a fair chance of happening.\n \
# 0.8000 - 1.0000: Highly likely to occur, with a strong probability of happening.\n \
# The factor considered is the predicate relation provided, and the relation between the two entities is unidirectional, with subject entity A pointing to object entity B.\
# Present the answer in the format: 'value of the triplet's likelihood = 0.XXXX, Object Entity: XXXX' where 0.XXXX is a value between 0 and 1 with four decimal places, and 'Object Entity: XXXX' represents the Object Entity B provided.\n\
# Ensure that the response strictly adheres to the format provided above. No additional explanation about the answers. [/INST]"

# #Ensure that each response strictly adheres to the format provided above, and that the number of answers matches the quantity requested, with one answer per row. No additional explanation about the answers.

#                         #print(prompt)
#                         # # 打开文件并写入内容
#                         # with open("测试prompt.txt", "w") as file:
#                         #     file.write(prompt)
#                         while True:
#                             output = together.Complete.create(
#                                 prompt=prompt,
#                                 model=self.together_ai_model,
#                                 max_tokens=4096,
#                                 temperature=0.8,
#                             )

#                             # parse the completion then print the whole output
#                             generatedText = output['output']['choices'][0]['text']
#                             print(generatedText)
#                             #print('-----------------------------------------------------------------------------------------')
#                             # 使用正则表达式在文本中搜索包含特定句子的部分
#                             likelihoods_str = re.findall(r'likelihood = (\d+\.\d+)', generatedText)
#                             score = [[float(likelihood) for likelihood in likelihoods_str][0]]
#                             print(f"当前i为{i},j为{j},有{len(score)}个数据")
#                             if len(score) == 1:
#                                 break
#                         print(score)
#                         print('\n')
#                         end_time = time.time()
#                         iteration_time = end_time - start_time
#                         print("循环执行时间:", iteration_time, "秒")
#                         row_scores.extend(score)
#                         retry = False  # 成功获取响应后退出重试循环
#                     except Exception as e:#requests.exceptions.HTTPError as e:
#                         print(f"Thread {threading.current_thread().name} is working")
#                         print(e)
#                         continue
#             chunk_score.append(row_scores)
#         print("hhhhhhhhhhhhhhhhhh")
#         print(len(chunk_score))
#         return chunk_score
    



  
# #这是用openAI api的版本 先改成每次一个实体同时去掉description的情况试试
#     def process_triplet_together_ai(self, head, head_description, tail, tail_description, relation, start_index, end_index):
#         print(f"Thread {threading.current_thread().name} is working")
#         chunk_score = []
#         TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

#         client = OpenAI(
#         api_key=TOGETHER_API_KEY,
#         base_url='https://api.together.xyz/v1',
#         )
#         for i in range(start_index, end_index):
#             row_scores = []
#             for j in range(0, len(tail)):
#                 retry = True  # 是否重试的标志
#                 while retry:  # 循环重试直到请求成功
#                     try:
#                         start_time = time.time()

#                         # 初始化 prompt
#                         prompt = f"I would like you to handle this task for me : For the triplet I will provide shortly, estimate the likelihood that the subject entity A and the object entity B are connected by a predicate relation, forming a plausible true triplet in some knowledge graph.\
# I will give you the triplet in the following format: \"triplet : Subject Entity A: xxx ; Object Entity B: xxx ; Predicate Relation: xxx.\"\nxxx will be the corresponding texts.\
# Now the triplet is as follows :\n\n"
#                         # The entities may consist of one or more words, and you should avoid confusing similar entities like \"human\" and \"human caused phenomenon or process\".\
#                         # 循环生成三元组
#                         prompt += f"\"triplet : Subject Entity A: {head[i]} ; \
# Object Entity B: {tail[j]} ; \
# Predicate Relation: {relation[i]}.\"\n\n"

#                         # 添加后续指示
#                         prompt += f"Provide the score with fine granularity on a scale from 0 to 1, divided into five parts:\n\
# 0.0000 - 0.1999: Extremely unlikely to occur, almost impossible.\n \
# 0.2000 - 0.3999: Very unlikely to occur, with a small chance of happening.\n \
# 0.4000 - 0.5999: Uncertain, neither likely nor unlikely.\n \
# 0.6000 - 0.7999: Likely to occur, with a fair chance of happening.\n \
# 0.8000 - 1.0000: Highly likely to occur, with a strong probability of happening.\n \
# The factor considered is the predicate relation provided, and the relation between the two entities is unidirectional, with subject entity A pointing to object entity B.\
# Present the answer in the format: 'value of the triplet's likelihood = 0.XXXX, Object Entity: XXXX' where 0.XXXX is a value between 0 and 1 with four decimal places, and 'Object Entity: XXXX' represents the Object Entity B provided.\n\
# Ensure that the response strictly adheres to the format provided above. No additional explanation about your answer."


#                         #print(prompt)
#                         # # 打开文件并写入内容
#                         # with open("测试prompt.txt", "w") as file:
#                         #     file.write(prompt)
#                         while True:
#                             chat_completion = client.chat.completions.create(
#                             messages=[
#                                 {
#                                 "role": "system",
#                                 "content": "You are a well-pretrained LLM, possesses a strong ability to analyze the relation between entities.",
#                                 },
#                                 {
#                                 "role": "user",
#                                 "content": prompt
#                                 }
#                             ],
#                             model=self.together_ai_model
#                             )

#                             # parse the completion then print the whole output
#                             generatedText = chat_completion.choices[0].message.content
#                             print(generatedText)
#                             #print('-----------------------------------------------------------------------------------------')
#                             # 使用正则表达式在文本中搜索包含特定句子的部分
#                             likelihoods_str = re.findall(r'likelihood = (\d+\.\d+)', generatedText)
#                             score = [[float(likelihood) for likelihood in likelihoods_str][0]]
#                             print(f"当前i为{i},j为{j},有{len(score)}个数据")
#                             if len(score) == 1:
#                                 break
#                         print(score)
#                         print('\n')
#                         end_time = time.time()
#                         iteration_time = end_time - start_time
#                         print("循环执行时间:", iteration_time, "秒")
#                         row_scores.extend(score)
#                         retry = False  # 成功获取响应后退出重试循环
#                     except Exception as e:#requests.exceptions.HTTPError as e:
#                         print(f"Thread {threading.current_thread().name} is working")
#                         print(e)
#                         continue
#             chunk_score.append(row_scores)
#         print("hhhhhhhhhhhhhhhhhh")
#         print(len(chunk_score))
#         return chunk_score

#这是用openAI api的版本 先改成每次一个实体同时去掉description的情况,另外加入hr2t和b的邻居
    def process_triplet_together_ai(self, head, head_description, tail, tail_description, relation, start_index, end_index, one2n_text, all_neighbors_text,h2rt_all_text):
        print(f"Thread {threading.current_thread().name} is working")
        chunk_score = []
        TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

        client = OpenAI(
        api_key=TOGETHER_API_KEY,
        base_url='https://api.together.xyz/v1',
        )
        for i in range(start_index, end_index):
            row_scores = []
            prompt_to_add = ""
            if len(one2n_text[i]) >0:
                for k in range(len(one2n_text[i])):
                    #print(len(one2n_text[i]))
                    one2n_text_list = list(one2n_text[i])
                    prompt_to_add += f"correct triplet : Subject Entity A: {head[i]} ; \
Object Entity B: {one2n_text_list[k]} ; \
Predicate Relation: {relation[i]}.\n"
            if len(all_neighbors_text[i]) > 0:
                for k in range(len(all_neighbors_text[i])):
                    prompt_to_add += f"correct triplet : Subject Entity A: {head[i]} ; \
Object Entity B: {all_neighbors_text[i][k][1]} ; \
Predicate Relation: {all_neighbors_text[i][k][0]}.\n"             
            for j in range(0, len(tail)):
                prompt_to_add_b = ""
                if len(h2rt_all_text[j]) > 0:
                    #print(len(h2rt_all_text[j]))
                    for k in range(len(h2rt_all_text[j])):
                        prompt_to_add_b += f"correct triplet : Subject Entity A: {tail[j]} ; \
Object Entity B: {h2rt_all_text[j][k][1]} ; \
Predicate Relation: {h2rt_all_text[j][k][0]}.\n"   
                    
                retry = True  # 是否重试的标志
                while retry:  # 循环重试直到请求成功
                    try:
                        start_time = time.time()

                        # 初始化 prompt
                        prompt = f"I would like you to handle this task for me : For the triplet I will provide shortly, estimate the likelihood that the subject entity A and the object entity B are connected by a predicate relation, forming a plausible true triplet in a knowledge graph. \
I will also provide some correct triplets from the real knowledge graph as a judgement factor for your score. These correct triplets, sharing the same entity partly as the triplet you need to score, can assist your judgment through this proximity relationship. \
I will give you the triplet in this following format: \n\"target triplet : Subject Entity A: xxx ;\nObject Entity B: xxx ;\nPredicate Relation: xxx.\"\n\
And the correct triplets will be provided like this: \"correct triplet : Subject Entity A: xxx ; Object Entity B: xxx ; Predicate Relation: xxx.\"\n\
xxx will be the corresponding texts. \
Now the triplet is as follows :\n\n"
                        # The entities may consist of one or more words, and you should avoid confusing similar entities like \"human\" and \"human caused phenomenon or process\".\
                        prompt += f"\"target triplet: Subject Entity A: {head[i]} ;\n\
Object Entity B: {tail[j]} ;\n\
Predicate Relation: {relation[i]}.\n\n"

                        prompt += prompt_to_add
                        prompt += prompt_to_add_b
                        # 添加后续指示
                        prompt += f"\"\n\nYou should divide the scale from 0 to 1 into five parts and provide scores with fine granularity, like:\n\
0.0000 - 0.1999: Extremely unlikely to occur, almost impossible.\n\
0.2000 - 0.3999: Very unlikely to occur, with a small chance of happening.\n\
0.4000 - 0.5999: Uncertain, neither likely nor unlikely.\n\
0.6000 - 0.7999: Likely to occur, with a fair chance of happening.\n\
0.8000 - 1.0000: Highly likely to occur, with a strong probability of happening.\n\
Present your answer in the format: 'value of the target triplet's likelihood = 0.XXXX, Object Entity: XXXX' where 0.XXXX is a value between 0 and 1 with four decimal places, and 'Object Entity: XXXX' represents the Object Entity B provided.\n\
Ensure that your response strictly adheres to the format provided above: 'value of the target triplet's likelihood = 0.XXXX, Object Entity: XXXX'. Only one row, no additional explanation."


                        #print(prompt)
                        # # 打开文件并写入内容
                        # with open("测试prompt.txt", "w") as file:
                        #     file.write(prompt)
                        while True:
                            chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                "role": "system",
                                "content": "You are a well-pretrained LLM, possesses a strong ability to analyze the relation between entities.",
                                },
                                {
                                "role": "user",
                                "content": prompt
                                }
                            ],
                            model=self.together_ai_model
                            )

                            # parse the completion then print the whole output
                            generatedText = chat_completion.choices[0].message.content
                            print(generatedText)
                            #print('-----------------------------------------------------------------------------------------')
                            # 使用正则表达式在文本中搜索包含特定句子的部分
                            likelihoods_str = re.findall(r'likelihood = (\d+\.\d+)', generatedText)
                            score = [float(likelihood) for likelihood in likelihoods_str]
                            print(f"当前i为{i}, j为{j}, 有{len(score)}个数据")
                            if len(score) > 1:
                                score = [score[0]]
                                break
                            elif len(score) == 1:
                                break
                        print(score)
                        print('\n')
                        end_time = time.time()
                        iteration_time = end_time - start_time
                        print("循环执行时间:", iteration_time, "秒")
                        row_scores.extend(score)
                        retry = False  # 成功获取响应后退出重试循环
                    except Exception as e:#requests.exceptions.HTTPError as e:
                        print(f"Thread {threading.current_thread().name} is working")
                        print(e)
                        continue
            chunk_score.append(row_scores)
        print("hhhhhhhhhhhhhhhhhh")
        print(len(chunk_score))
        return chunk_score

    def process_triplet_claude(self, head, head_description, tail, tail_description, relation, start_index, end_index, one2n_text, all_neighbors_text,h2rt_all_text):
        print(f"Thread {threading.current_thread().name} is working")
        chunk_score = []
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        for i in range(start_index, end_index):
            row_scores = []
            prompt_to_add = ""
            if len(one2n_text[i]) >0:
                for k in range(len(one2n_text[i])):
                    #print(len(one2n_text[i]))
                    one2n_text_list = list(one2n_text[i])
                    prompt_to_add += f"correct triplet : Subject Entity A: {head[i]} ; \
Object Entity B: {one2n_text_list[k]} ; \
Predicate Relation: {relation[i]}.\n"
            if len(all_neighbors_text[i]) > 0:
                for k in range(len(all_neighbors_text[i])):
                    prompt_to_add += f"correct triplet : Subject Entity A: {head[i]} ; \
Object Entity B: {all_neighbors_text[i][k][1]} ; \
Predicate Relation: {all_neighbors_text[i][k][0]}.\n"             
            for j in range(0, len(tail)):
                prompt_to_add_b = ""
                if len(h2rt_all_text[j]) > 0:
                    #print(len(h2rt_all_text[j]))
                    for k in range(len(h2rt_all_text[j])):
                        prompt_to_add_b += f"correct triplet : Subject Entity A: {tail[j]} ; \
Object Entity B: {h2rt_all_text[j][k][1]} ; \
Predicate Relation: {h2rt_all_text[j][k][0]}.\n"   
                    
                retry = True  # 是否重试的标志
                while retry:  # 循环重试直到请求成功
                    try:
                        start_time = time.time()

                        # 初始化 prompt
                        prompt = f"I would like you to handle this task for me : For the triplet I will provide shortly, estimate the likelihood that the subject entity A and the object entity B are connected by a predicate relation, forming a plausible true triplet in a knowledge graph. \
I will also provide some correct triplets from the real knowledge graph as a judgement factor for your score. These correct triplets, sharing the same entity partly as the triplet you need to score, can assist your judgment through this proximity relationship. \
I will give you the triplet in this following format: \n\"target triplet : Subject Entity A: xxx ;\nObject Entity B: xxx ;\nPredicate Relation: xxx.\"\n\
And the correct triplets will be provided like this: \"correct triplet : Subject Entity A: xxx ; Object Entity B: xxx ; Predicate Relation: xxx.\"\n\
xxx will be the corresponding texts. \
Now the triplet is as follows :\n\n"
                        # The entities may consist of one or more words, and you should avoid confusing similar entities like \"human\" and \"human caused phenomenon or process\".\
                        prompt += f"\"target triplet: Subject Entity A: {head[i]} ;\n\
Object Entity B: {tail[j]} ;\n\
Predicate Relation: {relation[i]}.\n\n"

                        prompt += prompt_to_add
                        prompt += prompt_to_add_b
                        # 添加后续指示
                        prompt += f"\"\n\nYou should divide the scale from 0 to 1 into five parts and provide scores with fine granularity, like:\n\
0.0000 - 0.1999: Extremely unlikely to occur, almost impossible.\n\
0.2000 - 0.3999: Very unlikely to occur, with a small chance of happening.\n\
0.4000 - 0.5999: Uncertain, neither likely nor unlikely.\n\
0.6000 - 0.7999: Likely to occur, with a fair chance of happening.\n\
0.8000 - 1.0000: Highly likely to occur, with a strong probability of happening.\n\
Present your answer in the format: 'value of the target triplet's likelihood = 0.XXXX, Object Entity: XXXX' where 0.XXXX is a value between 0 and 1 with four decimal places, and 'Object Entity: XXXX' represents the Object Entity B provided.\n\
Ensure that your response strictly adheres to the format provided above: 'value of the target triplet's likelihood = 0.XXXX, Object Entity: XXXX'. Only one row, no additional explanation."


                        #print(prompt)
                        # # 打开文件并写入内容
                        # with open("测试prompt.txt", "w") as file:
                        #     file.write(prompt)
                        while True:
                            message = client.messages.create(
                                model="claude-3-opus-20240229",
                                max_tokens=4096,
                                temperature=0.8,
                                system="You are a well-pretrained LLM, possesses a strong ability to analyze the relation between entities.",
                                messages=[{
                                    "role": "user",
                                    "content": prompt
                                }]
                            )

                            # parse the completion then print the whole output
                            generatedText = message.content[0].text
                            print(generatedText)
                            #print('-----------------------------------------------------------------------------------------')
                            # 使用正则表达式在文本中搜索包含特定句子的部分
                            likelihoods_str = re.findall(r'likelihood = (\d+\.\d+)', generatedText)
                            score = [float(likelihood) for likelihood in likelihoods_str]
                            print(f"当前i为{i}, j为{j}, 有{len(score)}个数据")
                            if len(score) > 1:
                                score = [score[0]]
                                break
                            elif len(score) == 1:
                                break
                        print(score)
                        print('\n')
                        end_time = time.time()
                        iteration_time = end_time - start_time
                        print("循环执行时间:", iteration_time, "秒")
                        row_scores.extend(score)
                        retry = False  # 成功获取响应后退出重试循环
                    except Exception as e:#requests.exceptions.HTTPError as e:
                        print(f"Thread {threading.current_thread().name} is working")
                        print(e)
                        continue
            chunk_score.append(row_scores)
        print("hhhhhhhhhhhhhhhhhh")
        print(len(chunk_score))
        return chunk_score

# 三个的claude，老旧版本0.15
#     def process_triplet_claude(self, head, head_description, tail, tail_description, relation, start_index, end_index):
#         print(f"Thread {threading.current_thread().name} is working")
#         chunk_score = []
#         client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
#         for i in range(start_index, end_index):
#             row_scores = []
#             #num_triplets = 3  # 设置所需的三元组数量
#             for j in range(0, len(tail),self.num_triplets):
#                 num_triplets_this = min(self.num_triplets, len(tail) - j)
#                 retry = True  # 是否重试的标志
#                 while retry:  # 循环重试直到请求成功
#                     try:
#                         start_time = time.time()
# #                         # 初始化 prompt
# #                         prompt = f"<s>[INST] <<SYS>>\
# # You are a well-pretrained LLM, possesses a strong ability to analyze the relation between entities.<</SYS>>\n \
# # I will give you {num_triplets_this} triplets in the following format: \"triplet 1 : Subject Entity A: xxx, Description: xxx;\nObject Entity B: xxx, Description: xxx;\nPredicate Relation: xxx.\"\nxxx will be the corresponding texts. The entity may consist of one or more words, and you should avoid confusing similar entities like \"human\" and \"human caused phenomenon or process\". Each index represents a triplet I give you.\n \
# # I want you to provide a numerical value for each triplet, expressing the likelihood that subject entity A and object entity B are linked by a predicate relation to form a plausible triplet in a knowledge graph.\n \
# # Now the triplets are as follows :\n\n"

# #                         # 循环生成三元组
# #                         for k in range(num_triplets_this):
# #                             prompt += f"triplet {k+1} : Subject Entity A: {head[i]}, Description: {head_description[i]};\n\
# # Object Entity B: {tail[j+k]}, Description: {tail_description[j+k]};\n\
# # Predicate Relation: {relation[i]}.\n\n"

# #                         #这里别漏了
# #                         #j+=num_triplets
# #                         # 添加后续指示
# #                         prompt += f"You should divide the scale from 0 to 1 into five parts and provide scores with finer granularity.\n\
# # Emphasizing, the factor you refer to must be the predicate relation I provided, and the relation between the two entities is an unidirectional predicate, with entity A pointing to entity B. If in your view, there is almost no predicate relation between Entity A and Entity B, you may give a low score near 0. If you believe there is a strong predicate relation from Entity A to Entity B, you may give a high score near 1. \
# # Present your answer using this format: \"value of the 1st(2nd, 3rd and so on, until reach {num_triplets_this}) triplet's likelihood = 0.XXXX, Object Entity:XXXX \" where 0.XXXX is a value between 0 and 1 with four decimal places, and Object Entity :XXXX shows the Object Entity B I give you.\n\
# # Ensure that your response for each corresponding triplet strictly adheres to the format provided above, and also ensure that the number of answers matches the quantity I requested, one answer per row. No extra explanation about your answers. [/INST]"



#                         # 初始化 prompt
#                         prompt = f"<s>[INST] <<SYS>>You are a well-pretrained LLM, possesses a strong ability to analyze the relation between entities.<</SYS>>\n \
# I would like you to handle this task for me : For each triplet I give you, provide me a numerical value, expressing the likelihood that subject entity A and object entity B are linked by a predicate relation to form a plausible triplet in some knowledge graphs.\
# I will give you {num_triplets_this} triplets in the following format: \"triplet 1 : Subject Entity A: xxx, Description: xxx;\nObject Entity B: xxx, Description: xxx;\nPredicate Relation: xxx.\"\nxxx will be the corresponding texts.\
# The Subject Entity A and the Predicate Relation are fixed, while the Object Entity B varies with the triplet index. Therefore, I will only provide the description of Subject Entity A in the first triplet.\n \
# The entities may consist of one or more words, and you should avoid confusing similar entities like \"human\" and \"human caused phenomenon or process\".\
# Now the triplets are as follows :\n\n"

#                         # 循环生成三元组
#                         for k in range(num_triplets_this):
#                             if k==0 :
#                                 prompt += f"triplet {k+1} : Subject Entity A: {head[i]}, Description: {head_description[i]}\n\
# Object Entity B: {tail[j+k]}, Description: {tail_description[j+k]}\n\
# Predicate Relation: {relation[i]}.\n\n"
#                             else :
#                                 prompt += f"triplet {k+1} : Subject Entity A: {head[i]}, Description: Refer to triplet 1\n\
# Object Entity B: {tail[j+k]}, Description: {tail_description[j+k]}\n\
# Predicate Relation: {relation[i]}.\n\n"

#                         #这里别漏了
#                         #j+=num_triplets
#                         # 添加后续指示
#                         prompt += f"You should divide the scale from 0 to 1 into five parts and provide scores with fine granularity, like:\n\
# 0.0000 - 0.1999: Extremely unlikely to occur, almost impossible.\n\
# 0.2000 - 0.3999: Very unlikely to occur, with a small chance of happening.\n\
# 0.4000 - 0.5999: Uncertain, neither likely nor unlikely.\n\
# 0.6000 - 0.7999: Likely to occur, with a fair chance of happening.\n\
# 0.8000 - 1.0000: Highly likely to occur, with a strong probability of happening.\n\
# Emphasizing, the factor you refer to must be the predicate relation I provided, and the relation between the two entities is an unidirectional predicate, with subject entity A pointing to object entity B.\
# Present your answer using this format: \"value of the 1st(2nd and so on, until reach {num_triplets_this}) triplet's likelihood = 0.XXXX, Object Entity:XXXX \" where 0.XXXX is a value between 0 and 1 with four decimal places, and Object Entity :XXXX means the Object Entity B I give you.\n\
# Ensure that your response for each corresponding triplet strictly adheres to the format provided above, and also ensure that the number of answers matches the quantity I requested, one answer per row. No extra explanation about your answers. [/INST]"



#                         #print(prompt)
#                         # # 打开文件并写入内容
#                         # with open("测试prompt.txt", "w") as file:
#                         #     file.write(prompt)
#                         while True:
#                             message = client.messages.create(
#                                 model="claude-2.1",
#                                 max_tokens=4096,
#                                 temperature=0.8,
#                                 system="You are a well-pretrained LLM, possesses a strong ability to analyze the relation between entities.",
#                                 messages=[{
#                                     "role": "user",
#                                     "content": prompt
#                                 }]
#                             )

#                             # parse the completion then print the whole output
#                             generatedText = message.content[0].text
#                             print(generatedText)
#                             #print('-----------------------------------------------------------------------------------------')
#                             # 使用正则表达式在文本中搜索包含特定句子的部分
#                             likelihoods_str = re.findall(r'likelihood = (\d+\.\d+)', generatedText)
#                             score = [float(likelihood) for likelihood in likelihoods_str]
#                             print(f"当前i为{i},j为{j},有{len(score)}个数据")
#                             if len(score) == num_triplets_this:
#                                 break
#                         print(score)
#                         print('\n')
#                         end_time = time.time()
#                         iteration_time = end_time - start_time
#                         print("循环执行时间:", iteration_time, "秒")
#                         row_scores.extend(score)
#                         retry = False  # 成功获取响应后退出重试循环
#                     except Exception as e:#requests.exceptions.HTTPError as e:
#                         print(f"Thread {threading.current_thread().name} is working")
#                         print(e)
#                         continue
#             chunk_score.append(row_scores)
#         print("hhhhhhhhhhhhhhhhhh")
#         print(len(chunk_score))
#         return chunk_score




    def claude(self, head, head_description, relation, tail, tail_description, one2n_text, all_neighbors_text,h2rt_all_text):
        num_threads = self.num_threads  # 根据需要调整线程数量
        chunk_size = len(head) // num_threads
        remainder = len(head) % num_threads

        # 提交任务给线程池
        results = []
        start_index = 0
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(num_threads):
                # 计算当前线程的结束索引
                end_index = start_index + chunk_size
                if remainder > 0:
                    end_index += 1
                    remainder -= 1

                # 提交任务
                result = executor.submit(self.process_triplet_claude, head, head_description, tail, tail_description, relation, start_index, end_index, one2n_text, all_neighbors_text,h2rt_all_text)
                results.append((start_index, result))  # 保存起始索引和结果的对应关系

                # 更新下一个线程的开始索引
                start_index = end_index

            executor.shutdown(wait=True)

        # 根据线程启动的顺序获取结果
        score_list = []
        for start_index, future in sorted(results, key=lambda x: x[0]):
            chunk_score = future.result()
            for row_score in chunk_score:
                score_list.append(row_score)

        return score_list


    


    def process_triplet_llama2(self, head, head_description, tail, tail_description, relation, start_index, end_index):
        pid = os.getpid()
        print(f"Process {pid} is working")
        with open(f"process_output_{pid}.txt", "w") as output_file:
            chunk_score = []
            for i in range(start_index, end_index):
                row_scores = []
                for j in range(len(tail)):
                    start_time = time.time()
                    prompt = f"<s>[INST] <<SYS>>\
                    You are a well-pretrained LLM, possesses a strong ability to analyze the relation between entities.\
                    <</SYS>> \
                    I want you to give me a numerical value expressing the likelihood that subject entity A and object entity B are linked by a predicate relation to form a plausible triplet in a knowledge graph.\n \
                    I will describe a triplet in the following format: Subject Entity A: xxx, Description: xxx;\n Object Entity B: xxx, Description: xxx;\n Predicate Relation: xxx.\n xxx will be the corresponding texts. \
                    Now the triplet is as follows :\n \
                    \"Subject Entity A: {head[i]}, Description: {head_description[i]};\n Object Entity B: {tail[j]}, Description: {tail_description[j]};\n Predicate Relation: {relation[i]}.\n\"  You should divide the scale from 0 to 1 into five parts and provide scores with finer granularity, like: \n \
                    0.0000 - 0.1999: Extremely unlikely to occur, almost impossible.\n \
                    0.2000 - 0.3999: Very unlikely to occur, with a small chance of happening.\n \
                    0.4000 - 0.5999: Uncertain, neither likely nor unlikely.\n \
                    0.6000 - 0.7999: Likely to occur, with a fair chance of happening.\n \
                    0.8000 - 1.0000: Highly likely to occur, with a strong probability of happening.\n \
                    Emphasizing, the factor you refer to must be the predicate relation I provided, and the relation between the two entities is an unidirectional predicate, with entity A pointing to entity B. If in your view, there is almost no predicate relation between Entity A and Entity B, you may give a low score near 0. If you believe there is a strong predicate relation from Entity A to Entity B, you may give a high score near 1. \
                    Present your answer using this format: \"value of a plausible triplet's likelihood = 0.XXXX\" where 0.XXXX is a value between 0 and 1. \
                    Ensure that your answer follows the same format as the example above, and there are significant figures in all decimal places. No extra explanation about your answer. [/INST]"
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.gpu)
                    #inputs = inputs.half()
                    tokens = self.model.generate(inputs["input_ids"], max_length=4096, do_sample=True, temperature=0.8)
                    generated_text = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
                    #print(generated_text)
                    # 使用正则表达式在文本中搜索包含特定句子的部分
                    match = re.search(r'value of a plausible triplet\'s likelihood = (\d+\.\d+)', generated_text,re.IGNORECASE)
                    score = float(match.group(1))
                    print(match,file=output_file)
                    print('\n',file=output_file)
                    
                    end_time = time.time()
                    iteration_time = end_time - start_time
                    print("第", j + 1, "轮循环执行时间:", iteration_time, "秒",file=output_file)

                    print('-----------------------------------------------------------------------------------------',file=output_file)
                    row_scores.append(score)
                chunk_score.append(row_scores)
        return chunk_score




    def llama2(self, head, head_description, relation, tail, tail_description, one2n_text):
        # 获取系统 CPU 核心数量
        num_processes = 4

        # 创建进程池，进程数根据 CPU 核心数量确定
        pool = multiprocessing.Pool(processes=num_processes)
        
        # 计算每个进程要处理的任务数量
        chunk_size = len(head) // num_processes
        
        # 计算余数
        remainder = len(head) % num_processes

        # 提交任务给进程池
        results = []
        start_index = 0
        for i in range(num_processes):
            # 计算当前进程的结束索引
            end_index = start_index + chunk_size
            if i == num_processes - 1:
                end_index += remainder
            
            result = pool.apply_async(self.process_triplet_llama2, (head, head_description, tail, tail_description, relation, start_index, end_index))
            results.append(result)

            # 更新下一个进程的开始索引
            start_index = end_index


        # 关闭进程池
        pool.close()
        pool.join()
        # 获取结果
        score_list = []
        for result in results:
            chunk_score = result.get()
            for row_score in chunk_score:
                score_list.extend(row_score)
        return score_list
