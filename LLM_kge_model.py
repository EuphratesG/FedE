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
import traceback
from py2neo import Graph, Node, Relationship


class LLMKGEModel():#nn.Module):
    def __init__(self, args, LLM_model_name):
        super(LLMKGEModel, self).__init__()
        #multiprocessing.set_start_method('spawn')
        self.gpu=args.gpu
        if(args.LLMModel=="llama2"):
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").half()
            self.model = self.model.to(args.gpu)
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        elif(args.LLMModel=="together_ai"):
            self.together_ai_model = args.together_ai_model

        self.LLMModel = args.LLMModel
        self.num_triplets = args.num_triplets
        self.num_threads = args.num_threads        

        # # 读取文本文件
        # file_path = '/home/yvhe/FedE/fb15k-237/entity2textlong_sample.txt'  # 替换为你的文件路径
        # with open(file_path, 'r', encoding='utf-8') as file:
        #     lines = file.readlines()
        # self.entity_description = [line.strip().split(None, 1)[1] for line in lines]

        # file_path = '/home/yvhe/FedE/umls/entity2text.txt'  # 替换为你的文件路径
        # with open(file_path, 'r', encoding='utf-8') as file:
        #     lines = file.readlines()
        # self.entity_text = [line.strip().split(None, 1)[1] for line in lines]

        # # 读取文本文件
        # file_path = '/home/yvhe/FedE/umls/relation2text.txt'  # 替换为你的文件路径
        # with open(file_path, 'r', encoding='utf-8') as file:
        #     lines = file.readlines()
        # self.relation_text = [line.strip().split(None, 1)[1] for line in lines]

        # 读取文本文件
        file_path = '/home/yvhe/FedE/umls/entity2textlong.txt'  # 替换为你的文件路径
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        self.entity_description = [line.strip().split(None, 1)[1] for line in lines]



        # 读取关系ID和关系名字对应的文本文件
        self.entity_text = {}
        with open('/home/yvhe/FedE/umls/entity2text.txt', 'r') as file:
            for line_num, line in enumerate(file, start=0):
                parts = line.strip().split(None, 1)
                if len(parts) == 2:  # 确保有两个部分
                    key, value = line_num, parts[1]  # 行号作为键，第二个字符串作为值
                    self.entity_text[key] = value


        # 读取关系ID和关系名字对应的文本文件
        self.relation_text = {}
        with open('/home/yvhe/FedE/umls/relation2text.txt', 'r') as file:
            for line_num, line in enumerate(file, start=0):
                parts = line.strip().split(None, 1)
                if len(parts) == 2:  # 确保有两个部分
                    key, value = line_num, parts[1]  # 行号作为键，第二个字符串作为值
                    self.relation_text[key] = value



# MRR的打分函数
    def forward(self, sample, one2n, all_neighbors, h2rt_all,neg=True):
        #print(len(one2n))
        if not neg:#无负例情况
            head = [self.entity_text[index[0]] for index in sample]

            relation = [self.relation_text[index[1]] for index in sample]

            tail = [self.entity_text[index[2]] for index in sample]
        else:# 这里是默认，有负例情况         
            head_part, tail_part = sample
            print(head_part[0])
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
        }

        score = model_func[self.LLMModel](head, head_description, relation, tail, tail_description, one2n_text, all_neighbors_text,h2rt_all_text)
        
        return score



    def check_triple_exists(self, graph, start_node_id, relationship_id, end_node_id):
        query = """
        MATCH (a)-[r]->(b)
        WHERE a.id = $start_node_id AND r.id = $relationship_id AND b.id = $end_node_id
        RETURN a, r, b
        """
        result = graph.run(query, start_node_id=start_node_id, relationship_id=relationship_id, end_node_id=end_node_id).data()
        return len(result) > 0

    def get_neighbors_triples(self, graph, start_node_id, relationship_id, end_node_id):
        query = """
        MATCH (a)-[r]->(b)
        WHERE a.id = $start_node_id AND r.id = $relationship_id AND b.id = $end_node_id
        WITH a, b
        MATCH (a)-[r1]->(n)
        RETURN a.id AS start_node_id, r1.id AS relationship_id, n.id AS end_node_id
        UNION
        MATCH (n)-[r2]->(a)
        RETURN n.id AS start_node_id, r2.id AS relationship_id, a.id AS end_node_id
        UNION
        MATCH (b)-[r3]->(n)
        RETURN b.id AS start_node_id, r3.id AS relationship_id, n.id AS end_node_id
        UNION
        MATCH (n)-[r4]->(b)
        RETURN n.id AS start_node_id, r4.id AS relationship_id, b.id AS end_node_id
        """
        result = graph.run(query, start_node_id=start_node_id, relationship_id=relationship_id, end_node_id=end_node_id).data()
        
        # 过滤结果，仅保留直接连接 start_node_id 和 end_node_id 的邻居
        filtered_result = [record for record in result if record['start_node_id'] == start_node_id or record['end_node_id'] == end_node_id or record['start_node_id'] == end_node_id or record['end_node_id'] == start_node_id]

        return [(record['start_node_id'], record['relationship_id'], record['end_node_id']) for record in filtered_result]



    def add_triples_to_db(self, graph, triples, entity_names, relationship_names):
        for triple in triples:
            start_node_id, relationship_id, end_node_id = triple
            start_node_id = int(start_node_id)
            relationship_id = int(relationship_id)
            end_node_id = int(end_node_id)
            start_node_name = entity_names.get(start_node_id, "Unknown")
            end_node_name = entity_names.get(end_node_id, "Unknown")
            relationship_name = relationship_names.get(relationship_id, "UNKNOWN_RELATION")

            start_node = Node(start_node_name, id=start_node_id, name=start_node_name)
            end_node = Node(end_node_name, id=end_node_id, name=end_node_name)
            relationship = Relationship(start_node, relationship_name, end_node, id=relationship_id)

            # 检查是否存在重复的三元组
            if not self.check_triple_exists(graph, start_node_id, relationship_id, end_node_id):
                graph.merge(start_node, start_node_name, "id")
                graph.merge(end_node, end_node_name, "id")
                graph.merge(relationship)

    def process_triple(self, graph, client_data, start_node_id, relationship_id, end_node_id, entity_names, relationship_names):
        if self.check_triple_exists(graph, start_node_id, relationship_id, end_node_id):
            print("exists",start_node_id, relationship_id, end_node_id)
            triples = self.get_neighbors_triples(graph, start_node_id, relationship_id, end_node_id)
            return triples
        else:
            for triple in client_data:
                if (triple[0] == start_node_id) and (triple[1] == relationship_id) and (triple[2] == end_node_id):
                    neighbors = [
                        (t[0], t[1], t[2]) for t in client_data
                        if t[0] == start_node_id or t[2] == start_node_id or t[0] == end_node_id or t[2] == end_node_id
                    ]
                    self.add_triples_to_db(graph, neighbors, entity_names, relationship_names)
                    return neighbors

        return []

# 加上neo4j操作的打分函数
    def forward_PRAUC(self, triples, all_client_data, client_idx):       
        # Neo4j数据库连接参数
        uri = "bolt://localhost:7687"
        username = "neo4j"
        password = os.environ.get("neo4j_PASSWORD")  # 请替换为你的实际密码

        # 连接到Neo4j数据库
        graph = Graph(uri, auth=(username, password))

        head_text = [self.entity_text[index[0]] for index in triples]
        head_idx = [index[0] for index in triples]
        #head_description = [self.entity_description[index[0]] for index in triples]

        tail_text = [self.entity_text[index[2]] for index in triples]
        tail_idx = [index[2] for index in triples]
        #tail_description = [self.entity_description[index[2]] for index in triples]



        row_scores = []
        for target_triple in triples:
            # 示例实体ID和关系类型
            start_node_id = target_triple[0]
            relationship_id = target_triple[1]
            end_node_id = target_triple[2]
            target_triple_tmp = (start_node_id, relationship_id, end_node_id)

            related_triples = self.process_triple(graph, all_client_data[client_idx], start_node_id, relationship_id, end_node_id, self.entity_text, self.relation_text)
            print(related_triples)
            if target_triple_tmp in related_triples:
                related_triples.remove(target_triple_tmp)
            #print(related_triples)
            # # 输出格式化的相关三元组
            # for triple in related_triples:
            #     print(f"{triple[0]} {triple[1]} {triple[2]}")

            model_func = {
                'together_ai': self.together_ai_PRAUC
                #'claude': self.claude_PRAUC,
            }
            row_scores.extend(model_func[self.LLMModel](related_triples, target_triple))
        
        return row_scores


    def together_ai_PRAUC(self, related_triples, target_triple):
        # chunk_score = []
        TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

        client = OpenAI(
        api_key=TOGETHER_API_KEY,
        base_url='https://api.together.xyz/v1',
        )

        prompt_to_add = ""
        for triple in related_triples:
            prompt_to_add += f"correct triplet : Subject Entity A: {self.entity_text[triple[0]]} ; \
Object Entity B: {self.relation_text[triple[1]]} ; \
Predicate Relation: {self.entity_text[triple[2]]}.\n"    

                
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
                prompt += f"\"target triplet: Subject Entity A: {self.entity_text[target_triple[0]]} ;\n\
Object Entity B: {self.relation_text[target_triple[1]]} ;\n\
Predicate Relation: {self.entity_text[target_triple[2]]}.\n\n"
                prompt += prompt_to_add
                # 添加后续指示
                prompt += f"\"\n\nYou should divide the scale from 0 to 1 into five parts and provide scores with fine granularity, like:\n\
0.0000 - 0.1999: Extremely unlikely to occur, almost impossible.\n\
0.2000 - 0.3999: Very unlikely to occur, with a small chance of happening.\n\
0.4000 - 0.5999: Uncertain, neither likely nor unlikely.\n\
0.6000 - 0.7999: Likely to occur, with a fair chance of happening.\n\
0.8000 - 1.0000: Highly likely to occur, with a strong probability of happening.\n\
Present your answer in the format: 'value of the target triplet's likelihood = 0.XXXX, Predicate Relation: XXXX' where 0.XXXX is a value between 0 and 1 with four decimal places, and 'Predicate Relation: XXXX' represents the Predicate Relation provided.\n\
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
                retry = False  # 成功获取响应后退出重试循环
            except Exception as e:#requests.exceptions.HTTPError as e:
                print(f"Thread {threading.current_thread().name} is working")
                print(e)
                traceback.print_exc()
                continue

        return score




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




