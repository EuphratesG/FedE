import torch
import torch.nn as nn
import torch.optim as optim
from transformers import LlamaForCausalLM,LlamaModel,AutoTokenizer,AutoModelForCausalLM
import re


# 初始化LLM tokenizer和模型
gpu_num = "4"
gpu = torch.device('cuda:' + gpu_num)



# 定义网络结构
class ScoreNetwork(nn.Module):
    def __init__(self):
        super(ScoreNetwork, self).__init__()
        self.gamma = nn.Parameter(
            torch.Tensor([10]),
            requires_grad=False
        )

    def forward(self, sample, relation_embedding, entity_embedding, neg=True):
        if not neg:#无负例情况
            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=sample[:, 0]
            )#.unsqueeze(1)
            #head = torch.sum(head, dim=1, keepdim=True)
            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=sample[:, 1]
            )#.unsqueeze(1)
            #relation = torch.sum(relation, dim=1, keepdim=True)
            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=sample[:, 2]
            )#.unsqueeze(1)
            #tail = torch.sum(tail, dim=1, keepdim=True)

        # 计算score function的损失
        score = self.TransE(head, relation, tail)

        return score

    def TransE(self, head, relation, tail):
        print(head.shape)
        print(relation.shape)
        print(tail.shape)
        relation = torch.cat((relation, torch.zeros(relation.shape[0], tail.shape[1] - relation.shape[1], relation.shape[2]).to(gpu)), dim=1)
        score = (head + relation) - tail
        #print(score.shape)[16,14541,128]
        score = self.gamma.item() - torch.sum(torch.norm(score, p=1, dim=2),dim=1)
        #print(score.shape) [16,14541]
        return score



class AllNetwork(nn.Module):
    def __init__(self, entity_embedding, relation_embedding):
        super(AllNetwork, self).__init__()

        self.score_model = ScoreNetwork()
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding
        self.llm_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").half().to(gpu)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.tokenizer.pad_token = self.tokenizer.eos_token

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
    def forward(self, positive_sample):


        # 计算score function的输出
        score = self.score_model(positive_sample,self.relation_embedding, self.entity_embedding, neg=False)

        head = [self.entity_text[index[0]] for index in positive_sample]

        relation = [self.relation_text[index[1]] for index in positive_sample]

        tail = [self.entity_text[index[2]] for index in positive_sample]
        llm_score = []
        for i in range(len(head)):
            # like:\n\
            # 0.0000 - 0.1999: Extremely unlikely to occur, almost impossible.\n\
            # 0.2000 - 0.3999: Very unlikely to occur, with a small chance of happening.\n\
            # 0.4000 - 0.5999: Uncertain, neither likely nor unlikely.\n\
            # 0.6000 - 0.7999: Likely to occur, with a fair chance of happening.\n\
            # 0.8000 - 1.0000: Highly likely to occur, with a strong probability of happening.\n\
            prompt = "<s>[INST] <<SYS>>You are a professional assistant in handling this task : For the triplet's entity and relation embeddings I will provide shortly, estimate the likelihood that the subject entity A and the object entity B are connected by a predicate relation, forming a plausible true triplet in a knowledge graph. \
            I will give you the triplet in this following format: \ntarget triplet :\nSubject Entity A: name embedding\nObject Entity B: name embedding\nPredicate Relation: name embedding\n\
            The first row represents the name and embedding of Subject Entity A, the second row represents the name and embedding of Object Entity B, and the third row represents the name and embedding of Predicate Relation. The position 'embedding' in the format will be the corresponding embeddings. "
            # 添加后续指示
            prompt += "You should divide the scale from 0 to 1 into many parts and provide the score with fine granularity.\
            Present your answer in this format: 'value of the target triplet's likelihood = 0.XXXX, Object Entity: XXXX' where 0.XXXX is a value between 0 and 1 with four decimal places, and 'Object Entity: XXXX' represents the Object Entity's name I provide for you.\n\
            Ensure that: 1. You should strictly adhere to the answer format provided above, 'value of the target triplet's likelihood = 0.XXXX, Object Entity: XXXX', only one row. 2. No additional explanation. 3. Only score the triplet itself.<</SYS>> "
            prompt += "Now the triplet is as follows:\ntarget triplet:\n"

            entity_a_embed = self.entity_embedding[positive_sample[i][0]].unsqueeze(0).to(gpu)
            entity_b_embed = self.entity_embedding[positive_sample[i][2]].unsqueeze(0).to(gpu)
            relation_embed = self.relation_embedding[positive_sample[i][1]].unsqueeze(0).to(gpu)
            print(entity_b_embed.shape)
            # 输入LLM，得到输出
            prompt_token = self.tokenizer.encode(prompt, return_tensors="pt").to(gpu)
            prompt_embeds = self.llm_model.model.embed_tokens(prompt_token)

            a_token = self.tokenizer.encode("Subject Entity A: "+head[i]+' ', return_tensors="pt").to(gpu)
            a_embeds = self.llm_model.model.embed_tokens(a_token)   

            b_token = self.tokenizer.encode("Object Entity B: "+tail[i]+' ', return_tensors="pt").to(gpu)
            b_embeds = self.llm_model.model.embed_tokens(b_token)   

            c_token = self.tokenizer.encode("Predicate Relation: "+relation[i]+' ', return_tensors="pt").to(gpu)
            c_embeds = self.llm_model.model.embed_tokens(c_token)

            enter_token = self.tokenizer.encode("\n", return_tensors="pt").to(gpu)
            enter_embeds = self.llm_model.model.embed_tokens(enter_token)
            
            inst_token = self.tokenizer.encode("[/INST]", return_tensors="pt").to(gpu)
            inst_embeds = self.llm_model.model.embed_tokens(inst_token)       




            # 按照你的顺序拼接张量
            inputs_embeds = torch.cat((prompt_embeds, a_embeds,entity_a_embed, enter_embeds, b_embeds,entity_b_embed, enter_embeds, c_embeds,relation_embed,inst_embeds), dim=1).to(torch.float16).to(gpu)
            print(inputs_embeds.shape)
            outputs = self.llm_model.generate(inputs_embeds=inputs_embeds)
            outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(outputs)
            # 使用正则表达式在文本中搜索包含特定句子的部分
            likelihoods_str = re.findall(r' (\d+\.\d+)', outputs)
            llm_score.append([float(likelihood) for likelihood in likelihoods_str][0])
            # llm_output = outputs.last_hidden_state

            # # 在LLM输出上添加额外的层和逻辑来计算LLM损失
            # # 这里假设LLM输出的是一个二分类问题的logits，实际问题可以根据具体需求调整
            # llm_logits = torch.mean(llm_output, dim=1)  # 这里简单地取平均值
            # llm_loss = torch.mean(torch.sigmoid(llm_logits))

        return score, llm_score#,inputs_prompt,inputs_enter#, llm_output




# 定义训练数据和模型参数
num_entities = 1000
embedding_dim = 100

# 从文件加载张量
with torch.no_grad():
    entity_embedding = torch.load('/home/yvhe/FedE/with_embedding_llama_framework/510entity_embeddings.pt',map_location='cpu').to(torch.float32).to(gpu).requires_grad_(True)
# with torch.no_grad():
#     self.entity_embedding = torch.sum(self.entity_embedding, dim=1, keepdim=True).to(gpu).requires_grad_()
    relation_embedding = torch.load('/home/yvhe/FedE/with_embedding_llama_framework/510relation_embeddings.pt',map_location='cpu').to(torch.float32).to(gpu).requires_grad_(True)
# with torch.no_grad():
#     self.relation_embedding = torch.sum(self.relation_embedding, dim=1, keepdim=True).to(gpu).requires_grad_()
optimizer = torch.optim.Adam(
    [{'params': entity_embedding},
        {'params': relation_embedding}], lr=0.001
)

model = AllNetwork(entity_embedding, relation_embedding).to(gpu)

data = [
    [49, 28, 50],
    [81, 5, 21],
    [115, 43, 107],
    [89, 27, 50],
    [103, 26, 81],
    [22, 13, 128],
    [42, 19, 116],
    [111, 28, 63],
    [20, 40, 81],
    [19, 19, 49]
]
# 转换为张量
data = torch.tensor(data).to(gpu)

model.train()

score, llm_score= model(data)
score_loss = score.mean()


print(score.shape)
print(score)
print(llm_score)

# 模型的原始输出 logits（假设 batch size 为 10）
logits = torch.tensor(llm_score)

# 真实标签
labels = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32)

# 定义 BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()

# 计算损失
llm_loss = criterion(logits, labels)

print(f"BCEWithLogitsLoss: {llm_loss.item()}")
loss = llm_loss + score_loss

# 反向传播和更新权重
optimizer.zero_grad()
loss.backward()
optimizer.step()
# # 训练模型
# num_epochs = 10
# for epoch in range(num_epochs):
#     entity_ids = torch.randint(0, num_entities, (32,))
#     prompt_texts = ["example prompt"] * 32  # 假设每个样本都有相同的prompt

#     optimizer.zero_grad()
#     score_loss, llm_loss = model(entity_ids, prompt_texts)
#     total_loss = score_loss + llm_loss
#     total_loss.backward()
#     optimizer.step()

#     print(f"Epoch {epoch+1}, Score Loss: {score_loss.item()}, LLM Loss: {llm_loss.item()}")

# # 保存训练好的模型
# torch.save(model.state_dict(), "entity_network.pth")
