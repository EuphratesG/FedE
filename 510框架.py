import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Llama2Tokenizer, Llama2Model,LlamaForCausalLM,AutoTokenizer

# 初始化LLM tokenizer和模型
gpu_num = "2"
gpu = torch.device('cuda:' + gpu_num)
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").half()
model = model.to(gpu)
print(model)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token

# 定义实体嵌入网络
class EREmbedding(nn.Module):
    def __init__(self, entity_embedding_tensor, relation_embedding_tensor):
        super(EREmbedding, self).__init__()
        self.entity_embedding = nn.Parameter(entity_embedding_tensor)
        self.relation_embedding = nn.Parameter(relation_embedding_tensor)

    def forward(self, entity_ids, relation_ids):
        entity_embeddings = torch.index_select(self.entity_embedding, dim=0, index=entity_ids)
        relation_embeddings = torch.index_select(self.relation_embedding, dim=0, index=relation_ids)
        return entity_embeddings, relation_embeddings

# 定义网络结构
class ERNetwork(nn.Module):
    def __init__(self, num_entities, embedding_dim):
        super(ERNetwork, self).__init__()
        self.entity_embedding = EREmbedding(num_entities, embedding_dim)
        self.score_function = nn.Linear(embedding_dim, 1)
        self.llm_model = llama2_model
        # 从文件加载张量
        self.loaded_entity_embeddings = torch.load('/home/yvhe/510entity_embeddings.pt')

        self.loaded_relation_embeddings = torch.load('/home/yvhe/510relation_embeddings.pt')



    def forward(self, sample, relation_embedding, entity_embedding, neg=True):
        if not neg:#无负例情况
            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
        else:# 这里是默认，有负例情况
#如果 entity_embedding 的形状是 (N,)，那么经过 unsqueeze(0) 操作后，
#形状将变为 (1, N)。这里 (1, N) 表示一个包含一个元素的行向量，其中 N 是原始嵌入向量的长度。
#注意这里理解成负样本仅仅是tail entity的序号而已           
            head_part, tail_part = sample
            # print(head_part.shape)
            # print("zheshi1tou")
            # print("\n")
            # print(len(tail_part[0]))
            # print("\n")
            #print(tail_part)
            batch_size = head_part.shape[0] # [512,3]
            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)  # [512,1,128]如果不squeeze的话就是[512,128]
            # print("\n")
            # print(head.shape)
            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1) #[512,1,128]
            #print(relation.shape)
            if tail_part == None:
                tail = entity_embedding.unsqueeze(0)
                #注意这里的tailpart是[512,256]的形状，是512个一维数组
            else:
                negative_sample_size = tail_part.size(1) # 256
                tail = torch.index_select(
                    entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1) #[512,256,128]
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
#            'LLM': self.LLM
        }

        score = model_func[self.model_name](head, relation, tail)
        # print(score.shape)
        # print(head.shape)[16,1,128][16,1,128][1,14541,128]
        # print(tail.shape)
        return score

    def forward(self, entity_ids,relation_ids, prompt_texts):

        embedding_model = EREmbedding(self.loaded_entity_embeddings, self.loaded_relation_embeddings)
        entity_embeddings, relation_embeddings = embedding_model(entity_ids, relation_ids)
        # 计算score function的损失
        score_output = self.score_function(entity_embeddings)
        score_loss = torch.mean(score_output)

        # 输入LLM，得到输出
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.llm_model(**inputs)
        llm_output = outputs.last_hidden_state

        # 在LLM输出上添加额外的层和逻辑来计算LLM损失
        # 这里假设LLM输出的是一个二分类问题的logits，实际问题可以根据具体需求调整
        llm_logits = torch.mean(llm_output, dim=1)  # 这里简单地取平均值
        llm_loss = torch.mean(torch.sigmoid(llm_logits))

        return score_loss, llm_loss

    def TransE(self, head, relation, tail):
        score = (head + relation) - tail
        #print(score.shape)[16,14541,128]
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        #print(score.shape) [16,14541]
        return score






# 定义训练数据和模型参数
num_entities = 1000
embedding_dim = 100
model = ERNetwork(num_entities, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)





# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    entity_ids = torch.randint(0, num_entities, (32,))
    prompt_texts = ["example prompt"] * 32  # 假设每个样本都有相同的prompt

    optimizer.zero_grad()
    score_loss, llm_loss = model(entity_ids, prompt_texts)
    total_loss = score_loss + llm_loss
    total_loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Score Loss: {score_loss.item()}, LLM Loss: {llm_loss.item()}")

# 保存训练好的模型
torch.save(model.state_dict(), "entity_network.pth")
