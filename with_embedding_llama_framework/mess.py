import torch
import torch.nn as nn

class EREmbedding(nn.Module):
    def __init__(self, entity_embedding_tensor, relation_embedding_tensor):
        super(EREmbedding, self).__init__()
        self.entity_embedding = nn.Parameter(entity_embedding_tensor)
        self.relation_embedding = nn.Parameter(relation_embedding_tensor)

    def forward(self, entity_ids, relation_ids):
        entity_embeddings = torch.index_select(self.entity_embedding, dim=0, index=entity_ids)
        relation_embeddings = torch.index_select(self.relation_embedding, dim=0, index=relation_ids)
        return entity_embeddings, relation_embeddings

gpu_num = "6"
gpu = torch.device('cuda:' + gpu_num)
# 从文件加载张量
entity_embedding_tensor = torch.load('/home/yvhe/510entity_embeddings.pt')


relation_embedding_tensor = torch.load('/home/yvhe/510relation_embeddings.pt')

# 创建 EREmbedding 实例时传递实体嵌入张量和关系嵌入张量
entity_embedding_model = EREmbedding(entity_embedding_tensor, relation_embedding_tensor)

# 示例输入，要获取第1个实体和第1个关系的嵌入
entity_ids = torch.tensor([0])  # 实体索引
relation_ids = torch.tensor([0])  # 关系索引

# 使用模型进行前向传播
entity_embeddings, relation_embeddings = entity_embedding_model(entity_ids, relation_ids)
print("Entity Embeddings:", entity_embeddings.shape)
print("Relation Embeddings:", relation_embeddings.shape)
