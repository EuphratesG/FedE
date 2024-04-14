import torch
import torch.nn as nn
from transformers import LlamaForCausalLM,LlamaModel,AutoTokenizer

class KGEModel(nn.Module):
    def __init__(self, args, model_name):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim])
        self.gamma = nn.Parameter(
            torch.Tensor([args.gamma]),
            requires_grad=False
        )

#forward就是实体输入参数之后得到一个返回值,前向传播
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
    
    # 维度不同怎么做加法？不够的张量自动扩展，因此这里体现了负样本规模越多投入的负样本就越多
    def TransE(self, head, relation, tail):
        score = (head + relation) - tail
        #print(score.shape)[16,14541,128]
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        #print(score.shape) [16,14541]
        return score

    def DistMult(self, head, relation, tail):
        score = (head * relation) * tail
        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score
    
    # def LLM(self, head, relation, tail):

    #     # print(type(model))
    #     #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     print(type(head))
    #     result = torch.cat((head, relation, tail), dim=1)
    #     # print(model)

    #     # prompt = "Hey, are you conscious? Can you talk to me? Holy shit!"
    #     # inputs = tokenizer(prompt, return_tensors="pt")
    #     gene_outputs = self.LLMModel(inputs_embeds=result)
    #     output_logits=gene_outputs.last_hidden_state.reshape(-1)
    #     score = torch.sum(torch.sigmoid(output_logits))
    #     print(score)
    #     print(score.shape)
        
    #     return score