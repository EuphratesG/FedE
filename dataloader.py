import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict as ddict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class TrainDataset(Dataset):
    def __init__(self, triples, nentity, negative_sample_size):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.negative_sample_size = negative_sample_size

# 创建一个以集合为元素的dict
        self.hr2t = ddict(set)
        for h, r, t in triples:
            self.hr2t[(h, r)].add(t)
# 每个hr对的对应target集合转list，list转np数组
        for h, r in self.hr2t:
            self.hr2t[(h, r)] = np.array(list(self.hr2t[(h, r)]))
#得到的结果是hr对为索引，np数组为值的index三元组
        #print(self.hr2t)
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        #一个batch里面的一个样本
        positive_sample = self.triples[idx]
        #print(positive_sample)
        head, relation, tail = positive_sample

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            # 这一行代码生成一个包含随机整数的NumPy数组，范围是 [0, self.nentity)，
            #数组的长度是 self.negative_sample_size * 2。这里生成了两倍于目标负样本数量的候选负样本,nentity的作用是确定了数值范围
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)# 512的数组
            #print(negative_sample)
            #print(len(negative_sample))
            #print("\n")
            #生成一个布尔数组，指示所有不在正样本中的负样本（因为负样本是随机生成的所以可能在正样本中）
            mask = np.in1d(
                negative_sample,
                self.hr2t[(head, relation)],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
#保留前size个元素（所有行就一行，size列）
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)
# 注意这里返回的都是embedding的index一维列表没有取具体
        return positive_sample, negative_sample, idx

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        sample_idx = torch.tensor([_[2] for _ in data])
        return positive_sample, negative_sample, sample_idx


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, ent_mask=None):
        self.len = len(triples)
        self.triple_set = all_true_triples
        self.triples = triples
        self.nentity = nentity

        self.ent_mask = ent_mask

        self.hr2t_all = ddict(set)
        for h, r, t in all_true_triples:
            self.hr2t_all[(h, r)].add(t)

    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        return triple, trp_label

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        label = self.hr2t_all[(head, relation)]
        trp_label = self.get_label(label)
        triple = torch.LongTensor((head, relation, tail))

        return triple, trp_label

    def get_label(self, label):
        y = np.zeros([self.nentity], dtype=np.float32)

        if type(self.ent_mask) == np.ndarray:
            y[self.ent_mask] = 1.0

        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)


class TestDataset_Entire(Dataset):
    #    test_dataset = TestDataset_Entire(test_triples, all_triples, nentity, test_client_idx, ent_mask)
    def __init__(self, triples, all_true_triples, nentity, triple_client_idx=None, ent_mask=None):
        self.len = len(triples)
        self.triple_set = all_true_triples
        self.triples = triples
        self.nentity = nentity

        self.triple_client_idx = torch.tensor(triple_client_idx, dtype=torch.int)
        self.ent_mask = ent_mask

        self.hr2t_all = ddict(set)
        for h, r, t in all_true_triples:
            self.hr2t_all[(h, r)].add(t)          

    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        triple_idx = torch.stack([_[2] for _ in data], dim=0)
        return triple, trp_label, triple_idx

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        # 这里是client的序号012，单个三元组的client序号
        triple_idx = self.triple_client_idx[idx]
        # 该batch里这个三元组的tail序号
        label = self.hr2t_all[(head, relation)]
        #这里label是个set，对应该batch里这个三元组的tail实体序号（因为可能有多个tail）
        #print(label)
        trp_label = self.get_label(label, triple_idx) #[14541]
        triple = torch.LongTensor((head, relation, tail))
        count_ones = torch.sum(trp_label == 1).item()

        print("该三元组一对多的个数:", count_ones)
        #trp_label就是没有在训练集里出现的实体的index加上该三元组上面的label
        return triple, trp_label, triple_idx

    def get_label(self, label, triple_idx=None):
        # 14541的0数组
        y = np.zeros([self.nentity], dtype=np.float32)
        if triple_idx is not None and type(self.ent_mask) == list:
            y[self.ent_mask[triple_idx]] = 1.0
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)



class TestDataset_Entire_with_hr2t(Dataset):
    #    test_dataset = TestDataset_Entire(test_triples, all_triples, nentity, test_client_idx, ent_mask)
    def __init__(self, triples, all_true_triples, nrelation, triple_client_idx=None, ent_mask=None):
        self.len = len(triples)
        self.triple_set = all_true_triples
        self.triples = triples
        self.nrelation = nrelation

        self.triple_client_idx = torch.tensor(triple_client_idx, dtype=torch.int)
        self.ent_mask = ent_mask

        self.hr2t_all = ddict(set)
        for h, r, t in all_true_triples:
            self.hr2t_all[(h, r)].add(t)     

        # 创建一个 defaultdict，值的默认类型是列表
        self.h2rt_all = ddict(list)
        # 遍历 all_true_triples 列表，并将 [r, t] 列表添加到 hr2rt_all 字典中对应的列表中
        for h, r, t in all_true_triples:
            self.h2rt_all[h].append([r, t])


        self.ht2r_all = ddict(set)
        for h, r, t in all_true_triples:
            self.ht2r_all[(h,t)].add(r) 


    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        triple_idx = torch.stack([_[2] for _ in data], dim=0)
        # 将每个 batch 中的 set 堆叠成一个列表
        one2n = [_[3] for _ in data]
        neighbor = [_[4] for _ in data]
        return triple, trp_label, triple_idx, one2n, neighbor

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        # 这里是client的序号012，单个三元组的client序号
        triple_idx = self.triple_client_idx[idx]
        # 该batch里这个三元组的tail序号
        one2n = self.hr2t_all[(head, relation)]
        # 这里直接暴力remove应该不会有什么问题
        one2n.remove(tail)
        neighbor = self.h2rt_all[head]
        neighbor.remove([relation,tail])
        #这里label是个set，对应该batch里这个三元组的tail实体序号（因为可能有多个tail）
        #print(label)
        ht2r = self.ht2r_all[(head,tail)]
        ht2r.remove(relation)
        trp_label = self.get_label(ht2r, triple_idx) #[14541]
        triple = torch.LongTensor((head, relation, tail))
        #count_ones = torch.sum(trp_label == 1).item()

        #print("该三元组一对多的个数:", count_ones)
        #trp_label就是没有在训练集里出现的实体的index加上该三元组上面的label
        return triple, trp_label, triple_idx, one2n, neighbor

    def get_label(self, label, triple_idx=None):
        # 14541的0数组
        y = np.zeros([self.nrelation], dtype=np.float32)
        # if triple_idx is not None and type(self.ent_mask) == list:
        #     y[self.ent_mask[triple_idx]] = 1.0
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)


#用于isolation
def get_task_dataset(data, args):


#这里用到非ori的就是分给共3个用户自己的
    train_triples = np.stack((data['train']['edge_index'][0],
                              data['train']['edge_type'],
                              data['train']['edge_index'][1])).T

    valid_triples = np.stack((data['valid']['edge_index'][0],
                              data['valid']['edge_type'],
                              data['valid']['edge_index'][1])).T

    test_triples = np.stack((data['test']['edge_index'][0],
                             data['test']['edge_type'],
                             data['test']['edge_index'][1])).T
    #print(type(train_triples))#<class 'numpy.ndarray'>
    #print(train_triples)#多个三元组（内容是数字索引）
    all_triples = np.concatenate([train_triples, valid_triples, test_triples])

    #这里用unique得到了实体数12379和关系数78
    # 提取所有三元组的关系和实体数, 获取唯一的关系和它们的数量
    nrelation = len(np.unique(all_triples[:, 1]))
    nentity = len(np.unique(np.concatenate((all_triples[:, 0], all_triples[:, 2]))))
    # nentity = len(np.unique(data['train']['edge_index'].reshape(-1)))
    # nrelation = len(np.unique(data['train']['edge_type']))
    # print(nentity)
    # print("\n")
    # print(nrelation)
    #这里又要跳转到新的类
    train_dataset = TrainDataset(train_triples, nentity, args.num_neg)
    valid_dataset = TestDataset(valid_triples, all_triples, nentity)
    test_dataset = TestDataset(test_triples, all_triples, nentity)

    return train_dataset, valid_dataset, test_dataset, nrelation, nentity

#用于collection
def get_task_dataset_entire(data, args):
    train_edge_index = np.array([[], []], dtype=np.int32)
    train_edge_type = np.array([], dtype=np.int32)

    valid_edge_index = np.array([[], []], dtype=np.int32)
    valid_edge_type = np.array([], dtype=np.int32)

    test_edge_index = np.array([[], []], dtype=np.int32)
    test_edge_type = np.array([], dtype=np.int32)

    train_client_idx = []
    valid_client_idx = []
    test_client_idx = []
    client_idx = 0
    for d in data:
        train_edge_index = np.concatenate([train_edge_index, d['train']['edge_index_ori']], axis=-1)
        valid_edge_index = np.concatenate([valid_edge_index, d['valid']['edge_index_ori']], axis=-1)
        test_edge_index = np.concatenate([test_edge_index, d['test']['edge_index_ori']], axis=-1)

        train_edge_type = np.concatenate([train_edge_type, d['train']['edge_type_ori']], axis=-1)
        valid_edge_type = np.concatenate([valid_edge_type, d['valid']['edge_type_ori']], axis=-1)
        test_edge_type = np.concatenate([test_edge_type, d['test']['edge_type_ori']], axis=-1)

        train_client_idx.extend([client_idx] * d['train']['edge_type_ori'].shape[0])
        #73492个0，后面1，后面2，的一维数组单纯的标号而已
        #print(train_client_idx)
        valid_client_idx.extend([client_idx] * d['valid']['edge_type_ori'].shape[0])
        test_client_idx.extend([client_idx] * d['test']['edge_type_ori'].shape[0])
        client_idx += 1

    # print(train_edge_index.shape)
    # print(valid_edge_index.shape)
    # 将所有的关系类型组合成一个数组
    all_edge_types = np.concatenate([train_edge_type, valid_edge_type, test_edge_type])

    # 计算唯一的关系类型
    unique_edge_types = np.unique(all_edge_types)
    all_entity_indices = np.concatenate([train_edge_index.flatten(), valid_edge_index.flatten(), test_edge_index.flatten()])

    # 计算唯一的实体索引
    unique_entity_indices = np.unique(all_entity_indices)

    nentity = len(unique_entity_indices)
    nrelation = len(unique_edge_types)
    ent_mask = []
    for idx, d in enumerate(data):
        client_mask_ent = np.setdiff1d(np.arange(nentity),
                                       np.unique(d['train']['edge_index_ori'].reshape(-1)), assume_unique=True)

        ent_mask.append(client_mask_ent)
    #print(ent_mask)
    train_triples = np.stack((train_edge_index[0],
                              train_edge_type,
                              train_edge_index[1])).T
    valid_triples = np.stack((valid_edge_index[0],
                              valid_edge_type,
                              valid_edge_index[1])).T
    test_triples = np.stack((test_edge_index[0],
                             test_edge_type,
                             test_edge_index[1])).T
    all_triples = np.concatenate([train_triples, valid_triples, test_triples])

    print("测试集形状")
    print(test_triples.shape)
    print(len(np.unique(np.concatenate([[row[0] for row in test_triples], [row[2] for row in test_triples]]))))
# #这里把test的三元组变成id搞到文件里
#     list = test_triples.tolist()
#     str_list = [str(element) for element in list]
#     # 使用列表解析处理每个元素，去掉空格和方括号
#     filtered_list = [x.replace(' ', '').replace('[', '').replace(']', '') for x in str_list]
#     # # 使用列表解析展平列表，并去掉所有方括号
#     # flat_list = [item for sublist in list for item in sublist]
#     # 打开一个文本文件以写入模式
#     with open("FB12k237/triplesid.txt", "w") as file:
#         # 遍历每个元素
#         for item in filtered_list:
#             # 将逗号替换为空格，并写入到文件中
#             file.write(item.replace(',', ' ') + '\n')

    train_dataset = TrainDataset(train_triples, nentity, args.num_neg)
    valid_dataset = TestDataset_Entire(valid_triples, all_triples, nentity, valid_client_idx, ent_mask)
    test_dataset = TestDataset_Entire(test_triples, all_triples, nentity, test_client_idx, ent_mask)


    return train_dataset, valid_dataset, test_dataset, nrelation, nentity


# 计算各客户端之间的共有实体
def find_common_entities(client_entities):
    common_entities = {}
    clients = list(client_entities.keys())
    for i in range(len(clients)):
        for j in range(i + 1, len(clients)):
            client1 = clients[i]
            client2 = clients[j]
            common = client_entities[client1].intersection(client_entities[client2])
            common_entities[(client1, client2)] = common
    return common_entities

#用于llm
def get_task_dataset_entire_with_hr2t(data, args):
    train_edge_index = np.array([[], []], dtype=np.int32)
    train_edge_type = np.array([], dtype=np.int32)

    valid_edge_index = np.array([[], []], dtype=np.int32)
    valid_edge_type = np.array([], dtype=np.int32)

    test_edge_index = np.array([[], []], dtype=np.int32)
    test_edge_type = np.array([], dtype=np.int32)

    # 初始化client-specific的字典
    train_client_data = {}
    valid_client_data = {}
    test_client_data = {}
    all_client_data = {}

    train_client_idx = []
    valid_client_idx = []
    test_client_idx = []
    client_idx = 0

    # 初始化实体集合字典
    client_entities = {}
    for d in data:
        train_edge_index = np.concatenate([train_edge_index, d['train']['edge_index_ori']], axis=-1)
        valid_edge_index = np.concatenate([valid_edge_index, d['valid']['edge_index_ori']], axis=-1)
        test_edge_index = np.concatenate([test_edge_index, d['test']['edge_index_ori']], axis=-1)

        train_edge_type = np.concatenate([train_edge_type, d['train']['edge_type_ori']], axis=-1)
        valid_edge_type = np.concatenate([valid_edge_type, d['valid']['edge_type_ori']], axis=-1)
        test_edge_type = np.concatenate([test_edge_type, d['test']['edge_type_ori']], axis=-1)



        # 提取实体集合
        train_entities = np.unique(d['train']['edge_index_ori'])
        valid_entities = np.unique(d['valid']['edge_index_ori'])
        test_entities = np.unique(d['test']['edge_index_ori'])

        all_entities = np.unique(np.concatenate([train_entities, valid_entities, test_entities]))
        client_entities[client_idx] = set(all_entities)

        # 将每个客户端的数据存储在字典中，并转换为三元组
        train_client_data[client_idx] = np.stack((d['train']['edge_index_ori'][0],
                                                d['train']['edge_type_ori'],
                                                d['train']['edge_index_ori'][1])).T
        valid_client_data[client_idx] = np.stack((d['valid']['edge_index_ori'][0],
                                                d['valid']['edge_type_ori'],
                                                d['valid']['edge_index_ori'][1])).T
        test_client_data[client_idx] = np.stack((d['test']['edge_index_ori'][0],
                                                d['test']['edge_type_ori'],
                                                d['test']['edge_index_ori'][1])).T

        all_client_data[client_idx] = np.concatenate([
            np.stack((d['train']['edge_index_ori'][0],
                    d['train']['edge_type_ori'],
                    d['train']['edge_index_ori'][1])).T,
            np.stack((d['valid']['edge_index_ori'][0],
                    d['valid']['edge_type_ori'],
                    d['valid']['edge_index_ori'][1])).T,
            np.stack((d['test']['edge_index_ori'][0],
                    d['test']['edge_type_ori'],
                    d['test']['edge_index_ori'][1])).T
        ])
        
        train_client_idx.extend([client_idx] * d['train']['edge_type_ori'].shape[0])
        #73492个0，后面1，后面2，的一维数组单纯的标号而已
        #print(train_client_idx)
        valid_client_idx.extend([client_idx] * d['valid']['edge_type_ori'].shape[0])
        test_client_idx.extend([client_idx] * d['test']['edge_type_ori'].shape[0])
        client_idx += 1



    common_entities = find_common_entities(client_entities)

    # 打印共有实体
    for (client1, client2), entities in common_entities.items():
        print(f"Clients {client1} and {client2} have {len(entities)} common entities: {entities}")


    # print(train_edge_index.shape)
    # print(valid_edge_index.shape)
    # 将所有的关系类型组合成一个数组
    all_edge_types = np.concatenate([train_edge_type, valid_edge_type, test_edge_type])

    # 计算唯一的关系类型
    unique_edge_types = np.unique(all_edge_types)
    all_entity_indices = np.concatenate([train_edge_index.flatten(), valid_edge_index.flatten(), test_edge_index.flatten()])

    # 计算唯一的实体索引
    unique_entity_indices = np.unique(all_entity_indices)

    nentity = len(unique_entity_indices)
    nrelation = len(unique_edge_types)
    ent_mask = []
    for idx, d in enumerate(data):
        client_mask_ent = np.setdiff1d(np.arange(nentity),
                                       np.unique(d['train']['edge_index_ori'].reshape(-1)), assume_unique=True)

        ent_mask.append(client_mask_ent)
    #print(ent_mask)
    train_triples = np.stack((train_edge_index[0],
                              train_edge_type,
                              train_edge_index[1])).T
    valid_triples = np.stack((valid_edge_index[0],
                              valid_edge_type,
                              valid_edge_index[1])).T
    test_triples = np.stack((test_edge_index[0],
                             test_edge_type,
                             test_edge_index[1])).T
    all_triples = np.concatenate([train_triples, valid_triples, test_triples])

    print("测试集形状")
    print(test_triples.shape)
    print(len(np.unique(np.concatenate([[row[0] for row in test_triples], [row[2] for row in test_triples]]))))




    # 创建一个 defaultdict，值的默认类型是列表
    h2rt_all = ddict(list)

    # 遍历 all_true_triples 列表，并将 [r, t] 列表添加到 hr2rt_all 字典中对应的列表中
    for h, r, t in all_triples:
        h2rt_all[h].append([r, t])
    train_dataset = TrainDataset(train_triples, nentity, args.num_neg)
    valid_dataset = TestDataset_Entire_with_hr2t(valid_triples, all_triples, nrelation, valid_client_idx, ent_mask)
    test_dataset = TestDataset_Entire_with_hr2t(test_triples, all_triples, nrelation, test_client_idx, ent_mask)

    return train_dataset, valid_dataset, test_dataset, nrelation, nentity, h2rt_all, all_client_data



def get_all_clients(all_data, args):
    all_ent = np.array([], dtype=int)
    #reshape展平为一维数组，然后union1d指取并集
    for data in all_data:
        all_ent = np.union1d(all_ent, data['train']['edge_index_ori'].reshape(-1))
    nentity = len(all_ent)

    train_dataloader_list = []
    test_dataloader_list = []
    valid_dataloader_list = []
    rel_embed_list = []

    ent_freq_list = []

    for data in tqdm(all_data):
    #循环单位是每个client,暂时3个client
        nrelation = len(np.unique(data['train']['edge_type']))

        train_triples = np.stack((data['train']['edge_index_ori'][0],
                                  data['train']['edge_type'],
                                  data['train']['edge_index_ori'][1])).T
        #print(train_triples)这里还是三元组的index
        valid_triples = np.stack((data['valid']['edge_index_ori'][0],
                                  data['valid']['edge_type'],
                                  data['valid']['edge_index_ori'][1])).T

        test_triples = np.stack((data['test']['edge_index_ori'][0],
                                 data['test']['edge_type'],
                                 data['test']['edge_index_ori'][1])).T

# 在原数据集中剔除掉训练集以留给验证集和测试集
        client_mask_ent = np.setdiff1d(np.arange(nentity),
                                       np.unique(data['train']['edge_index_ori'].reshape(-1)), assume_unique=True)

        all_triples = np.concatenate([train_triples, valid_triples, test_triples])
        train_dataset = TrainDataset(train_triples, nentity, args.num_neg)
        valid_dataset = TestDataset(valid_triples, all_triples, nentity, client_mask_ent)
        test_dataset = TestDataset(test_triples, all_triples, nentity, client_mask_ent)

        # dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_list.append(train_dataloader)

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.test_batch_size,
            collate_fn=TestDataset.collate_fn
        )
        valid_dataloader_list.append(valid_dataloader)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            collate_fn=TestDataset.collate_fn
        )
        test_dataloader_list.append(test_dataloader)

        embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim])
        if args.model in ['ComplEx']:
            rel_embed = torch.zeros(nrelation, args.hidden_dim * 2).to(args.gpu).requires_grad_()
        else:
            rel_embed = torch.zeros(nrelation, args.hidden_dim).to(args.gpu).requires_grad_()
#关系embedding正则化
        nn.init.uniform_(
            tensor=rel_embed,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )
        rel_embed_list.append(rel_embed)

        ent_freq = torch.zeros(nentity)
        for e in data['train']['edge_index_ori'].reshape(-1):
            ent_freq[e] += 1
        ent_freq_list.append(ent_freq)

    ent_freq_mat = torch.stack(ent_freq_list).to(args.gpu)

# 所有client的dataloader，所有client的实体频率列表，所有client的关系embedding，总实体数
    return train_dataloader_list, valid_dataloader_list, test_dataloader_list, \
           ent_freq_mat, rel_embed_list, nentity
