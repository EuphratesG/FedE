import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import pickle
import logging
import numpy as np
from collections import defaultdict as ddict
from dataloader import get_task_dataset, get_task_dataset_entire, get_task_dataset_entire_llm,\
    TrainDataset, TestDataset, TestDataset_Entire, TestDataset_Entire_llm
from kge_model import KGEModel
from LLM_kge_model import LLMKGEModel
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from py2neo import Graph, Node, Relationship



#涉及fede的collection、isolation
class KGETrainer():
    def __init__(self, args, data):
        self.args = args
        self.data = data

        if args.setting == 'Collection':
            train_dataset, valid_dataset, test_dataset, nrelation, nentity = get_task_dataset_entire(data, args)
        elif args.setting == 'Isolation':
            #dataset就是正例、负例、idx
            train_dataset, valid_dataset, test_dataset, nrelation, nentity = get_task_dataset(data, args)
        elif args.setting == 'LLM':
            #dataset就是正例、负例、idx
            train_dataset, valid_dataset, test_dataset, nrelation, nentity, h2rt_all , all_client_data= get_task_dataset_entire_llm(data, args)

        self.all_client_data = all_client_data
        self.nentity = nentity
        self.nrelation = nrelation

        #邻居相关
        self.h2rt_all = h2rt_all


        # embedding 向量维度
        embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim]) # tensor([0.0938])
        if args.model in ['RotatE', 'ComplEx']:
            self.entity_embedding = torch.zeros(self.nentity, args.hidden_dim * 2).to(args.gpu).requires_grad_()
        else:
            self.entity_embedding = torch.zeros(self.nentity, args.hidden_dim).to(args.gpu).requires_grad_()
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )
        if args.model in ['ComplEx']:
            self.relation_embedding = torch.zeros(self.nrelation, args.hidden_dim * 2).to(args.gpu).requires_grad_()
        else:
            self.relation_embedding = torch.zeros(self.nrelation, args.hidden_dim).to(args.gpu).requires_grad_()
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )

        # dataloader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size = args.batch_size,
            shuffle = True,
            collate_fn = TrainDataset.collate_fn
        )

        if args.setting == 'Collection':
            self.valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=args.test_batch_size,
                collate_fn=TestDataset_Entire.collate_fn
            )

            self.test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.test_batch_size,
                collate_fn=TestDataset_Entire.collate_fn
            )
        elif args.setting == 'Isolation':
            self.valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=args.test_batch_size,
                collate_fn=TestDataset.collate_fn
            )

            self.test_dataloader = DataLoader(
                test_dataset,
                batch_size = args.test_batch_size,
                collate_fn=TestDataset.collate_fn
            )
        elif args.setting == 'LLM':
            self.valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=args.test_batch_size,
                collate_fn=TestDataset_Entire_llm.collate_fn,
                shuffle=False
            )

            self.test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.test_batch_size,
                collate_fn=TestDataset_Entire_llm.collate_fn,
                shuffle=False
            )


#初始化kgeModel类
        self.kge_model = KGEModel(args, args.model)
        if args.setting == 'LLM':
            self.LLM_kge_model = LLMKGEModel(args,args.LLMModel)

        self.optimizer = torch.optim.Adam(
            [{'params': self.entity_embedding},
             {'params': self.relation_embedding}], lr=args.lr
        )


    def before_test_load(self):
        state = torch.load(os.path.join(self.args.state_dir, self.args.name + '.best'),
                           map_location=self.args.gpu)
        self.relation_embedding = state['rel_emb']
        self.entity_embedding = state['ent_emb']

    def write_training_loss(self, loss, e):
        self.args.writer.add_scalar("training/loss", loss, e)

    def write_evaluation_result(self, results, e):
        self.args.writer.add_scalar("evaluation/mrr", results['mrr'], e)
        self.args.writer.add_scalar("evaluation/hits10", results['hits@10'], e)
        self.args.writer.add_scalar("evaluation/hits5", results['hits@5'], e)
        self.args.writer.add_scalar("evaluation/hits1", results['hits@1'], e)

    def save_checkpoint(self, e):
        state = {'rel_emb': self.relation_embedding,
                 'ent_emb': self.entity_embedding}
        # delete previous checkpoint
        for filename in os.listdir(self.args.state_dir):
            if self.args.name in filename.split('.') and os.path.isfile(os.path.join(self.args.state_dir, filename)):
                os.remove(os.path.join(self.args.state_dir, filename))
        # save current checkpoint
        torch.save(state, os.path.join(self.args.state_dir,
                                       self.args.name + '.' + str(e) + '.ckpt'))

    def save_model(self, best_epoch):
        os.rename(os.path.join(self.args.state_dir, self.args.name + '.' + str(best_epoch) + '.ckpt'),
                  os.path.join(self.args.state_dir, self.args.name + '.best'))


#llm不需要训练，仅collection、isolation
    def train(self):
        best_epoch = 0
        best_mrr = 0
        bad_count = 0

        for epoch in range(self.args.max_epoch):
            losses = []
            #注意这里仅仅是将模型设置为训练模式，没有开始训练！！
            self.kge_model.train()
            for batch in self.train_dataloader:

                positive_sample, negative_sample, _ = batch

                positive_sample = positive_sample.to(self.args.gpu)
                negative_sample = negative_sample.to(self.args.gpu)
#pytorch nn里面的forward函数就是前向传播
#那么这里第一个negetive_score就是直接获得了打分
                negative_score = self.kge_model((positive_sample, negative_sample),
                                                  self.relation_embedding,
                                                  self.entity_embedding)

                negative_score = (F.softmax(negative_score * self.args.adversarial_temperature, dim=1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim=1)

                positive_score = self.kge_model(positive_sample,
                                                self.relation_embedding, self.entity_embedding, neg=False)


                positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

                positive_sample_loss = - positive_score.mean()
                negative_sample_loss = - negative_score.mean()

                loss = (positive_sample_loss + negative_sample_loss) / 2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

            if epoch % self.args.log_per_epoch == 0:
                logging.info('epoch: {} | loss: {:.4f}'.format(epoch, np.mean(losses)))
                self.write_training_loss(np.mean(losses), epoch)

            if epoch % self.args.check_per_epoch == 0:
                if self.args.setting == 'Collection':
                    eval_res = self.evaluate_multi()
                elif self.args.setting == 'Isolation':
                    eval_res = self.evaluate()
                self.write_evaluation_result(eval_res, epoch)

                if eval_res['mrr'] > best_mrr:
                    best_mrr = eval_res['mrr']
                    best_epoch = epoch
                    logging.info('best model | mrr {:.4f}'.format(best_mrr))
                    self.save_checkpoint(epoch)
                    bad_count = 0
                else:
                    bad_count += 1
                    logging.info('best model is at round {0}, mrr {1:.4f}, bad count {2}'.format(
                        best_epoch, best_mrr, bad_count))

            if bad_count >= self.args.early_stop_patience:
                logging.info('early stop at round {}'.format(epoch))
                break

        logging.info('finish training')
        logging.info('save best model')
        self.save_model(best_epoch)

        logging.info('eval on test set')
        self.before_test_load()
        if self.args.setting == 'Collection':
            eval_res = self.evaluate_multi(eval_split='test')
        elif self.args.setting == 'Isolation':
            eval_res = self.evaluate(eval_split='test')

#collection的eval函数，这里前面默认取的是best模型
    def evaluate_multi(self, eval_split='valid'):

        if eval_split == 'test':
            dataloader = self.test_dataloader
        elif eval_split == 'valid':
            dataloader = self.valid_dataloader
        #自动生成的字典且元素是列表
        client_ranks = ddict(list)
        all_ranks = []
        for batch in dataloader:

            triplets, labels, triple_idx = batch #[16,3],[16,14541],[16]
            #print(triple_idx.shape)
            triplets, labels = triplets.to(self.args.gpu), labels.to(self.args.gpu)
            head_idx, rel_idx, tail_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            #得到score
            pred = self.kge_model((triplets, None),
                                   self.relation_embedding,
                                   self.entity_embedding)# [16,14541]!形状的意义？16是测试集batch大小。我傻掉了，kgemodel里的score也是这个形状
            #我测了你这种写法，这里原来是用的else里面负样本集是none的情况，意义是获得所有实体作为tail的打分（head和relation是固定）
            #每一个三元组都有14541个打分
            #print(pred.shape)
            #print(tail_idx.shape)
            #一个16维的0到16整数的一维数组
            #print(type(pred))
            b_range = torch.arange(pred.size()[0], device=self.args.gpu)

            #就是取tail_idx对应实体作为tail的score，相当于正例（1个正确的，n-1个错误的）
            target_pred = pred[b_range, tail_idx] #16的一维数组
            # if len(tail_idx)==5:
            #     print(pred)
            #     print(b_range)
            #     print(target_pred)
            #print(target_pred)
            #这行代码的目的是将 pred 中对应于 labels 中 True 值的位置的数值替换为 -10000000 为了排除一对多的和在训练集里出现过的实体
            #保证true三元组的唯一性
            print(labels.shape)
            pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
            pred[b_range, tail_idx] = target_pred

            #先降序再升序得到本batch中正确三元组的排名
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                      dim=1, descending=False)[b_range, tail_idx]

            ranks = ranks.float()
            # 012345
            # 528631
            # 230415
            # 240135 升序排列的序号 还真是，这样获得了真正的rank


            for i in range(self.args.num_client):
                client_ranks[i].extend(ranks[triple_idx == i].tolist())

            all_ranks.extend(ranks.tolist())

        for i in range(self.args.num_client):
            results = ddict(float)
            ranks = torch.tensor(client_ranks[i])
            count = torch.numel(ranks)
            results['count'] = count
            results['mr'] = torch.sum(ranks).item() / count
            results['mrr'] = torch.sum(1.0 / ranks).item() / count
            for k in [1, 5, 10]:
                results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k]) / count
            logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
                results['mrr'], results['hits@1'],
                results['hits@5'], results['hits@10']))

        results = ddict(float)
        ranks = torch.tensor(all_ranks)
        count = torch.numel(ranks)
        results['count'] = count
        results['mr'] = torch.sum(ranks).item() / count
        results['mrr'] = torch.sum(1.0 / ranks).item() / count
        for k in [1, 5, 10]:
            results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k]) / count
        logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        return results
#isolation的eval函数
    def evaluate(self, eval_split='valid'):
        results = ddict(float)

        if eval_split == 'test':
            dataloader = self.test_dataloader
        elif eval_split == 'valid':
            dataloader = self.valid_dataloader

        pred_list = []
        rank_list = []
        results_list = []
        for batch in dataloader:
            triplets, labels = batch
            triplets, labels = triplets.to(self.args.gpu), labels.to(self.args.gpu)
            head_idx, rel_idx, tail_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            pred = self.kge_model((triplets, None),
                                  self.relation_embedding,
                                  self.entity_embedding)
            b_range = torch.arange(pred.size()[0], device=self.args.gpu)
            target_pred = pred[b_range, tail_idx]
            pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
            pred[b_range, tail_idx] = target_pred

            pred_argsort = torch.argsort(pred, dim=1, descending=True)
            ranks = 1 + torch.argsort(pred_argsort, dim=1, descending=False)[b_range, tail_idx]

            pred_list.append(pred_argsort[:, :10])
            rank_list.append(ranks)

            ranks = ranks.float()

            for idx, tri in enumerate(triplets):
                results_list.append([tri.tolist(), ranks[idx].item()])

            count = torch.numel(ranks)
            results['count'] += count
            results['mr'] += torch.sum(ranks).item()
            results['mrr'] += torch.sum(1.0 / ranks).item()

            for k in [1, 5, 10]:
                results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

        torch.save(torch.cat(pred_list, dim=0), os.path.join(self.args.state_dir,
                                                             self.args.name + '_' + str(self.args.one_client_idx) + '.pred'))
        torch.save(torch.cat(rank_list), os.path.join(self.args.state_dir,
                                                      self.args.name + '_' + str(self.args.one_client_idx) + '.rank'))

        for k, v in results.items():
            if k != 'count':
                results[k] /= results['count']

        logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        test_rst_file = os.path.join(self.args.log_dir, self.args.name + '.test.rst')
        pickle.dump(results_list, open(test_rst_file, 'wb'))

        return results
    


#llm的eval函数，metrics为MRR、hit@，这个版本每获得一个测试数据的还是计算的135个实体的打分
    def LLMevaluate_MRR(self, eval_split='valid'):
        
        if eval_split == 'test':
            dataloader = self.test_dataloader
        elif eval_split == 'valid':
            dataloader = self.valid_dataloader
        #自动生成的字典且元素是列表
        client_ranks = ddict(list)
        all_ranks = []


                # 初始化空列表来存储所有批次的数据
        all_triplets = []
        all_labels = []
        all_triple_idx = []
        all_one2n = []
        all_neighbors = []
        # 设置计数器
        batch_count = 0

        # 循环迭代数据加载器，并收集每个批次的数据
        for batch in dataloader:
            triplets, labels, triple_idx, one2n, neighbor= batch
            all_triplets.append(triplets)
            all_labels.append(labels)
            all_triple_idx.append(triple_idx)
            all_one2n.append(one2n)    
            all_neighbors.append(neighbor)    
            # 增加计数器
            batch_count += 1
            
            # 如果计数器达到三，则退出循环
            if batch_count >= 1:
                break
        #print(type(all_one2n[0])) #65个len10的list的集合
        #print(type(all_one2n[0][0]))
        # 使用 torch.cat 将列表中的张量连接起来
        all_triplets = torch.cat(all_triplets, dim=0)

        all_labels = torch.cat(all_labels, dim=0)
        all_triple_idx = torch.cat(all_triple_idx, dim=0)
        print(all_triple_idx)
        print(len(all_triple_idx))       
        # print(len(all_triplets))
        # print(len(all_labels))
        # print(len(all_triple_idx))
        # print(len(all_one2n))
        # print(len(all_neighbors))
        all_one2n = [item for sublist in all_one2n for item in sublist]
        all_neighbors = [item for sublist in all_neighbors for item in sublist]
        # 这里每行是那个测试集三元组head的所有邻居
        print("wolaikankan")
        print(all_neighbors[0])
        print(len(all_neighbors))

        #print(all_triplets.shape)
        #[30000,14541]

        #print(f'{self.args.LLMModel}_{self.args.num_triplets}_{self.args.num_threads}pred.txt')

        #all_triplets, all_labels, all_triple_idx = batch #[16,3],[16,14541],[16]

        #all_triplets, all_labels = all_triplets.to(self.args.gpu), all_labels.to(self.args.gpu)
        head_idx, rel_idx, tail_idx = all_triplets[:, 0], all_triplets[:, 1], all_triplets[:, 2]
        # print("看看正确实体的位置对不对")
        # print(tail_idx)
        #得到score
        all_triplets = all_triplets.tolist()
        pred = self.LLM_kge_model.forward((all_triplets, None), all_one2n, all_neighbors, self.h2rt_all)
        # 打开文件并将pred写入文件中
        with open(f'results/{self.args.name}_{self.args.num_triplets}_{self.args.num_threads}_pred.txt', 'w') as f:
            for item in pred:
                f.write(str(item) + '\n')
        pred = torch.tensor(pred)

        # print(pred.shape)
        # print(pred.shape)
        # print(pred.shape)
        # print(pred.shape)
        #pred = torch.rand((len(all_triple_idx), 14541),device=self.args.gpu)
        #print(tail_idx.shape)

        #一个16维的0到16整数的一维数组
        b_range = torch.arange(pred.size()[0], device=self.args.gpu)
        #就是取tail_idx对应实体作为tail的score，相当于正例（1个正确的，n-1个错误的）
        target_pred = pred[b_range, tail_idx] #16的一维数组
        #print(target_pred)
        #这行代码的目的是将 pred 中对应于 labels 中 True 值的位置的数值替换为 -10000000 为了排除某些实体，不是很懂
        print(all_labels.shape)
        pred = torch.where(all_labels.byte(), -torch.ones_like(pred) * 10000000, pred)
        pred[b_range, tail_idx] = target_pred

        #print(torch.argsort(pred, dim=1, descending=True)[0])
        #print(torch.argsort(torch.argsort(pred, dim=1, descending=True),dim=1, descending=False)[0])
        #先降序再升序得到本batch中正确三元组的排名
        ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                    dim=1, descending=False)[b_range, tail_idx]

        ranks = ranks.float()
        # print("获得正确三元组在当前一行中的打分")
        # print(pred[b_range, tail_idx])

        # 将正确三元组在当前一行中的打分写入文件中
        with open(f'results/{self.args.name}_{self.args.num_triplets}_{self.args.num_threads}_rightscore.txt', 'w') as f:
            for item in pred[b_range, tail_idx]:
                f.write(str(item) + '\n')
        # print("这是正确三元组在当前一行中的排名\n")
        # print(ranks)

        # 将正确三元组在当前一行中的打分写入文件中
        with open(f'results/{self.args.name}_{self.args.num_triplets}_{self.args.num_threads}_rightrank.txt', 'w') as f:
            for item in ranks:
                f.write(str(item) + '\n')
        # 012345
        # 528631
        # 230415
        # 240135 升序排列的序号 还真是，这样获得了真正的rank


        for i in range(self.args.num_client):
            client_ranks[i].extend(ranks[all_triple_idx == i].tolist())
        #把ranks所有元素添加到all_ranks末尾
        all_ranks.extend(ranks.tolist())
        print(all_ranks)
        for i in range(self.args.num_client):
            results = ddict(float)
            ranks = torch.tensor(client_ranks[i])
            count = torch.numel(ranks)
            results['count'] = count
            results['mr'] = torch.sum(ranks).item() / count
            results['mrr'] = torch.sum(1.0 / ranks).item() / count
            for k in [1, 5, 10]:
                results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k]) / count
            logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
                results['mrr'], results['hits@1'],
                results['hits@5'], results['hits@10']))

        results = ddict(float)
        ranks = torch.tensor(all_ranks)
        count = torch.numel(ranks)
        results['count'] = count
        results['mr'] = torch.sum(ranks).item() / count
        results['mrr'] = torch.sum(1.0 / ranks).item() / count
        for k in [1, 5, 10]:
            results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k]) / count
        logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        return results
    




#llm的eval函数，metrics为待定，加入了neo4j数据库操作
    def LLMevaluate_PRAUC(self, eval_split='valid'):
        


        # Neo4j数据库连接参数
        uri = "bolt://137.132.92.43:7687"
        username = "neo4j"
        password = "Xiao001112"  # 请替换为你的实际密码

        # 连接到Neo4j数据库
        graph = Graph(uri, auth=(username, password))



        if eval_split == 'test':
            dataloader = self.test_dataloader
        elif eval_split == 'valid':
            dataloader = self.valid_dataloader
        #自动生成的字典且元素是列表
        client_ranks = ddict(list)
        all_ranks = []

        # 初始化空列表来存储所有批次的数据
        all_triplets = []
        all_labels = []
        all_triple_idx = []
        all_one2n = []
        all_neighbors = []
        # # 设置计数器
        # batch_count = 0

        # 循环迭代数据加载器，并收集每个批次的数据
        for batch in dataloader:
            triplets, labels, triple_idx, one2n, neighbor= batch
            all_triplets.append(triplets)
            all_labels.append(labels)
            all_triple_idx.append(triple_idx)
            all_one2n.append(one2n)    
            all_neighbors.append(neighbor)    
            # # 增加计数器
            # batch_count += 1
            
            # # 如果计数器达到三，则退出循环
            # if batch_count >= 1:
            #     break
        #print(type(all_one2n[0])) #65个len10的list的集合
        #print(type(all_one2n[0][0]))
        # 使用 torch.cat 将列表中的张量连接起来
        all_triplets = torch.cat(all_triplets, dim=0)

        all_labels = torch.cat(all_labels, dim=0)
        all_triple_idx = torch.cat(all_triple_idx, dim=0)

        all_one2n = [item for sublist in all_one2n for item in sublist]
        all_neighbors = [item for sublist in all_neighbors for item in sublist]


        #all_triplets, all_labels = all_triplets.to(self.args.gpu), all_labels.to(self.args.gpu)
        head_idx, rel_idx, tail_idx = all_triplets[:, 0], all_triplets[:, 1], all_triplets[:, 2]
        # 将所有的pred分开计算

        for client_idx in range(3):#self.args.num_client):
            client_mask = (all_triple_idx == client_idx)
            if client_mask.sum() == 0:
                continue

            client_triplets = all_triplets[client_mask]
            client_labels = all_labels[client_mask]
            client_one2n = [all_one2n[i] for i in range(len(all_one2n)) if client_mask[i]]
            client_neighbors = [all_neighbors[i] for i in range(len(all_neighbors)) if client_mask[i]]
            client_rel_idx = rel_idx[client_mask]
            client_h2rt = ddict(list)
            for h, r, t in self.all_client_data[client_idx]:
                client_h2rt[h].append([r, t])
            # print(len(self.all_client_data))
            # print(len(client_triplets))
            # print(len(client_labels))
            # print(len(client_one2n))
            # print(len(client_neighbors))
            # print(len(client_tail_idx))
            # print(len(client_h2rt))
            # 得到score
            client_triplets = client_triplets.tolist()
            pred = self.LLM_kge_model.forward_PRAUC(client_triplets, self.all_client_data, client_idx)
            pred = torch.tensor(pred, device=self.args.gpu)

            # 保存张量到文件
            torch.save(pred, f'{self.args.name}_{self.args.num_triplets}_{self.args.num_threads}_client{client_idx}.pt')
            print("看一下pred长度", pred.size())
            print(pred.shape)
            print(pred)


        # for i in range(self.args.num_client):
        #     if len(client_ranks[i]) == 0:
        #         continue
        #     results = ddict(float)
        #     ranks = torch.tensor(client_ranks[i])
        #     count = torch.numel(ranks)
        #     results['count'] = count
        #     results['mr'] = torch.sum(ranks).item() / count
        #     results['mrr'] = torch.sum(1.0 / ranks).item() / count
        #     for k in [1, 5, 10]:
        #         results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k]) / count
        #     logging.info('Client {}: mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
        #         i, results['mrr'], results['hits@1'],
        #         results['hits@5'], results['hits@10']))

        # # 计算所有客户端的总体结果
        # all_ranks = [rank for client in client_ranks.values() for rank in client]
        # results = ddict(float)
        # ranks = torch.tensor(all_ranks)
        # count = torch.numel(ranks)
        # results['count'] = count
        # results['mr'] = torch.sum(ranks).item() / count
        # results['mrr'] = torch.sum(1.0 / ranks).item() / count
        # for k in [1, 5, 10]:
        #     results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k]) / count
        # logging.info('Overall: mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
        #     results['mrr'], results['hits@1'],
        #     results['hits@5'], results['hits@10']))

        #return results