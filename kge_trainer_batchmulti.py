import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import pickle
import logging
import numpy as np
from collections import defaultdict as ddict
from dataloader import get_task_dataset, get_task_dataset_entire, \
    TrainDataset, TestDataset, TestDataset_Entire
from kge_model import KGEModel
from LLM_kge_model import LLMKGEModel
import multiprocessing
import sys

#先研究这个，根据这个再弄个和isolation并列的类做比对，先别想联邦的问题
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
            train_dataset, valid_dataset, test_dataset, nrelation, nentity = get_task_dataset_entire(data, args)

        self.nentity = nentity
        self.nrelation = nrelation

        # embedding 向量维度姑且让两者都一致吧
        embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim]) # tensor([0.0938])
        #print(embedding_range)
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
                collate_fn=TestDataset_Entire.collate_fn
            )

            self.test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.test_batch_size,
                collate_fn=TestDataset_Entire.collate_fn
            )

        # model 在这里尝试加入LLMModel
        # if(args.usingLLM):
        #     self.kge_model = LLMKGEModel(args,args.LLMModel)
        # else:
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

#这里前面默认取的是best模型
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
            b_range = torch.arange(pred.size()[0], device=self.args.gpu)

            #就是取tail_idx对应实体作为tail的score，相当于正例（1个正确的，n-1个错误的）
            target_pred = pred[b_range, tail_idx] #16的一维数组
            # if len(tail_idx)==5:
            #     print(pred)
            #     print(b_range)
            #     print(target_pred)
            #print(target_pred)
            #这行代码的目的是将 pred 中对应于 labels 中 True 值的位置的数值替换为 -10000000 为了排除某些实体，不是很懂
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
    



    def LLMevaluate(self, eval_split='valid'):
        def process_batch(batch):
            triplets, labels, triple_idx = batch
            head_idx, rel_idx, tail_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            pred = self.LLM_kge_model((triplets, None))
            return pred, tail_idx, triple_idx, labels

        if eval_split == 'test':
            dataloader = self.test_dataloader
        elif eval_split == 'valid':
            dataloader = self.valid_dataloader
        #自动生成的字典且元素是列表
        client_ranks = ddict(list)
        all_ranks = []

        #print(multiprocessing.cpu_count()) 64
        #pool = multiprocessing.Pool(processes=1)

        def worker(batch):
            result = process_batch(batch)
            pred, tail_idx, triple_idx, labels = result
            b_range = torch.arange(pred.size()[0], device=self.args.gpu)
            target_pred = pred[b_range, tail_idx]
            pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
            pred[b_range, tail_idx] = target_pred
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, tail_idx]
            ranks = ranks.float()
            for i in range(self.args.num_client):
                client_ranks[i].extend(ranks[triple_idx == i].tolist())
            all_ranks.extend(ranks.tolist())

        # Redirect stdout to a file
        with open('output.log', 'w') as f:
            sys.stdout = f
            
            processes = []
            for batch in dataloader:
                process = multiprocessing.Process(target=worker, args=(batch,))
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

        # Restore stdout
        sys.stdout = sys.__stdout__



        if len(all_ranks) == 0:
            results = ddict(float)
            return results
        else:
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