# -*- coding: utf-8 -*-
import os
import math
import argparse
import random
import numpy
import torch
import torch.nn as nn
from bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils_DE import ABSADatesetReader
# from models import LSTM, ASCNN, ASGCN, kdd_yunyu, nips_yunyu

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, opt=opt)
        self.absa_dataset = absa_dataset
        # self.train_data_raw = absa_dataset.train_data_raw # wenwen
        # self.test_data_raw = absa_dataset.test_data_raw # wenwen
        self.raw_text_output = {}
        self.raw_text_output = absa_dataset.raw_text_output
        # import pdb; pdb.set_trace()
        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=1, shuffle=True)
        # self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=1, shuffle=False)

        print(len(self.train_data_loader.graph_list), len(self.test_data_loader.graph_list))

        # self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
        # print(opt.model_class)
        # self.model = opt.model_class.nips_yunyu(absa_dataset.embedding_matrix, opt) #.to(opt.device)
        from models_DE import DE, DE_LSTM, DE_LSTM_exp
        print("initialize model")
        if opt.model_name == 'DE':
            self.model = DE(input_dim=301, hidden_dim=50, output_dim=3, embedding_matrix=absa_dataset.embedding_matrix, opt=opt).to(opt.device)
        elif opt.model_name == 'DE_LSTM':
            self.model = DE_LSTM(input_dim=301, hidden_dim=50, output_dim=3, embedding_matrix=absa_dataset.embedding_matrix, opt=opt).to(opt.device)
        elif opt.model_name == 'DE_LSTM_exp':
            self.model = DE_LSTM_exp(input_dim=301, hidden_dim=50, output_dim=3, embedding_matrix=absa_dataset.embedding_matrix, opt=opt).to(opt.device)
        # self._print_args()
        self.global_f1 = 0.
        print("Finish model")
        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self):

        # obtain and print the highest performance
        max_test_acc = 0
        max_test_f1 = 0

        global_step = 0

        loop_count = 0

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0)

        # import pdb; pdb.set_trace()
        # start training the model
        next_time_output = False
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            increase_flag = False

            # ===================
            # training dataset setup:
            for i_batch, sample_batched in enumerate(self.train_data_loader): # datasets for evaluation training/testing sets
            # for i_batch, sample_batched in enumerate(self.test_data_loader):

                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                # arrange the feature and ground-truth label
                # import pdb; pdb.set_trace()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)

                if self.opt.model_name == "DE_LSTM_exp":
                    outputs, _ = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                # import pdb; pdb.set_trace()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                # print(outputs, targets)
                train_batch = len(sample_batched)
                # n_correct += (torch.argmax(outputs, -1) == targets.repeat([train_batch, 1]).view(-1)).sum().item()
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(targets)
                # print(outputs)
                # print(torch.argmax(outputs, -1))


                if global_step % self.opt.log_step == 0:

                    # print(torch.argmax(outputs, -1), targets)
                    train_acc = n_correct / n_total

                    test_acc, test_f1 = self._evaluate_acc_f1(next_time_output)
                    next_time_output = False
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        next_time_output = True
                    if test_f1 > max_test_f1:
                    #     increase_flag = True
                        max_test_f1 = test_f1
                        next_time_output = True
                    #     if self.opt.save and test_f1 > self.global_f1:
                    #         self.global_f1 = test_f1
                    #         torch.save(self.model.env.classifier.state_dict(), 'state_dict/'+self.opt.model_name+'_classifier_'+self.opt.dataset+'.pkl')
                    #         torch.save(self.model.env.rnn.state_dict(), 'state_dict/' + self.opt.model_name + '_rnn_' + self.opt.dataset + '.pkl')
                    #         print('>>> best model saved.')
                    # print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), train_acc, test_acc, test_f1))
                    print('train_correct: {:d}, train_num: {:d}, train_acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}, MAX_acc: {:.4f}, MAX_f1: {:.4f}'.format(n_correct, n_total, train_acc, test_acc, test_f1, max_test_acc, max_test_f1))
                    n_correct, n_total = 0, 0

            # if increase_flag == False:
            #     continue_not_increase += 1
            #     if continue_not_increase >= 5:
            #         print('early stop.')
            #         break
            # else:
            #     continue_not_increase = 0
        # self._evaluate_acc_f1()
        return 1

    def _evaluate_acc_f1(self, next_time_output):
        # switch model to evaluation mode
        # self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None

        label_output = []
        if next_time_output:
            f = open(f'./{self.opt.dataset}_{self.opt.model_name}_exp.txt', 'w')
        with torch.no_grad():

            # ===================
            # testing dataset setup:
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
            # for t_batch, t_sample_batched in enumerate(self.train_data_loader):

                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]

                # =======================
                # the t_target->testing ground-truth  t_outputs->testing predicted
                # (they are only a batch)
                # =======================
                t_targets = t_sample_batched['polarity'].to(opt.device)
                # t_outputs = self.model(t_inputs)

                # ===== switch
                # t_outputs = self.model.pretrain_test(t_inputs, t_targets, opt.test_num) # pretrain testing
                # t_outputs = self.model.pretrain_test_nornn(t_inputs, t_targets, opt.test_num) # pretrain testing
                # t_outputs = self.model.run_test_nornn(t_inputs)  # policy network testing
                # t_outputs = self.model.pretrain_test_lstm(t_inputs, t_targets, opt.test_num)
                # t_outputs = self.model.test(t_inputs)
                if self.opt.model_name == "DE_LSTM_exp":
                    t_outputs, exp = self.model(t_inputs)
                    # import pdb; pdb.set_trace()
                    if next_time_output:
                        batch_size = t_outputs.size()[0]
                        exp = exp.reshape(batch_size, -1, 3)
                        solution = torch.argmax(exp, dim=1)
                        _t_outputs = torch.argmax(t_outputs, -1)
                        posnew = torch.stack((torch.arange(batch_size), _t_outputs.cpu()), 0).numpy()
                        path_list = solution[posnew]
                        posnew_2 = torch.stack((torch.arange(batch_size), path_list.cpu()), 0).numpy()
                        output_seq = self.absa_dataset.tokenizer.seq_to_text(t_inputs[1][posnew_2])
                        output_ori = self.absa_dataset.tokenizer.seq_to_text(t_inputs[0])
                        for i in range(batch_size):
                            f.write("================\n")
                            f.write("output: " + str(_t_outputs[i].cpu().numpy())+"\n")
                            f.write("gt:     " + str(t_targets[i].cpu().numpy())+"\n")
                            f.write(str(output_ori[i]))
                            f.write("\n")
                            f.write(str(output_seq[i]))
                            f.write("\n")
                            f.write("================\n\n")
                else:
                    t_outputs = self.model(t_inputs)

                # print(self.absa_dataset.tokenizer.seq_to_text(t_inputs[0]))
                label_output.append(torch.argmax(t_outputs, -1).cpu().numpy())
                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)
                t_correct_m = (torch.argmax(t_outputs, -1) == t_targets)

                # =======================
                # concat the batch together to get the whole samples
                # =======================
                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                    t_correct_all = t_correct_m
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
                    t_correct_all = torch.cat((t_correct_all, t_correct_m), dim=0) # get the whole correct 0/1

        # print(label_output)

        # # =======================
        # # print the ground-truth and predicted labels
        # # =======================
        # print(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), n_test_total)
        # print(t_targets_all.cpu(), len(torch.argmax(t_outputs_all, -1).cpu()), n_test_total)

        # ========================
        # switch to numpy and save the data
        # ========================
        # print(t_targets_all.cpu().data.numpy(), torch.argmax(t_outputs_all, -1).cpu(), n_test_total)
        list_target = t_targets_all.cpu().data.numpy()
        list_output = torch.argmax(t_outputs_all, -1).cpu().numpy()
        print_list = [[0,0,0], [0,0,0], [0,0,0]]
        for i in range(len(list_output)):
            print_list[list_target[i]][list_output[i]] += 1
        print(print_list[0])
        print(print_list[1])
        print(print_list[2])
        test_acc = n_test_correct / n_test_total
        # f1 = 0 # metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        if next_time_output:
            f.close()
        return test_acc, f1



    def run(self, repeats=3):
        # Loss and Optimizer
        # criterion = nn.CrossEntropyLoss()
        # _params = filter(lambda p: p.requires_grad, self.model.parameters())
        # optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        # save/create the log file
        if not os.path.exists('log/'):
            os.mkdir('log/')
        f_out = open('log/'+self.opt.model_name+'_'+self.opt.dataset+'_val.txt', 'w', encoding='utf-8')

        # save the best performance
        max_test_acc_avg = 0
        max_test_f1_avg = 0

        for i in range(repeats):
            print('repeat: ', (i+1))
            f_out.write('repeat: '+str(i+1))
            # self._reset_params()
            label_run_succes = self._train()
            # print(label_run_succes)
            # print('max_test_acc: {0}     max_test_f1: {1}'.format(max_test_acc, max_test_f1))
            # f_out.write('max_test_acc: {0}, max_test_f1: {1}'.format(max_test_acc, max_test_f1))
            # max_test_acc_avg += max_test_acc
            # max_test_f1_avg += max_test_f1
            # print('#' * 100)
        print("max_test_acc_avg:", max_test_acc_avg / repeats)
        print("max_test_f1_avg:", max_test_f1_avg / repeats)
        f_out.close()


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='DE', type=str)
    parser.add_argument('--dataset', default='rest15', type=str, help='twitter, rest14, lap14, rest15, rest16')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate_p', default=0.0001, type=float)
    parser.add_argument('--learning_rate_c', default=0.0001, type=float)
    parser.add_argument('--learning_rate_lstm', default=0.0001, type=float)

    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=1000, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--log_step', default=500, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--device', default=0, type=str)
    parser.add_argument('--load', default=0, type=int)
    parser.add_argument('--dead', default=1, type=int) # dead
    # parser.add_argument('--dead', default=1, type=int)  # dead
    parser.add_argument('--succ', default=1, type=int) # one step, classification right
    parser.add_argument('--fail', default=1, type=int) # one step, classification wrong
    parser.add_argument('--max_length', default=7, type=int) # step Length
    parser.add_argument('--test_num', default=2, type=int) # test n times, one right then right
    parser.add_argument('--tau', default=0.0001, type=float)  # test n times, one right then right
    opt = parser.parse_args()

    # model_classes = {
    #     'lstm': LSTM,
    #     'ascnn': ASCNN,
    #     'asgcn': ASGCN,
    #     'astcn': ASGCN,
    #     'kdd_yunyu': kdd_yunyu,
    #     'nips_yunyu': nips_yunyu,
    # }
    input_colses = {
        'lstm': ['text_indices'],
        'ascnn': ['text_indices', 'aspect_indices', 'left_indices'],
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'astcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'kdd_yunyu':['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'nips_yunyu':['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'DE': ['seq', 'de'],
        'DE_LSTM': ['text_indices', 'seq', 'de', 'text_pos'],
        'DE_LSTM_exp': ['text_indices', 'seq', 'de', 'text_pos']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    # opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins.run()
