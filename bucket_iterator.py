# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True, get_idx=0):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.get_idx = get_idx
        self.graph_list = []


        self.batches = self.sort_and_pad(data, batch_size=4)
        self.batch_len = len(self.batches)

        print('finished BucketIterator')

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))

        # get the len of the complete sample
        data_len = len(data)
        for i in range(data_len):
            data[i]['sort_idx'] = i

        # return data

        # if self.sort:
        #     sorted_data = sorted(data, key=lambda x: len(x[self.sort_key])) # sort all the data in the list

        #     # # ========================
        #     # # save the testing idx
        #     # # ========================
        #     # sorted_idx_str = ''
        #     # if self.get_idx == 1:
        #     #     for i in range(data_len):
        #     #         print(sorted_data[i]['sort_idx'])
        #     #         sorted_idx_str = sorted_idx_str + str(sorted_data[i]['sort_idx']) + '\n'
        #     #
        #     #     # save the all sentences
        #     #     test_save_name = './case_study/sorted_idx_' + 'twitter' + '.txt'
        #     #     fo = open(test_save_name, "w")
        #     #     fo.write(sorted_idx_str)
        #     #     fo.close()

        # else:
        #     sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        # import pdb ;pdb.set_trace()
        batch_text_indices = []
        batch_context_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_polarity = []
        batch_dependency_graph = []
        batch_de = []
        batch_seq_matrix = []
        batch_text_pos = []
        max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            text_indices, context_indices, aspect_indices, left_indices, polarity, dependency_graph, de, seq_matrix, text_pos = \
                item['text_indices'], item['context_indices'], item['aspect_indices'], item['left_indices'],\
                item['polarity'], item['dependency_graph'], item['de'], item['seq'], item['text_pos']

            # save each dependency graph
            self.graph_list.append(dependency_graph)


            text_padding = [0] * (max_len - len(text_indices))
            context_padding = [0] * (max_len - len(context_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))
            batch_text_indices.append(text_indices + text_padding)
            batch_context_indices.append(context_indices + context_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_left_indices.append(left_indices + left_padding)
            batch_polarity.append(polarity)
            batch_dependency_graph.append(numpy.pad(dependency_graph, \
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant'))
            batch_de.append(de)
            batch_seq_matrix.append(seq_matrix)
            batch_text_pos.append(text_pos)
            


        return { \
                'text_indices': torch.tensor(batch_text_indices), \
                'context_indices': torch.tensor(batch_context_indices), \
                'aspect_indices': torch.tensor(batch_aspect_indices), \
                'left_indices': torch.tensor(batch_left_indices), \
                'polarity': torch.tensor(batch_polarity), \
                'dependency_graph': torch.tensor(batch_dependency_graph), \
                'de': torch.tensor(batch_de).float(), \
                'seq': torch.tensor(batch_seq_matrix).long(), \
                'text_pos': torch.tensor(batch_text_pos).long()
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
