# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle

nlp = spacy.load('en_core_web_sm')

LSTM_length = 10

def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    
    for token in document:
        if token.i < seq_len:
            # matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1
            
            # update 2022.05.13
            # we try to add more edges in the matrix
            # to mimic the LSTM (or global information)
            # for l in range(max(0, token.i - LSTM_length), min(seq_len, token.i + LSTM_length)):
            #     matrix[token.i][l] = 1

    return matrix

def anonymous(sequence_matrix):
    anonymous_matrix = np.zeros_like(sequence_matrix)
    for i in range(len(sequence_matrix)):
        dic = {}
        h = 1
        for l in range(len(sequence_matrix[i])):
            
            if sequence_matrix[i, l] in dic:
                anonymous_idx = dic[sequence_matrix[i, l]]
            else:
                dic[sequence_matrix[i, l]] = h
                anonymous_idx = h
                h += 1
            anonymous_matrix[i,l] = anonymous_idx

    return anonymous_matrix
        

def sample_sequences_from_matrix(adj_matrix, start_idx, sequence_num, length_num):
    # 3*3*3*3, each time 
    # step = [3,3,3,3]
    sequence_matrix = np.zeros((sequence_num, length_num)) 
    random_matrix = np.random.rand(sequence_num, length_num)
    idx_array = np.arange(len(adj_matrix))
    # idx_array_temp = np.arange(len(adj_matrix))
    for i in range(sequence_num):
        h = 0
        p = start_idx
        if adj_matrix[p].sum() == 0:
            for h in range(length_num):
                sequence_matrix[i][h] = p
            continue
        for h in range(length_num):
            sequence_matrix[i][h] = p
            
            # import pdb; pdb.set_trace()
            try:
                # total_neighbour = idx_array[(adj_matrix[p] > 0) * (idx_array != p)]
                total_neighbour = idx_array[(adj_matrix[p] > 0)]
                # if len(total_neighbour) == 0:
                #     p = total_neighbour[0]
                # else:
                p = total_neighbour[int(random_matrix[i, h] / (1.0 / adj_matrix[p].sum()))]
            except:
                import pdb; pdb.set_trace()
                print("------error------")
    
    return sequence_matrix

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    idx2sequence = {}
    idx2DE = {}
    fout = open(filename+f'.graph', 'wb')
    fout_seq = open(filename+f'.sequence', 'wb')
    fout_de = open(filename+f'.de', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
        idx2graph[i] = adj_matrix
        # import pdb; pdb.set_trace()
        st = len(text_left.split(' '))-1
        # if adj_matrix[st].sum() == 0:
        #     import pdb; pdb.set_trace()
        #     print("wrong")
        idx2sequence[i] = sample_sequences_from_matrix(adj_matrix=adj_matrix, start_idx=st, sequence_num=128, length_num=10)
        idx2DE[i] = anonymous(idx2sequence[i])
    pickle.dump(idx2graph, fout)
    pickle.dump(idx2sequence, fout_seq)
    pickle.dump(idx2DE, fout_de)
    fout.close() 

if __name__ == '__main__':
    process('./datasets/acl-14-short-data/train.raw')
    process('./datasets/acl-14-short-data/test.raw')
    process('./datasets/semeval14/restaurant_train.raw')
    process('./datasets/semeval14/restaurant_test.raw')
    process('./datasets/semeval14/laptop_train.raw')
    process('./datasets/semeval14/laptop_test.raw')
    process('./datasets/semeval15/restaurant_train.raw')
    process('./datasets/semeval15/restaurant_test.raw')
    process('./datasets/semeval16/restaurant_train.raw')
    process('./datasets/semeval16/restaurant_test.raw')