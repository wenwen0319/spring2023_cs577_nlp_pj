# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                continue
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type, opt):
    embedding_matrix_file_name = '{0}_{1}_{2}_embedding_matrix_original.pkl'.format(str(embed_dim), type, opt.model_name) # may change here if rand/consistant
    # embedding_matrix_file_name = '{0}_{1}_embedding_matrix_rand.pkl'.format(str(embed_dim), type) # may change here if rand/consistant
    # embedding_matrix_file_name = '{0}_{1}_embedding_matrix_consistant.pkl'.format(str(embed_dim), type) # may change here if rand/consistant
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))

        # fname = './glove/glove.840B.300d.txt'
        fname = '/home/liu3154/code/AllSet/data/scratch/CS577_project/glove.840B.300d.txt'
        # fname = './vocab_bert.txt'

        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():

            vec = word_vec.get(word)
            # vec = np.random.rand(embed_dim)
            # vec = np.ones(embed_dim)

            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        # for w in words:
        #     id = self.word2idx[w]
        #     if id not in self.idx2word:
        #         self.idx2word[id] = w
        if len(sequence) == 0:
            sequence = [0]
        return sequence
    
    def seq_to_text(self, seq):
        # text = text.lower()
        # words = text.split()
        # unknownidx = 1
        # import pdb; pdb.set_trace()
        seq = seq.cpu().numpy().tolist()
        sequence_batch = []
        for _seq in seq:
            sequence_batch.append([self.idx2word[int(w)] for w in _seq])
        # if len(sequence) == 0:
        #     sequence = [0]
        return sequence_batch
    
    # def seq_to_text_onesentence(self, seq):
    #     # text = text.lower()
    #     # words = text.split()
    #     # unknownidx = 1
    #     # import pdb; pdb.set_trace()
    #     seq = seq.cpu().numpy().tolist()
    #     sequence_batch = []
    #     for _seq in seq:
    #         sequence_batch.append([self.idx2word[int(w)] for w in _seq])
    #     # if len(sequence) == 0:
    #     #     sequence = [0]
    #     return sequence_batch



class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname+'.graph', 'rb')
        # fin = open(fname + '_label.graph', 'rb')
        idx2gragh = pickle.load(fin)
        fin.close()

        fin = open(fname+'.de', 'rb')
        idx2de = pickle.load(fin)
        fin.close()

        fin = open(fname+'.sequence', 'rb')
        idx2sequence = pickle.load(fin)
        fin.close()

        # changed by wenwen
        raw_text_output = {}
        raw_text_output['train'] = []
        raw_text_output['test'] = []

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]

            raw_text_output['test'].append(lines[i][:-1]+lines[i+1]) # wenwen

            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            # import pdb; pdb.set_trace()
            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_indices = tokenizer.text_to_sequence(text_left)
            polarity = int(polarity)+1
            dependency_graph = idx2gragh[i]
            de_emb = idx2de[i]
            text_pos = len(left_indices) - 1

            try:
                seq_matrix = np.array(text_indices)[idx2sequence[i].astype(int)]
                # text_indices[int(x)]
                # seq_matrix = [text_indices[int(x)] for x in idx2sequence[i]]
            except:
                import pdb; pdb.set_trace()
                print("--------wrong----------")
            
            # import pdb;pdb.set_trace()
            data = {
                'text_indices': text_indices,
                'context_indices': context_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
                # TODO: add distance encoding matrix and random walk 
                'de': de_emb,
                'seq': seq_matrix,
                'text_pos': text_pos,
            }

            all_data.append(data)
        # return all_data
        # import pdb; pdb.set_trace()
        return all_data, raw_text_output # changed by wenwen

    def __init__(self, dataset='twitter', embed_dim=300, opt=None):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'rest15': {
                'train': './datasets/semeval15/restaurant_train.raw',
                'test': './datasets/semeval15/restaurant_test.raw'
            },
            'rest16': {
                'train': './datasets/semeval16/restaurant_train.raw',
                'test': './datasets/semeval16/restaurant_test.raw'
            },

        }
        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        
        self.raw_text = text # wenwen
        self.raw_text_output = {}
        self.raw_text_output['train'] = []
        self.raw_text_output['test'] = []
        
        if os.path.exists(dataset+'_word2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset+'_word2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(dataset+'_word2idx.pkl', 'wb') as f:
                 pickle.dump(tokenizer.word2idx, f)
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset, opt)
        self.train_data, _ = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer))
        self.test_data, self.raw_text_output = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer))
        self.tokenizer = tokenizer
        
        # self.train_data_raw = ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer)
        # self.test_data_raw = ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer)
    