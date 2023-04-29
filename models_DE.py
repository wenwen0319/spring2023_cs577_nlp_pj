import torch


# NOTE: In addition to __init__() and forward(), feel free to add
# other functions or attributes you might need.
class DAN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        # TODO: Declare DAN architecture
        super(DAN, self).__init__()
        self.model = torch.nn.ModuleList()
        if num_layers == 1:
            self.model.append(torch.nn.Linear(input_dim, output_dim))
        else:
            self.model.append(torch.nn.Linear(input_dim, hidden_dim))
            self.model.append(torch.nn.ReLU())
            if num_layers > 1:
                for i in range(1, num_layers-1):
                    self.model.append(torch.nn.Linear(hidden_dim, hidden_dim))
                    self.model.append(torch.nn.ReLU())
            self.model.append(torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, x, pos):
        # TODO: Implement DAN forward pass
        # pass
        # x[:,:, -1]
        # torch.mul(x[:,:,-1].unsqueeze(1), x)
        mean = 1 / x[:, :, -1].sum(-1)
        # weight = torch.matmul(x[:, :, -1].unsqueeze(1), mean)
        weight = x[:,:,-1] * mean.unsqueeze(1)
        y = torch.matmul(weight.unsqueeze(1), x).squeeze(1)
        for layer in self.model:
            y = layer(y)
        return y


class RNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False):
        # TODO: Declare RNN model architecture
        super(RNN, self).__init__()
        self.model = torch.nn.RNN(input_dim, hidden_dim, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, pos):
        # TODO: Implement RNN forward pass
        # pass
        # posnew = torch.stack((torch.arange(len(x)), pos))
        posnew = torch.stack((torch.arange(len(x)), pos), 0).numpy()
        p, _ = self.model(x)
        # return h_n[pos]

        return p[posnew]

class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False, device=None):
        # TODO: Declare LSTM model architecture
        super(LSTM, self).__init__()
        self.model = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=batch_first, bidirectional=bidirectional).to(device)

    def forward(self, x):
        # TODO: Implement LSTM forward pass
        # posnew = torch.stack((torch.arange(len(x)), pos), 0).numpy()
        import pdb; pdb.set_trace()
        p = self.model(x)[0]
        # return h_n[pos]
        print(p)
        # return p[posnew]
        return p[:, -1, :]

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(MLP, self).__init__()
        self.model = torch.nn.ModuleList()
        if num_layers == 1:
            self.model.append(torch.nn.Linear(input_dim, output_dim))
        else:
            self.model.append(torch.nn.Linear(input_dim, hidden_dim))
            self.model.append(torch.nn.ReLU())
            if num_layers > 1:
                for i in range(1, num_layers-1):
                    self.model.append(torch.nn.Linear(hidden_dim, hidden_dim))
                    self.model.append(torch.nn.ReLU())
            self.model.append(torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layers in self.model:
            x = layers(x)
        return x

class DE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, output_dim=3, batch_first=True, bidirectional=False, embedding_matrix=None, opt=None):
        # TODO: Declare LSTM model architecture
        super(DE, self).__init__()
        # self.model = LSTM(input_dim, hidden_dim, num_layers, batch_first=batch_first, bidirectional=bidirectional, device=opt.device)
        self.model = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, 
                                   num_layers=num_layers, batch_first=batch_first, 
                                   bidirectional=bidirectional)# .to(opt.device)

        print("start loading embedding")
        # import pdb; pdb.set_trace()
        self.embed = torch.nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))# .to(opt.device)
        print("finish loading embedding")
        self.cls = MLP(input_dim=hidden_dim, hidden_dim=30, output_dim=3, num_layers=2)
        
        # self.MLP = self.MLP.to(opt.device)

        self.Enc = MLP(input_dim=input_dim, hidden_dim=50, output_dim=hidden_dim, num_layers=2)
        # self.rnn = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, 
        #                          num_layers=num_layers, batch_first=batch_first, 
        #                          bidirectional=bidirectional)

    def forward(self, inputs):
        # TODO: Implement LSTM forward pass
        # import pdb; pdb.set_trace()
        x, de = inputs[0], inputs[1]
        x = self.embed(x)
        x = torch.cat((x, de.unsqueeze(-1)), -1)
        batch_size, seq_size, seq_length, word_dim = x.shape
        x = x.reshape((batch_size * seq_size, seq_length, word_dim))
        x = self.Enc(x)

        # input = torch.randn(batch_size * seq_size, seq_length, x.size()[-1]).to(x.device)
        # output, _ = self.rnn(input)
        
        outx, _ = self.model(x)
        outx = outx[:,-1,:].reshape(batch_size, seq_size, -1)
        y = outx.sum(1)
        final_output = self.cls(y)

        # return p[posnew]
        return final_output  # p[:, -1, :]

class DE_LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, output_dim=3, batch_first=True, bidirectional=False, embedding_matrix=None, opt=None):
        # TODO: Declare LSTM model architecture
        super(DE_LSTM, self).__init__()
        # self.model = LSTM(input_dim, hidden_dim, num_layers, batch_first=batch_first, bidirectional=bidirectional, device=opt.device)
        self.model = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, 
                                   num_layers=num_layers, batch_first=batch_first, 
                                   bidirectional=bidirectional)# .to(opt.device)

        print("start loading embedding")
        # import pdb; pdb.set_trace()
        self.embed = torch.nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))# .to(opt.device)
        print("finish loading embedding")
        self.cls = MLP(input_dim=2*hidden_dim, hidden_dim=30, output_dim=3, num_layers=3)
        
        # self.MLP = self.MLP.to(opt.device)

        self.Enc = MLP(input_dim=input_dim, hidden_dim=50, output_dim=hidden_dim, num_layers=3)
        self.Enc_rnn = MLP(input_dim=input_dim-1, hidden_dim=50, output_dim=hidden_dim, num_layers=3)
        self.rnn = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim // 2, 
                                 num_layers=num_layers, batch_first=batch_first, 
                                 bidirectional=True)

    def forward(self, inputs):
        # TODO: Implement LSTM forward pass
        # import pdb; pdb.set_trace()
        
        all_emb, x, de, pos = inputs[0], inputs[1], inputs[2], inputs[3]
        # batch_size, seq_size, seq_length = idx_in_x.shape
        # idx_x = torch.arange(batch_size).to(idx_in_x.device)
        # idx_x_new = idx_x.reshape(batch_size,1,1).repeat(1,seq_size,seq_length)
        # new_pos = torch.cat((idx_x_new, idx_in_x), 2)
        x = self.embed(x)
        x = torch.cat((x, de.unsqueeze(-1)), -1)
        batch_size, seq_size, seq_length, word_dim = x.shape
        x = x.reshape((batch_size * seq_size, seq_length, word_dim))
        x = self.Enc(x)

        # input = torch.randn(batch_size * seq_size, seq_length, x.size()[-1]).to(x.device)
        # output, _ = self.rnn(input)
        
        outx, _ = self.model(x)
        # import pdb; pdb.set_trace()
        all_emb = self.Enc_rnn(self.embed(all_emb))
        y_lstm, _  = self.rnn(all_emb)

        # import pdb; pdb.set_trace()
        posnew = torch.stack((torch.arange(batch_size), pos.cpu()), 0).numpy()
        outx = (outx[:,-1,:]).reshape(batch_size, seq_size, -1)
        y = torch.cat((outx.sum(1), y_lstm[posnew]), 1)
        final_output = self.cls(y)

        
        # return p[posnew]
        return final_output  # p[:, -1, :]


class DE_LSTM_exp(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, output_dim=3, batch_first=True, bidirectional=False, embedding_matrix=None, opt=None):
        # TODO: Declare LSTM model architecture
        super(DE_LSTM_exp, self).__init__()
        self.hidden_dim = hidden_dim
        # self.model = LSTM(input_dim, hidden_dim, num_layers, batch_first=batch_first, bidirectional=bidirectional, device=opt.device)
        self.model = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, 
                                   num_layers=num_layers, batch_first=batch_first, 
                                   bidirectional=bidirectional)# .to(opt.device)

        print("start loading embedding")
        # import pdb; pdb.set_trace()
        self.embed = torch.nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))# .to(opt.device)
        print("finish loading embedding")
        self.cls = MLP(input_dim=hidden_dim, hidden_dim=30, output_dim=3, num_layers=1)
        
        # self.MLP = self.MLP.to(opt.device)

        self.Enc = MLP(input_dim=input_dim, hidden_dim=50, output_dim=hidden_dim, num_layers=3)
        self.Enc_rnn = MLP(input_dim=input_dim-1, hidden_dim=50, output_dim=hidden_dim, num_layers=3)
        self.rnn = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim // 2, 
                                 num_layers=num_layers, batch_first=batch_first, 
                                 bidirectional=True)

    def forward(self, inputs):
        # TODO: Implement LSTM forward pass
        # import pdb; pdb.set_trace()
        
        all_emb, x, de, pos = inputs[0], inputs[1], inputs[2], inputs[3]
        # batch_size, seq_size, seq_length = idx_in_x.shape
        # idx_x = torch.arange(batch_size).to(idx_in_x.device)
        # idx_x_new = idx_x.reshape(batch_size,1,1).repeat(1,seq_size,seq_length)
        # new_pos = torch.cat((idx_x_new, idx_in_x), 2)
        x = self.embed(x)
        x = torch.cat((x, de.unsqueeze(-1)), -1)
        batch_size, seq_size, seq_length, word_dim = x.shape
        x = x.reshape((batch_size * seq_size, seq_length, word_dim))
        x = self.Enc(x)

        # input = torch.randn(batch_size * seq_size, seq_length, x.size()[-1]).to(x.device)
        # output, _ = self.rnn(input)
        
        outx, _ = self.model(x)
        # import pdb; pdb.set_trace()
        all_emb = self.Enc_rnn(self.embed(all_emb))
        y_lstm, _  = self.rnn(all_emb)

        # import pdb; pdb.set_trace()
        posnew = torch.stack((torch.arange(batch_size), pos.cpu()), 0).numpy()
        outx = (outx[:,-1,:]).reshape(batch_size, seq_size, -1)
        # y = torch.cat((outx.sum(1), y_lstm[posnew]), 1)
        y = outx.sum(1) + y_lstm[posnew]
        final_output = self.cls(y)

        
        # return p[posnew]
        return final_output, self.cls(outx.reshape(-1, self.hidden_dim))  # p[:, -1, :]
