import torch
from torch import nn
from torchcrf import CRF

class NERLSTM_CRF(nn.Module):
    def __init__(self, opt, embedding_dim, hidden_dim, dropout, word2id, tag2id):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.word2id = word2id + 2
        self.word_num = len(word2id)
        self.tag2id = tag2id
        self.tag_num = len(tag2id)
        self.gpu = opt.gpu
        self.device = opt.device
        self.embeds = nn.Embedding(self.word_num, self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                           hidden_size=self.hidden_dim//2,
                           num_layers=1,
                           bidirectional=True,
                           batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_num)
        self.crf = CRF(self.tag_num)
        if self.gpu:
            self.to(self.device)
    
    def forwarf(self, x):
        """
        
        :param x: [batch_size, max_seq] 
        :return: 
        """
        x = x.transpose(0, 1)  # [max_seq, batch_size]

        # 由于LSTM batch_first为False
        x = self.embeds(x)
        out, hidden = self.lstm(x)
        out = self.dropout(out)
        out = self.hidden2tag(out)

        out = self.crf.decode(out)
        # find the most likely tag sequence
        return out

    def log_likelihood(self, x, tags):
        x = x.transpose(0,1)
        x = self.embeds(x)
        out, hidden = self.lstm(x)
        out = self.dropout(out)
        out = self.hidden2tag(out)

        return -self.crf(out, tags)
