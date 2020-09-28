import torch

class DefaultConfig(object):
    model = 'NERLSTM_CRF'
    pickle_path = './renmindata.pkl'

    batch_size = 32
    num_workers = 4

    embedding_dim = 100
    hidden_dim =200
    dropout = 0.8
    lr = 0.001

    device = torch.device('cuda:3')
    gpu = True if torch.cuda.is_available() else False

    max_epoch = 10


opt = DefaultConfig()