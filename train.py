from config import opt
import torch
from torch import nn,optim
from loader import data_load, NERDataset
from torch.utils.data import DataLoader
from model import NERLSTM_CRF

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def train(model_name):
    model = models[model_name](opt, opt.embedding_dim, opt.hidden_dim, opt.dropout, word2id, tag2id)
    Loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    for epoch in range(opt.max_epoch):
        model.train()
        for index, batch in enumerate(train_dataloader):
            x = batch['x']
            y = batch['y']
            if opt.gpu:
                x = x.to(opt.device)
                y = y.to(opt.device)
            # CRF
            loss = model.log_likelihood(x,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_total = 0
        preds, labels = [], []
        for index, batch in enumerate(val_dataloader):
            model.eval()
            x = batch['x']
            y = batch['y']
            if opt.gpu:
                x = x.to(opt.device)
                y = y.to(opt.device)

            predict = model(x)
            loss = model.log_likelihood(x, y)
            loss_total += loss.item()

            #统计非0的（真实的）标签长度
            length = []
            for i in y.cpu():
                temp = []
                for j in i:
                    if j.item() > 0:
                        temp.append(j.item)
                length.append(temp)

            for index, pred in enumerate(predict):
                preds += pred[: len(length[index])]
            for index, iter_y in enumerate(y.tolist()):
                labels += iter_y[: len(length[index])]

        aver_loss = loss_total / (len(val_dataloader)* 64)
        print("len(val_dataloader):{}, loss:{}".format(len(val_dataloader), aver_loss))
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        f1 = f1_score(labels, preds, average='macro')
        report = classification_report(labels, preds, )
        print(report)
        torch.save(model.state_dict(), 'params.pkl')


if __name__ == '__main__':
    word2id, id2word, tag2id, id2tag, x_train, y_train, x_test, y_test, x_val, y_val = data_load(opt.pickle_path)

    train_dataset = NERDataset(x_train,y_train)
    val_dataset = NERDataset(x_val, y_val)
    test_dataset = NERDataset(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True,num_workers=opt.num_workers)

    # models = {'NERLSTM': NERLSTM,
    #           'NERLSTM_CRF': NERLSTM_CRF}
    models = {'NERLSTM_CRF': NERLSTM_CRF}

    train(opt.model)
