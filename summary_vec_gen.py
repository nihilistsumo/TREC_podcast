import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy.random import seed
seed(42)
torch.manual_seed(42)
import random
random.seed(42)
import json
import argparse

class SummaryEmbedGen(nn.Module):

    def __init__(self, emb_size, max_seq_len):
        super(SummaryEmbedGen, self).__init__()
        self.emb_size = emb_size
        self.max_seq_len = max_seq_len
        self.lstm = nn.LSTM(emb_size, emb_size)
        self.lin = nn.Linear(emb_size, emb_size)
        self.lout = nn.Linear(emb_size, emb_size)

    def forward(self, X):
        vec_in = self.lin(X)
        out, _ = self.lstm(vec_in)
        last_lstm_output = out[:, -1, :]
        vec_out = self.lout(last_lstm_output)
        return vec_out

def run_model(qrels_secid_data, secids, secid_vecs, paraids, paraid_vecs, max_seq_len, emb_vec_size, iter, lr, outpath):
    X = []
    y = []
    for s in qrels_secid_data.keys():
        paralist = qrels_secid_data[s]['paras']
        if len(paralist) < 3:
            continue
        dat = []
        for p in paralist:
            paravec = paraid_vecs[paraids.index(p)]
            dat.append(paravec)
        if len(dat) > max_seq_len:
            dat = dat[:max_seq_len]
        elif len(dat) < max_seq_len:
            dat += list(np.zeros((max_seq_len - len(dat), emb_vec_size)))
        X.append(dat)
        y.append(secid_vecs[secids.index(s)])
    sample_size = len(X)
    train_indices = torch.tensor(random.sample(list(range(sample_size)), sample_size*4//5))
    val_indices = torch.tensor([i for i in range(sample_size) if i not in train_indices])

    X = torch.tensor(X).float().cuda()
    y = torch.tensor(y).float().cuda()
    X_val = torch.index_select(X, 0, val_indices)
    y_val = torch.index_select(y, 0, val_indices)
    X = torch.index_select(X, 0, train_indices)
    y = torch.index_select(y, 0, train_indices)

    m = SummaryEmbedGen(emb_vec_size, max_seq_len).cuda()
    opt = optim.Adam(m.parameters(), lr=lr)
    mseloss = nn.MSELoss()
    for i in range(iter):
        m.train()
        opt.zero_grad()
        ypred = m(X)
        loss = mseloss(ypred, y)
        loss.backward()
        opt.step()
        if i % 100 == 0:
            m.eval()
            ypred_val = m(X_val)
            val_loss = mseloss(ypred_val, y_val)
            print('Train loss: %.5f, Val loss: %.5f' % (loss.item(), val_loss.item()))
    print('Final loss: %.5f' % loss.item())
    print(y.detach().cpu().numpy())
    print(m(X).detach().cpu().numpy())
    torch.save(m.state_dict(), outpath)


def main():
    parser = argparse.ArgumentParser(description='Train summary embed vec generator')
    parser.add_argument('-sd', '--input_data', help='Path to treccar qrels data file with secid',
                        default='/media/sumanta/Seagate Backup Plus Drive/TREC2020_podcast_track/treccar_helper_data/'
                                'by1train_nodup_toplevel_secid.json')
    parser.add_argument('-si', '--secid', help='Path to secid list',
                        default='/media/sumanta/Seagate Backup Plus Drive/TREC2020_podcast_track/treccar_helper_data/'
                                'by1train_toplevel_sections.npy')
    parser.add_argument('-sv', '--sec_vecs', help='Path to section vecs',
                        default='/media/sumanta/Seagate Backup Plus Drive/TREC2020_podcast_track/treccar_helper_data/'
                                'by1train_toplevel_section_vecs.npy')
    parser.add_argument('-pi', '--paraid', help='Path to paraid list',
                        default='/media/sumanta/Seagate Backup Plus Drive/SentenceBERT_embeddings/'
                                'sentbert_embeddings_by1train/bert-base-passage-wiki-sec-mean/paraids.npy')
    parser.add_argument('-pv', '--paravecs', help='Path to para vecs',
                        default='/media/sumanta/Seagate Backup Plus Drive/SentenceBERT_embeddings/'
                                'sentbert_embeddings_by1train/bert-base-passage-wiki-sec-mean/'
                                'bert-base-wikipedia-sections-mean-tokens-passage-part1.npy')
    parser.add_argument('-mo', '--model_out', help='Path to model output',
                        default='/media/sumanta/Seagate Backup Plus Drive/TREC2020_podcast_track/treccar_helper_data/'
                                'by1train_nodup_toplevel.model')
    parser.add_argument('-seq', '--max_seq_len', type=int, default=10, help='Maximum sequence length')
    parser.add_argument('-emb', '--emb_dim', type=int, default=768, help='Embedding vector length')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-it', '--num_iter', type=int, default=1000, help='Num of iterations')
    args = parser.parse_args()
    with open(args.input_data, 'r') as qd:
        dat = json.load(qd)
    secids = list(np.load(args.secid))
    secid_vecs = np.load(args.sec_vecs)
    paraids = list(np.load(args.paraid))
    paraid_vecs = np.load(args.paravecs)

    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device('cuda:0'))
    else:
        torch.cuda.set_device(torch.device('cpu'))

    run_model(dat, secids, secid_vecs, paraids, paraid_vecs, args.max_seq_len, args.emb_dim, args.num_iter,
              args.learning_rate, args.model_out)

'''
def main():
    sample_size = 25
    emb_size = 8
    max_seq = 4
    X = np.random.randn(sample_size, max_seq, emb_size)
    y = np.random.randn(sample_size, emb_size)
    X = torch.tensor(X).float()
    y = torch.tensor(y).float()
    m = SummaryEmbedGen(emb_size, max_seq)
    opt = optim.Adam(m.parameters(), lr=0.001)
    mseloss = nn.MSELoss()
    for i in range(10000):
        opt.zero_grad()
        ypred = m(X)
        loss = mseloss(ypred, y)
        loss.backward()
        opt.step()
        if i % 100 == 0:
            print(loss)
    print(y)
    print(m(X))
'''

if __name__ == '__main__':
    main()