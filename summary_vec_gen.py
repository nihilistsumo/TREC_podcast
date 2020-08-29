import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

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

def run_model(qrels_secid_data, secids, secid_vecs, paraids, paraid_vecs, max_seq_len, emb_vec_size):
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
    X = torch.tensor(X).float()
    y = torch.tensor(y).float()

    m = SummaryEmbedGen(emb_vec_size, max_seq_len)
    opt = optim.Adam(m.parameters(), lr=0.001)
    mseloss = nn.MSELoss()
    for i in range(1000):
        opt.zero_grad()
        ypred = m(X)
        loss = mseloss(ypred, y)
        loss.backward()
        opt.step()
        if i % 100 == 0:
            print(loss)
    print(y)
    print(m(X))


def main():
    with open('/media/sumanta/Seagate Backup Plus Drive/TREC2020_podcast_track/treccar_helper_data/by1train_nodup_toplevel_secid.json', 'r') as qd:
        dat = json.load(qd)
    secids = list(np.load('/media/sumanta/Seagate Backup Plus Drive/TREC2020_podcast_track/treccar_helper_data/by1train_toplevel_sections.npy'))
    secid_vecs = np.load('/media/sumanta/Seagate Backup Plus Drive/TREC2020_podcast_track/treccar_helper_data/by1train_toplevel_section_vecs.npy')
    paraids = list(np.load('/media/sumanta/Seagate Backup Plus Drive/SentenceBERT_embeddings/sentbert_embeddings_by1train/bert-base-passage-wiki-sec-mean/paraids.npy'))
    paraid_vecs = np.load('/media/sumanta/Seagate Backup Plus Drive/SentenceBERT_embeddings/sentbert_embeddings_by1train/bert-base-passage-wiki-sec-mean/bert-base-wikipedia-sections-mean-tokens-passage-part1.npy')

    run_model(dat, secids, secid_vecs, paraids, paraid_vecs, 10, 768)

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