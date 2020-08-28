import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

if __name__ == '__main__':
    main()