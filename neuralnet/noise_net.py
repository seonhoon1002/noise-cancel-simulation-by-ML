import torch
import torch.nn as nn
import torch.nn.functional as F

n_hidden=20

class WaveRNN(nn.Module):
  def __init__(self):
    super(WaveRNN, self).__init__()

    self.rnn = nn.RNN(input_size=1, hidden_size=n_hidden, dropout=0.3,batch_first=True)
    self.fc1 = nn.Linear(n_hidden, n_hidden)
    self.fc2 = nn.Linear(n_hidden, 1)

  def forward(self, hidden, X):
    # X = X.transpose(0, 1)
    outputs, hidden = self.rnn(X, hidden)
    # outputs = outputs[-1]  # 최종 예측 Hidden Layer
    outputs = F.relu(self.fc1(outputs)) # 최종 예측 최종 출력 층
    outputs = self.fc2(outputs)
    return outputs

class WaveFC(nn.Module):
  def __init__(self):
    super(WaveFC, self).__init__()

    # self.rnn = nn.RNN(input_size=1, hidden_size=n_hidden, dropout=0.3,batch_first=True)
    self.fc1 = nn.Linear(41, n_hidden)
    self.fc2 = nn.Linear(n_hidden,int(n_hidden/2))
    self.fc3 = nn.Linear(int(n_hidden/2),int(n_hidden/4))
    self.fc4 = nn.Linear(int(n_hidden/4),int(n_hidden/2))
    self.fc5 = nn.Linear(int(n_hidden/2), 41)

  def forward(self, x):
    # X = X.transpose(0, 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    outputs = self.fc5(x)
    return outputs