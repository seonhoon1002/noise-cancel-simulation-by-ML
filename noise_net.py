import torch
import torch.nn as nn
import torch.nn.functional as F

n_hidden=5

class WaveRNN(nn.Module):
  def __init__(self):
    super(WaveRNN, self).__init__()

    self.rnn = nn.RNN(input_size=1, hidden_size=n_hidden, dropout=0.3,batch_first=True)
    self.fc1 = nn.Linear(n_hidden, 5)
    self.fc2 = nn.Linear(n_hidden, 1)

  def forward(self, hidden, X):
    # X = X.transpose(0, 1)
    outputs, hidden = self.rnn(X, hidden)
    # outputs = outputs[-1]  # 최종 예측 Hidden Layer
    outputs = F.relu(self.fc1(outputs)) # 최종 예측 최종 출력 층
    outputs = self.fc2(outputs)
    return outputs