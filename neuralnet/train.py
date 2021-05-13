import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from noise_net import WaveRNN
from dataset import WaveDataset
import torch

batch_size= 128
n_hidden=5

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_wave_dataset= WaveDataset("wave_data.pickle")
train_loader= DataLoader(train_wave_dataset, batch_size=batch_size, shuffle=True,num_workers=2)

test_wave_dataset= WaveDataset("wave_data.pickle")
test_loader=DataLoader(test_wave_dataset, batch_size=batch_size, shuffle=True,num_workers=2)

model= WaveRNN().to(device)
criterion= nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


total_loss=0
error=0
for epoch in range(10):
    for i_batch, sample_batched in enumerate(train_loader):
        # print(sample_batched['noise'].size())
        noisy_sig= torch.unsqueeze(sample_batched['noise'].float(),2).to(device)
        pure_sig= torch.unsqueeze(sample_batched['pure'].float(),2).to(device)
        h0=torch.zeros(1,pure_sig.size(0),n_hidden,requires_grad=True).to(device)
        
        outputs= model(h0, noisy_sig)
        loss = criterion(outputs, pure_sig)
        total_loss+= loss.item()
        if i_batch %100==99:
            print("loss",loss.item(),"error",torch.mean(torch.abs(outputs-pure_sig)).item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("final",total_loss/len(train_loader))
    total_loss=0

    error=0
    for i_batch, sample_batched in enumerate(test_loader):
        with torch.no_grad():
            noisy_sig= torch.unsqueeze(sample_batched['noise'].float(),2).to(device)
            pure_sig= torch.unsqueeze(sample_batched['pure'].float(),2).to(device)
            h0=torch.zeros(1,pure_sig.size(0),n_hidden,requires_grad=True).to(device)
            outputs= model(h0, noisy_sig)
            error+= torch.mean(torch.abs(outputs-pure_sig))
    
    print("error:",error.item()/len(train_loader))
    
torch.save(model.state_dict(), "weight.pth")
