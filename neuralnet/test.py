import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from noise_net import WaveRNN
from dataset import WaveDataset
import torch

batch_size= 1
n_hidden=5

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

test_wave_dataset= WaveDataset("wave_data_val.pickle")
test_loader=DataLoader(test_wave_dataset, batch_size=batch_size, shuffle=False,num_workers=2)

model= WaveRNN().to(device)
model.load_state_dict(torch.load("weight.pth"))
model.eval()
error=0

for i_batch, sample_batched in enumerate(test_loader):
    with torch.no_grad():
        noisy_sig= torch.unsqueeze(sample_batched['noise'].float(),2).to(device)
        pure_sig= torch.unsqueeze(sample_batched['pure'].float(),2).to(device)
        h0=torch.zeros(1,pure_sig.size(0),n_hidden,requires_grad=False).to(device)

        # print(noisy_sig.size())
        
        outputs= model(h0, noisy_sig)
        print(outputs.size())
        
        error+= torch.mean(torch.abs(outputs-pure_sig))

    print("error:",error.item()/len(test_loader))
    
