import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from noise_net import WaveRNN,WaveFC
from dataset import WaveDataset
import torch

batch_size= 128
n_hidden=20

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

test_wave_dataset= WaveDataset("wave_data_val.pickle")
test_loader=DataLoader(test_wave_dataset, batch_size=batch_size, shuffle=False,num_workers=2)

# model= WaveRNN().to(device)
model= WaveFC().to(device)
model.load_state_dict(torch.load("weight.pth"))
model.eval()
error=0
ori_error=0

# for i_batch, sample_batched in enumerate(test_loader):
#     with torch.no_grad():
#         noisy_sig= torch.unsqueeze(sample_batched['noise'].float(),2).to(device)
#         pure_sig= torch.unsqueeze(sample_batched['pure'].float(),2).to(device)
#         h0=torch.zeros(1,pure_sig.size(0),n_hidden,requires_grad=False).to(device)

#         # print(noisy_sig.size())
        
#         outputs= model(h0, noisy_sig)
#         error+= torch.mean(torch.abs(outputs-pure_sig))
#         ori_error+= torch.mean(torch.abs(noisy_sig-pure_sig))
        
#         if i_batch %1000==999:
#             print("batch",i_batch)
#             print("error:",error.item())
#             print("ori_error:",ori_error.item())
for i_batch, sample_batched in enumerate(test_loader):
    with torch.no_grad():
        noisy_sig= sample_batched['noise'].float().to(device)
        pure_sig= sample_batched['pure'].float().to(device)
        outputs= model(noisy_sig)
        error+= torch.mean(torch.abs(outputs-pure_sig))
        ori_error+= torch.mean(torch.abs(noisy_sig-pure_sig))
        
        if i_batch %1000==999:
            print("batch",i_batch)
            print("error:",error.item())
            print("ori_error:",ori_error.item())
print("error:",error.item()/len(test_loader))
print("ori_error:",ori_error.item()/len(test_loader))

    
    
