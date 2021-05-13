from torch.utils.data import Dataset
import pickle

class WaveDataset(Dataset):
    def __init__(self, pickle_file):
        with open('wave_data.pickle', 'rb') as f:
            """
            self.data['pure']: pure signal
            self.data['noise']: noise signal
            """
            self.data= pickle.load(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]
        
        
        