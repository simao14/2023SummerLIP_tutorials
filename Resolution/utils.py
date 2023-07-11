import torch
import numpy as np

class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):

        """
        data: the dict returned by utils.load_classification_data
        """
        
        train_X = data
        train_y = labels
        
        self.X = torch.tensor(train_X, dtype=torch.float32)
        self.y = torch.tensor(train_y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def prepdata(data):
    
    TreeS=data["TreeS"]
    TreeB=data["TreeB"]
    signal=TreeS.arrays()
    background=TreeB.arrays()
    stages=["var1","var2","var3","var4"]
    nsignal=len(signal["var1"])
    nback=len(background["var1"])
    nevents=nsignal+nback
    x=np.zeros([nevents,len(stages)])
    y=np.zeros(nevents)
    y[:nsignal]=1
    for i,j in enumerate(stages):
        x[:nsignal,i]=signal[j]
        x[nsignal:,i]=background[j]
    
    return x,y