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

    return "ERROR: Not implemented"  #this should return two arrays, one with the variables values for all of the events and one that represents the true labels of each event (0 for back 1 for signal)