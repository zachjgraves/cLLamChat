

# Import libraries
import os
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

# Create a chat datset class
class ChatDataset(Dataset):
    def __init__(self, context_window=1024):
        super(ChatDataset, self).__init__()

        # Set attributes
        self.context_window = context_window

        # Load datasets
        ds1 = load_dataset("yahma/alpaca-cleaned")
        ds2 = load_dataset("allenai/WildChat-1M")

    def __len__(self):
        return Error
    
    def __getitem__(self, idx):

        # Get data
        data = self.data[idx]

        # output
        return contextwindow
    
    def tokens2chars(self, tokens):
        return Error
    
    def chars2tokens(self, chars):
        return Error