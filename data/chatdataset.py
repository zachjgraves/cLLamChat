

# Import libraries
import os
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

# Create a chat dataset class
class ChatDataset(Dataset):
    def __init__(self,):
        super(ChatDataset, self).__init__()

        # Load datasets
        ds1 = load_dataset("yahma/alpaca-cleaned")
        ds2 = load_dataset("allenai/WildChat-1M")

        # Combine 'instruction', 'input', 'output'
        def combineText(example):
            example["input"] = example['instruction'] + ' ' + example['input'] + ': ' + example['output']
            return example
        updatedDataset = ds1.map(combineText)

        # Rename and and select text data
        updatedDataset = updatedDataset.rename_column('input', 'text')
        updatedDataset = updatedDataset.select_columns('text')

        # Add index
        indexedDataset = updatedDataset.map(lambda example, idx: {'idx': f'{idx}'}, with_indices=True)
        
        return indexedDataset

    def __len__(self):
        return Error
    
    def __getitem__(self, idx):

        # Get data
        data = self.data[idx]

        # output
        return contextwindow, nextcharacter
    
    def tokens2chars(self, tokens):
        return Error
    
    def chars2tokens(self, chars):
        return Error