

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
        #ds2 = load_dataset("allenai/WildChat-1M")

        # Combine 'instruction', 'input', 'output'
        def combineText(example):
            example["input"] = example['instruction'] + ' ' + example['input'] + ': ' + example['output']
            return example
        updatedDataset = ds1.map(combineText)

        # Rename and and select text data
        updatedDataset = updatedDataset.rename_column('input', 'text')
        updatedDataset = updatedDataset.select_columns('text')

        # Add index
        self.data = updatedDataset.map(lambda example, idx: {'idx': f'{idx}'}, with_indices=True)

    def __len__(self):
        return Error
    
    def __getitem__(self, idx):

        # Get data
        data = self.data[idx]
        contextwindow = data[inx]['train']['text'][:1024]
        nextcharacter = data[inx]['train']['text'][1025]

        # output
        return contextwindow, nextcharacter
    
    def tokens2chars(self, tokens):
        return Error
    
    def chars2tokens(self, chars):
        return Error

ds = ChatDataset()
#print(ds.data['train']['text'][:5]) # prints the first 5 portions of text data


## Tokenize:
# Unique Characters in Dataset
text = ' '.join(ds.data['train']['text'])
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Create mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # Encoder (string to integer)
decode = lambda l: ''.join([itos[i] for i in l] ) # Decoder (integer to string)

# Encode dataset and store in torch.tensor
encodedData = torch.tensor(encode(text), dtype=torch.short)
print(encodedData.shape, encodedData.dtype)
print(encodedData[:1000])