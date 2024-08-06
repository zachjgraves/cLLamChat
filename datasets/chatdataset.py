

# Import libraries
import os
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

# Create a chat dataset class
class ChatDataset(Dataset):
    def __init__(self, context_window=1024, vocab=None, stop_token=None, unknown_token=None):
        super(ChatDataset, self).__init__()

        # Configure vocab
        if stop_token is None:
            stop_token = "🛑"
        if unknown_token is None:
            unknown_token = "❓"
        if vocab is None:
            # This is a list of all characters the model can predict, including the stop token, and the unknown token
            vocab = (
                list(" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
                + [stop_token]
                + [unknown_token]
            )

        # Set attributes
        self.context_window = context_window
        self.vocab = vocab
        self.stop_token = stop_token
        self.unknown_token = unknown_token

        # Load datasets
        ds1 = load_dataset("yahma/alpaca-cleaned")
        self.ds1 = ds1
        #ds2 = load_dataset("allenai/WildChat-1M")
        #self.ds2 = ds2

        # Get number of characters in each dataset
        n_chars = 0
        n_chars_per_example = []
        for ds in [ds1]:
            for example in ds['train']['output']:
                n_chars += len(example) + 1                   # Add 1 for stop token
                n_chars_per_example.append(len(example) + 1)  # Add 1 for stop token
        self.n_chars = n_chars
        self.n_chars_per_example = n_chars_per_example


    def __len__(self):
        return len(self.n_chars)
    
    def __getitem__(self, idx, return_text=False):

        # Get attributes
        context_window = self.context_window
        stop_token = self.stop_token
        n_chars = self.n_chars
        n_chars_per_example = self.n_chars_per_example

        # Get example index
        example_idx = 0
        while idx >= n_chars_per_example[example_idx]:
            idx -= n_chars_per_example[example_idx]
            example_idx += 1

        # Get example
        example = self.ds1['train'][example_idx]
        example_text = (
            self.text2tokens(example['output'])
            + [self.vocab.index(self.stop_token)]
        )
        
        # Get text
        text = example_text[:idx+1]

        # Add in input text
        text = (
            self.text2tokens(example['instruction'])
            + self.text2tokens(example['input'])
            + [self.vocab.index(self.stop_token)]
            + text
        )

        # Get context
        context = text[-context_window:]

        # Pad context window if necessary
        while len(context) < context_window:
            context = [self.vocab.index(self.unknown_token)] + context

        # If return text, convert
        if return_text:
            context = self.tokens2text(context)

        # Return context
        return context
    
    def text2tokens(self, text):
        """
        This function converts a string of text into a list of tokens.
        """

        # Get attributes
        vocab = self.vocab
        stop_token = self.stop_token
        unknown_token = self.unknown_token

        # Tokenize text
        tokens = []
        for char in text:
            if char in vocab:
                tokens.append(vocab.index(char))
            else:
                tokens.append(vocab.index(unknown_token))

        # Return tokens
        return tokens

    def tokens2text(self, tokens):
        """
        This function converts a list of tokens into a string of text.
        """

        # Get attributes
        vocab = self.vocab

        # Convert tokens to text
        text = ""
        for token in tokens:
            text += vocab[token]

        # Return text
        return text
    
# Test
if __name__ == '__main__':

    # Load dataset
    ds = ChatDataset(context_window=10)
    
    # Get first few examples
    for i in range(750, 760):
        print(ds.tokens2text(ds[i]))

    # Done
    print("Done")

    # ds = ChatDataset()
    # #print(ds.data['train']['text'][:5]) # prints the first 5 portions of text data


    # ## Tokenize:
    # # Unique Characters in Dataset
    # text = ' '.join(ds.data['train']['text'])
    # chars = sorted(list(set(text)))
    # vocab_size = len(chars)
    # print(''.join(chars))
    # print(vocab_size)

    # # Create mapping from characters to integers
    # stoi = { ch:i for i,ch in enumerate(chars) }
    # itos = { i:ch for i,ch in enumerate(chars) }
    # encode = lambda s: [stoi[c] for c in s] # Encoder (string to integer)
    # decode = lambda l: ''.join([itos[i] for i in l] ) # Decoder (integer to string)

    # # Encode dataset and store in torch.tensor
    # encodedData = torch.tensor(encode(text), dtype=torch.short)
    # print(encodedData.shape, encodedData.dtype)
    # print(encodedData[:1000])