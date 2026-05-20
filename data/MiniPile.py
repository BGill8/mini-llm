from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from data.utils import TikTokenizer

class MiniPileDataset(IterableDataset):

    def __init__(self, tokenizer, split = 'train', chunk_size=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.data = load_dataset("JeanKaddour/minipile", cache_dir="data/cache", split=split, streaming=True)
    

    def __iter__(self):
        buffer = []
        
        for x in self.data:
            tokens = self.tokenizer.encode(x['text'])+[self.tokenizer.pad_id]
            buffer.extend(tokens)

            while len(buffer) >= self.chunk_size:
                chunk = buffer[:self.chunk_size]
                buffer = buffer[self.chunk_size:]
                yield torch.LongTensor(chunk)
    

def getMiniPileDataloder(tokenizer, batch_size=16, max_length=1024):
    train = MiniPileDataset(tokenizer, "train", max_length)
    loader = DataLoader(dataset=train, batch_size=batch_size)

    return loader

if __name__ == "__main__":
    loader = getMiniPileDataloder(TikTokenizer())
    
