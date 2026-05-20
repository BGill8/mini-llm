from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from data.utils import TikTokenizer

class SmolTalkDataset(Dataset):

    def __init__(self, tokenizer, split='train', max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_dataset("HuggingFaceTB/smol-smoltalk", cache_dir="data/cache", split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]

        tokens = []
        mask = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            turn_tokens = self.tokenizer.encode(f"<|{role}|>\n{content}") + [self.tokenizer.pad_id]
            tokens.extend(turn_tokens)
            is_assistant = 1 if role == "assistant" else 0
            mask.extend([is_assistant] * len(turn_tokens))

        # Truncate
        tokens = tokens[:self.max_length]
        mask = mask[:self.max_length]

        # Pad
        pad_len = self.max_length - len(tokens)
        tokens = tokens + [self.tokenizer.pad_id] * pad_len
        mask = mask + [0] * pad_len

        return torch.LongTensor(tokens), torch.LongTensor(mask)


def getSmolTalkDataloader(tokenizer, batch_size=16, max_length=1024):
    train = SmolTalkDataset(tokenizer, "train", max_length)
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)

    test = SmolTalkDataset(tokenizer, "test", max_length)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

    return train_loader,test_loader


if __name__ == "__main__":
    _,_ = getSmolTalkDataloader(TikTokenizer())
    