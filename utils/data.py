from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch

def yield_tokens(data_iter, tokenizer):
    for label, text in data_iter:
        yield tokenizer(text)

def build_dataset(batch_size, max_len):
    tokenizer = get_tokenizer("basic_english")
    train_iter, test_iter = AG_NEWS()

    vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    def encode(text):
        tokens = tokenizer(text)
        ids = vocab(tokens)[:max_len]
        return torch.tensor(ids + [0] * (max_len - len(ids))), len(ids)

    def collate_batch(batch):
        texts, labels = [], []
        for label, text in batch:
            encoded, _ = encode(text)
            texts.append(encoded)
            labels.append(label - 1)
        return torch.stack(texts), torch.tensor(labels)

    train_iter, test_iter = AG_NEWS()
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(list(test_iter), batch_size=batch_size, collate_fn=collate_batch)
    
    return train_dataloader, test_dataloader, len(vocab)
