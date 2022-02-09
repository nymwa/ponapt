import torch

class Dataset(torch.utils.data.Dataset):

    def __init__(self, sents):
        self.sents = sents
        self.lengths = torch.tensor([len(sent) for sent in sents])

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        x = self.sents[index]
        x = x.tolist()
        return x

