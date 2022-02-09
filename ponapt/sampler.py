import torch
import random as rd

class Sampler(torch.utils.data.Sampler):

    def __init__(
            self,
            dataset,
            max_tokens):

        self.dataset = dataset
        self.max_tokens = max_tokens
        self.batches = None

    def generate_batches(self):
        indices = self.get_indices()
        batches = []
        batch = []
        acc = 0
        max_len = 0
        for index in indices:
            acc += 1
            this_len = self.dataset.lengths[index]
            max_len = max(max_len, this_len)
            if (acc * max_len) > self.max_tokens:
                batches.append(batch)
                batch = [index]
                acc = 1
                max_len = this_len
            else:
                batch.append(index)
        if batch:
            batches.append(batch)
        rd.shuffle(batches)
        return batches

    def init_batches(self):
        if self.batches is None:
            self.batches = self.generate_batches()

    def __len__(self):
        self.init_batches()
        return len(self.batches)

    def __iter__(self):
        self.init_batches()
        for batch in self.batches:
            yield batch
        self.terminate_batches()


class FixedSampler(Sampler):

    def get_indices(self):
        if not hasattr(self, 'indices'):
            indices = torch.arange(len(self.dataset))
            indices = indices[self.dataset.lengths[indices].argsort(descending = True)]
            self.indices = indices
        return self.indices

    def terminate_batches(self):
        pass


class RandomSampler(Sampler):

    def get_indices(self):
        indices = torch.randperm(len(self.dataset))
        indices = indices[self.dataset.lengths[indices].argsort(descending = True)]
        return indices

    def terminate_batches(self):
        self.batches = None

