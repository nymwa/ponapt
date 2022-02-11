import random as rd
import torch
from torch.nn.utils.rnn import pad_sequence as pad
from ponapt.batch import Batch

def generate_square_subsequent_mask(trg_size, src_size=None):

    if src_size is None:
        src_size = trg_size

    mask = (torch.triu(torch.ones(src_size, trg_size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


class Collator:

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, batch):
        inputs  = [torch.tensor([self.vocab.bos] + sent) for sent in batch]
        outputs = [torch.tensor(sent + [self.vocab.eos]) for sent in batch]
        lengths = [len(sent) for sent in batch]

        inputs = pad(inputs, padding_value = self.vocab.pad)
        outputs = pad(outputs, padding_value = self.vocab.pad)
        padding = (inputs == self.vocab.pad).T
        mask = generate_square_subsequent_mask(outputs.shape[0])
        return Batch(inputs, outputs, lengths, None, padding, mask)


class TrainingCollator(Collator):

    def __init__(
            self,
            vocab,
            shift_prob = 0.80,
            max_shift = 64):

        super().__init__(vocab)
        self.shift_prob = shift_prob
        self.max_shift = max_shift

    def __call__(self, batch):
        inputs  = [torch.tensor([self.vocab.bos] + sent) for sent in batch]
        outputs = [torch.tensor(sent + [self.vocab.eos]) for sent in batch]
        lengths = [len(sent) for sent in batch]

        inputs = pad(inputs, padding_value = self.vocab.pad)
        outputs = pad(outputs, padding_value = self.vocab.pad)

        if rd.random() < self.shift_prob:
            dist = rd.randrange(self.max_shift)
            position = torch.arange(inputs.size(0)) + dist
            position = position.unsqueeze(-1)
        else:
            dist = 0
            position = None

        padding = (inputs == self.vocab.pad).T
        mask = generate_square_subsequent_mask(outputs.shape[0])
        misc = {'dist': dist, 'b0': inputs.shape[0], 'b1': inputs.shape[1]}

        return Batch(inputs, outputs, lengths, position, padding, mask, misc)

