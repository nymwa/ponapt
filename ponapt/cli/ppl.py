from argparse import ArgumentParser
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence as pad
from ponapt.vocab import load_vocab
from ponapt.dataset import Dataset
from ponapt.batch import Batch
from ponapt.collator import generate_square_subsequent_mask
from ponapt.lm import LM
from seriejo import Seriejo

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', default = 'lm.pt')
    parser.add_argument('--vocab', default = 'vocab.txt')
    parser.add_argument('--dataset', default = 'data/valid')
    parser.add_argument('--hidden-dim', type = int, default = 512)
    parser.add_argument('--nhead', type = int, default = 8)
    parser.add_argument('--feedforward-dim', type = int, default = 2048)
    parser.add_argument('--num-layers', type = int, default = 64)
    parser.add_argument('--max-len', type = int, default = 256)
    return parser.parse_args()


def load_dataset(name):
    data = Seriejo(name)
    dataset = Dataset(data)
    return dataset


def calc_probs(model, vocab, sent):
    inputs = torch.tensor([[vocab.bos] + sent]).T
    mask = generate_square_subsequent_mask(inputs.shape[0])
    batch = Batch(inputs, mask = mask)
    if torch.cuda.is_available():
        batch.cuda()

    with torch.no_grad():
        pred = model(batch)
    probs = torch.softmax(pred, dim = -1)
    return probs


def make_prob_list(model, vocab, dataset):
    for sent in dataset:
        probs = calc_probs(model, vocab, sent)
        for index, prob in zip(sent + [vocab.eos], probs):
            yield prob[0][index].item()


def main():
    args = parse_args()
    vocab = load_vocab(args.vocab)
    dataset = load_dataset(args.dataset)

    model = LM(
            len(vocab),
            args.hidden_dim,
            args.nhead,
            args.feedforward_dim,
            0, 0, 0, 0,
            args.num_layers,
            padding_idx = vocab.pad,
            max_len = args.max_len)
    model.load_state_dict(torch.load(args.checkpoint, map_location = 'cpu'))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    probs = [prob for prob in make_prob_list(model, vocab, dataset)]
    probs = [-np.log2(prob) for prob in probs]
    print(2 ** np.mean(probs))

