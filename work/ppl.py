from argparse import ArgumentParser
from seriejo import Seriejo
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence as pad
from ponapt.vocab import load_vocab
from ponapt.dataset import Dataset
from ponapt.sampler import FixedSampler
from ponapt.collator import Collator
from ponapt.preproc import LMPreproc
from ponapt.postproc import LMPostproc
from ponapt.log import init_logging
from ponapt.lm import LM
from ponapt.generation.sampler import SentenceSampler
from tabulate import tabulate

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', default = 'lm.pt')
    parser.add_argument('--vocab', default = 'vocab.txt')
    parser.add_argument('--hidden-dim', type = int, default = 512)
    parser.add_argument('--nhead', type = int, default = 8)
    parser.add_argument('--feedforward-dim', type = int, default = 2048)
    parser.add_argument('--dropout', type = float, default = 0.3)
    parser.add_argument('--attention-dropout', type = float, default = 0.2)
    parser.add_argument('--activation-dropout', type = float, default = 0.2)
    parser.add_argument('--num-layers', type = int, default = 24)
    parser.add_argument('--max-len', type = int, default = 256)
    parser.add_argument('--max-tokens', type = int, default = 4000)
    return parser.parse_args()


def load_dataset(name):
    data = Seriejo('data/{}'.format(name))
    dataset = Dataset(data)
    return dataset


def load_loaders(vocab, args):
    valid_dataset = load_dataset('valid')
    test_dataset = load_dataset('test')
    valid_sampler = FixedSampler(valid_dataset, args.max_tokens)
    test_sampler = FixedSampler(test_dataset, args.max_tokens)
    collator = Collator(vocab)
    valid_loader = DataLoader(
            valid_dataset,
            batch_sampler = valid_sampler,
            collate_fn = collator)
    test_loader = DataLoader(
            test_dataset,
            batch_sampler = test_sampler,
            collate_fn = collator)
    return valid_loader, test_loader


def calc_ppl(criterion, model, loader):
    acc = 0.0
    cnt = 0
    for step, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        with torch.no_grad():
            pred = model(batch)
            pred = pred.view(-1, pred.size(-1))
            loss = criterion(pred, batch.outputs.view(-1))
        acc += len(batch) * loss.item()
        cnt += len(batch)
    return acc / cnt


def main():
    args = parse_args()
    vocab = load_vocab(args.vocab)
    valid_loader, test_loader = load_loaders(vocab, args)

    model = LM(
            len(vocab),
            args.hidden_dim,
            args.nhead,
            args.feedforward_dim,
            args.dropout,
            args.attention_dropout,
            args.activation_dropout,
            args.num_layers,
            padding_idx = vocab.pad,
            max_len = args.max_len)
    model.load_state_dict(torch.load(args.checkpoint, map_location = 'cpu'))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index = vocab.pad)

    valid_ppl = calc_ppl(criterion, model, valid_loader)
    test_ppl = calc_ppl(criterion, model, test_loader)
    print('valid: {}'.format(valid_ppl))
    print('test: {}'.format(test_ppl))


if __name__ == '__main__':
    main()

