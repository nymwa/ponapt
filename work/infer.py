from argparse import ArgumentParser
import torch
from torch.nn.utils.rnn import pad_sequence as pad
from ponapt.vocab import load_vocab
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
    parser.add_argument('--iters', type = int, default = 10)
    parser.add_argument('--prefix', default = None)
    parser.add_argument('--terminate-quot', action = 'store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    vocab = load_vocab(args.vocab)
    preproc = LMPreproc()
    postproc = LMPostproc()

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

    sampler = SentenceSampler(vocab, model)

    if args.prefix is not None:
        prefix = preproc(args.prefix)
        prefix = [vocab(token) for token in prefix.split()]
    else:
        prefix = None

    if args.terminate_quot:
        terminal = {vocab('"')}
    else:
        terminal = None

    for i in range(args.iters):
        if prefix is None:
            sent = None
        else:
            sent = prefix[:]
        sent = sampler(sent = sent, terminal = terminal)
        sent = ' '.join([vocab[x] for x in sent])
        sent = postproc(sent)
        print(sent)


if __name__ == '__main__':
    main()

