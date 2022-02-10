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
    parser.add_argument('--checkpoint', default = 'bert.pt')
    parser.add_argument('--vocab', default = 'vocab.txt')
    parser.add_argument('--hidden-dim', type = int, default = 512)
    parser.add_argument('--nhead', type = int, default = 8)
    parser.add_argument('--feedforward-dim', type = int, default = 2048)
    parser.add_argument('--dropout', type = float, default = 0.3)
    parser.add_argument('--attention-dropout', type = float, default = 0.2)
    parser.add_argument('--activation-dropout', type = float, default = 0.2)
    parser.add_argument('--num-layers', type = int, default = 24)
    parser.add_argument('--max-len', type = int, default = 256)
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

    for i in range(30):
        sent = sampler()
        sent = ' '.join([vocab[x] for x in sent])
        sent = postproc(sent)
        print(sent)


if __name__ == '__main__':
    main()

