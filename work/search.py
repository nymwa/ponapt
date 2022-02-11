from argparse import ArgumentParser
import torch
from torch.nn.utils.rnn import pad_sequence as pad
from ponapt.vocab import load_vocab
from ponapt.preproc import LMPreproc
from ponapt.postproc import LMPostproc
from ponapt.log import init_logging
from ponapt.lm import LM
from ponapt.generation.search import NoisingBeamSearch
from tabulate import tabulate

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', default = 'lm.pt')
    parser.add_argument('--vocab', default = 'vocab.txt')
    parser.add_argument('--hidden-dim', type = int, default = 512)
    parser.add_argument('--nhead', type = int, default = 8)
    parser.add_argument('--feedforward-dim', type = int, default = 2048)
    parser.add_argument('--num-layers', type = int, default = 24)
    parser.add_argument('--max-len', type = int, default = 256)
    parser.add_argument('--beta', type = float, default = 1.0)
    parser.add_argument('--width', type = int, default = 5)
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
            0, 0, 0, 0,
            args.num_layers,
            padding_idx = vocab.pad,
            max_len = args.max_len)
    model.load_state_dict(torch.load(args.checkpoint, map_location = 'cpu'))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    search = NoisingBeamSearch(model, vocab, args.width, beta = args.beta)
    t = 0
    beam = None
    while True:
        beam = search.step(beam)
        if beam.terminated():
            break
        t += 1
        print('time: {}'.format(t))
        for sent in beam.sents:
            print(sent.as_str(), sent.score())
        print('---')

    print('result')
    for sent in beam.store:
        print(sent.as_str(), sent.score())


if __name__ == '__main__':
    main()

