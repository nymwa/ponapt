from argparse import ArgumentParser
from collections import Counter
from seriejo import SeriejoWriter
from pathlib import Path
from ponapt.vocab import Vocab
from ponapt.preproc import LMPreproc
from ponapt.postproc import LMPostproc

from ponapt.log import init_logging
from logging import getLogger
init_logging()
logger = getLogger(__name__)

def sents_to_data(vocab, sents):

    def sent_to_data(sent):
        sent = sent.split()
        sent = [vocab(token) for token in sent]
        return sent

    return [sent_to_data(sent) for sent in sents]


def make_raw(base, name, sents):
    Path(base).mkdir(parents = True, exist_ok = True)

    with open('{}/{}.txt'.format(base, name), 'w') as f:
        for x in sents:
            print(x, file = f)


def make_seriejo(base, name, data):
    Path(base).mkdir(parents = True, exist_ok = True)

    with SeriejoWriter('{}/{}'.format(base, name)) as f:
        for x in data:
            f.write(x)

    logger.info('Write Seriejo ({}/{}): {}'.format(base, name, len(data)))


def get_train_sents(train_path, max_len):
    preproc = LMPreproc()

    with open(train_path) as f:
        sents = [preproc(sent) for sent in f]
    logger.info('loaded train: {}'.format(len(sents)))

    sents = [
        sent
        for sent
        in sents
        if 1 <= len(sent.split()) <= max_len]
    logger.info('filtered train: {}'.format(len(sents)))
    return sents


def get_valid_sents(valid_path):
    preproc = LMPreproc()

    with open(valid_path) as f:
        sents = [preproc(sent) for sent in f]
    logger.info('loaded valid: {}'.format(len(sents)))
    return sents


def make_tokens(train_sents):
    freq = Counter([
        word
        for sent
        in train_sents
        for word
        in sent.split()
        ]).most_common()
    tokens = [w for w, f in freq if w != '<unk>']
    tokens = ['<pad>', '<bos>', '<eos>', '<unk>'] + tokens
    logger.info('Make Tokens -> vocab size: {}'.format(len(tokens)))
    return tokens


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--train', default = 'train.txt')
    parser.add_argument('--valid', default = 'valid.txt')
    parser.add_argument('--max-len', type = int, default = 120)
    return parser.parse_args()


def main():
    args = parse_args()

    train_sents = get_train_sents(args.train, args.max_len)
    tokens = make_tokens(train_sents)

    with open('vocab.txt', 'w') as f:
        for x in tokens:
            print(x, file = f)

    vocab = Vocab(tokens)
    valid_sents = get_valid_sents(args.valid)

    make_raw('data', 'train', train_sents)
    make_raw('data', 'valid', valid_sents)

    train_data = sents_to_data(vocab, train_sents)
    valid_data = sents_to_data(vocab, valid_sents)
    make_seriejo('data', 'train', train_data)
    make_seriejo('data', 'valid', valid_data)

