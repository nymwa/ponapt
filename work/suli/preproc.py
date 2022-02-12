import math
from collections import Counter
from contextlib import ExitStack
from seriejo import SeriejoWriter
from pathlib import Path
import random as rd
from ponapt.vocab import Vocab
from ponapt.preproc import LMPreproc
from ponapt.log import init_logging
from logging import getLogger
from collections import Counter
init_logging()
logger = getLogger(__name__)

path_list = [
        '../../../tokipona-corpus-collection/100tokipona/100tokipona.txt',
        '../../../tokipona-corpus-collection/tokipona1000/tokipona1000.txt',
        '../../../tokipona-corpus-collection/tatoeba/tonconved.txt',
        '../../../tokipona-corpus-collection/pu/pu.txt',
        '../../../tokipona-corpus-collection/matthew/dave.txt',
        '../../../tokipona-corpus-collection/matthew/mika.txt',
        '../../../tokipona-corpus-collection/matthew/ote.txt',
        '../../../tokipona-corpus-collection/matthew/prince.txt',
        '../../../tokipona-corpus-collection/nanko/panelopi.txt',
        '../../../tokipona-corpus-collection/sita/namako.txt']


augmented_list = [
        '../../../tokipona-corpus-collection/augmented/data.0.txt',
        '../../../tokipona-corpus-collection/augmented/data.1.txt',
        '../../../tokipona-corpus-collection/augmented/data.2.txt',
        '../../../tokipona-corpus-collection/augmented/data.3.txt',
        '../../../tokipona-corpus-collection/augmented/data.4.txt',
        '../../../tokipona-corpus-collection/augmented/data.5.txt',
        '../../../tokipona-corpus-collection/augmented/data.6.txt',
        '../../../tokipona-corpus-collection/augmented/data.7.txt',
        '../../../tokipona-corpus-collection/augmented/data.8.txt',
        '../../../tokipona-corpus-collection/augmented/data.9.txt']


def sents_to_data(vocab, sents):
    def sent_to_data(sent):
        sent = sent.split()
        sent = [vocab(token) for token in sent]
        return sent
    return [sent_to_data(sent) for sent in sents]


def make_seriejo(base, name, data):
    Path(base).mkdir(parents = True, exist_ok = True)
    with SeriejoWriter('{}/{}'.format(base, name)) as f:
        for x in data:
            f.write(x)
    logger.info('Write Seriejo ({}/{}): {}'.format(base, name, len(data)))

def load_corpora(corpora):
    preproc = LMPreproc()

    with ExitStack() as stack:
        sents = [
            preproc(sent)
            for corpus
            in corpora
            for sent
            in stack.enter_context(open(corpus))]
    return sents



def get_sents():
    sents = load_corpora(path_list)
    logger.info('data loaded')

    sents = [
        sent
        for sent
        in sents
        if 1 <= len(sent.split()) <= 120]
    logger.info('data filtered')
    return sents


def get_augmented_sents():
    raw_sents = load_corpora(augmented_list)
    logger.info('augmented data loaded')

    counter = Counter(raw_sents)
    sents = []
    for sent, freq in counter.most_common():
        n = math.ceil(math.sqrt(freq))
        for _ in range(n):
            sents.append(sent)
    logger.info('augmented data filtered (1)')

    sents = [
        sent
        for sent
        in sents
        if 1 <= len(sent.split()) <= 120]
    logger.info('augmented data filtered (2)')
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


def split_sents(sents, valid_size, test_size):
    train_sents = sents[: -(valid_size + test_size)]
    valid_sents = sents[-(valid_size + test_size) : -test_size]
    test_sents = sents[-test_size :]
    logger.info('split -> train: {}'.format(len(train_sents)))
    logger.info('split -> valid: {}'.format(len(valid_sents)))
    logger.info('split -> test : {}'.format(len(test_sents)))
    return train_sents, valid_sents, test_sents


def main():
    sents = get_sents()

    rd.seed(100)
    rd.shuffle(sents)

    train_sents, valid_sents, test_sents = split_sents(
            sents,
            valid_size = 2000,
            test_size = 2000)
    train_sents += get_augmented_sents()

    tokens = make_tokens(train_sents)
    with open('vocab.txt', 'w') as f:
        for x in tokens:
            print(x, file = f)

    vocab = Vocab(tokens)

    train_data = sents_to_data(vocab, train_sents)
    valid_data = sents_to_data(vocab, valid_sents)
    test_data = sents_to_data(vocab, test_sents)
    make_seriejo('data', 'train', train_data)
    make_seriejo('data', 'valid', valid_data)
    make_seriejo('data', 'test', test_data)


if __name__ == '__main__':
    main()

