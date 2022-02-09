def load_vocab(path):

    with open(path) as f:
        tokens = [x.strip() for x in f]

    vocab = Vocab(tokens)
    return vocab


class Vocab:

    def __init__(self, tokens):
        self.tokens = tokens
        self.token_dict = {token: index for index, token in enumerate(tokens)}
        self.pad = self.token_dict['<pad>']
        self.bos = self.token_dict['<bos>']
        self.eos = self.token_dict['<eos>']
        self.unk = self.token_dict['<unk>']

    def __contain__(self, x):
        return x in self.token_dict

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, x):
        return self.tokens[x]

    def __call__(self, x):
        if x in self:
            return self.token_dict[x]
        return self.unk

