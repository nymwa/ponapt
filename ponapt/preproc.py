from ilonimi import (
        Normalizer,
        Tokenizer,
        Splitter)

class LMPreproc:

    def __init__(
            self,
            convert_unk = True,
            convert_number = False,
            convert_proper = False,
            sharp = True):

        self.normalizer = Normalizer()

        self.tokenizer = Tokenizer(
                convert_unk = convert_unk,
                convert_number = convert_number,
                convert_proper = convert_proper)

        self.splitter = Splitter(sharp = sharp)

    def __call__(self, sent):
        sent = sent.strip()
        sent = self.normalizer(sent)
        sent = self.tokenizer(sent)
        sent = self.splitter(sent)
        return sent

