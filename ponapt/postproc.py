from ilonimi import (
        Joiner,
        Detokenizer)

class LMPostproc:

    def __init__(
            self,
            merge = True,
            no_sharp = False):

        self.merge = merge
        self.joiner = Joiner(no_sharp = no_sharp)
        self.detokenizer = Detokenizer()

    def __call__(self, sent):
        sent = sent.strip()
        if self.merge:
            sent = self.joiner(sent)
        sent = self.detokenizer(sent)
        return sent

