class Beam:

    def __init__(self, width, sents = None):

        self.width = width

        self.sents = []
        self.store = []
        if sents is not None:

            assert len(sents) <= width

            for sent in sents:
                if sent.terminated():
                    self.store.append(sent)
                else:
                    self.sents.append(sent)

        self.rest = width - len(self.store)

    def terminated(self):
        return self.rest == 0

