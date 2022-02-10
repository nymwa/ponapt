class ScoredSentence:

    def __init__(self, vocab, log_probs, sent):
        self.vocab = vocab
        self.log_probs = log_probs
        self.sent = sent

    def score(self):
        return sum(self.log_probs)

    def last(self):
        return self.sent[-1]

    def terminated(self):
        return self.last() == self.vocab.eos

    def add(self, log_prob, token):
        log_probs = self.log_probs + [log_prob]
        sent = self.sent + [token]
        return ScoredSentence(self.vocab, log_probs, sent)

    def as_str(self):
        return ' '.join([self.vocab[token] for token in self.sent])

