import numpy as np
import torch
from ponapt.batch import Batch
from ponapt.collator import generate_square_subsequent_mask
from .beam import Beam
from .sentence import ScoredSentence

class BeamSearch:

    def __init__(self, model, vocab, width, max_len = None):
        self.model = model
        self.model.eval()
        self.vocab = vocab
        self.width = width
        self.max_len = max_len if max_len is not None else 128

    def calc_logit(self, sent):
        inputs = torch.tensor([[self.vocab.bos] + sent]).T
        mask = generate_square_subsequent_mask(inputs.shape[0])
        batch = Batch(inputs, mask = mask)
        if torch.cuda.is_available():
            batch.cuda()

        with torch.no_grad():
            pred = self.model(batch)
        logit = pred[-1, 0, :]
        logit[self.vocab.pad] = float('-inf')
        return logit

    def calc_score(self, logit):
        return torch.log_softmax(logit, dim = -1)

    def step_for_not_empty_beam(self, beam):
        new_beam = []
        for n in range(len(beam.sents)):
            logit = self.calc_logit(beam.sents[n].sent)
            values, indices = self.calc_score(logit).topk(self.width)
            for value, index in zip(values, indices):
                sent = beam.sents[n].add(value.item(), index.item())
                new_beam.append(sent)

        new_beam += beam.store
        new_beam.sort(key = lambda sent: -sent.score())
        return Beam(self.width, new_beam[:self.width])

    def step_for_empty_beam(self):
        logit = self.calc_logit([])
        values, indices = torch.log_softmax(logit, dim = -1).topk(self.width)
        new_beam = [
                ScoredSentence(self.vocab, [value.item()], [index.item()])
                for value, index in zip(values, indices)]
        return Beam(self.width, new_beam)

    def step(self, beam = None):
        if beam is not None and beam.terminated():
            pass
        elif beam is None or len(beam.sents) == 0:
            beam = self.step_for_empty_beam()
        else:
            beam = self.step_for_not_empty_beam(beam)
        return beam


class NoisingBeamSearch(BeamSearch):

    def __init__(self, model, vocab, width, max_len = None, beta = None):
        super().__init__(model, vocab, width, max_len = max_len)
        self.beta = beta if beta is not None else 1.0

    def calc_score(self, logit):
        lprobs = torch.log_softmax(logit, dim = -1)
        lprobs += self.beta * torch.rand_like(lprobs)
        return lprobs


