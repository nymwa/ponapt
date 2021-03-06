import torch
from .sampling import (
        top_p_sampling,
        calc_logit)

class SentenceSampler:

    def __init__(
            self,
            vocab,
            model):

        self.vocab = vocab
        self.model = model

    def postproc_logit(
            self,
            logit,
            index,
            max_tokens,
            stop_ratio,
            beta,
            avoid_eos,
            terminal):

        logit[self.vocab.pad] = float('-inf')
        logit[self.vocab.bos] = float('-inf')
        logit[self.vocab.unk] = float('-inf')

        if avoid_eos:
            logit[self.vocab.eos] = float('-inf')
            if terminal is not None:
                for token in terminal:
                    logit[token] = float('-inf')

        if index > max_tokens * stop_ratio:
            logit[self.vocab.eos] += (index - max_tokens * stop_ratio) * beta
        return logit


    def get_next_token(
            self,
            sent,
            temperature,
            top_p,
            index,
            max_tokens,
            stop_ratio,
            beta,
            avoid_eos,
            terminal):

        logit = calc_logit(
                self.model,
                self.vocab,
                sent)

        logit = self.postproc_logit(
                logit,
                index,
                max_tokens,
                stop_ratio,
                beta,
                avoid_eos,
                terminal)

        next_token = top_p_sampling(
                logit,
                temperature,
                top_p)

        return next_token

    def __call__(
            self,
            temperature = 1.0,
            top_p = 0.8,
            max_tokens = 128,
            stop_ratio = 0.3,
            beta = 1.0,
            sent = None,
            terminal = None,
            min_len = None):

        if sent is None:
            sent = []

        for index in range(len(sent), max_tokens):

            avoid_eos = (min_len is not None) and (index < min_len)

            next_token = self.get_next_token(
                    sent,
                    temperature,
                    top_p,
                    index,
                    max_tokens,
                    stop_ratio,
                    beta,
                    avoid_eos,
                    terminal)

            if next_token == self.vocab.eos:
                break
            sent.append(next_token)
            if terminal is not None and next_token in terminal:
                break

        return sent

