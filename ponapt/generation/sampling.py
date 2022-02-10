import numpy as np
import torch
from ponapt.batch import Batch
from ponapt.collator import generate_square_subsequent_mask

def top_p_sampling(
        logit,
        temperature,
        top_p):

    logit = logit / temperature
    probs = torch.softmax(logit, dim = -1)
    values, indices = torch.sort(probs)

    cumlated = torch.cumsum(values, -1)
    is_removed = cumlated < (1 - top_p)
    probs[indices[is_removed]] = 0

    probs = probs.cpu().numpy()
    probs = probs / sum(probs)
    next_token = np.random.choice(range(len(probs)), p = probs)
    return next_token


def calc_logit(model, vocab, sent):
    inputs = torch.tensor([[vocab.bos] + sent]).T
    mask = generate_square_subsequent_mask(inputs.shape[0])
    batch = Batch(inputs, mask = mask)
    if torch.cuda.is_available():
        batch.cuda()

    with torch.no_grad():
        pred = model(batch)
    logit = pred[-1, 0, :]
    return logit

