from argparse import ArgumentParser
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from seriejo import Seriejo
from ponapt.vocab import load_vocab
from ponapt.dataset import Dataset
from ponapt.sampler import FixedSampler, RandomSampler
from ponapt.collator import Collator, TrainingCollator

from ponapt.accumulator import Accumulator

from ponapt.train.model import get_lm_model
from ponapt.train.opter import Opter
from ponapt.train.losscalc import PonaptLossCalc
from ponapt.train.trainer import Trainer

from ponapt.log import init_logging
from logging import getLogger
init_logging()
logger = getLogger(__name__)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--vocab', default = 'vocab.txt')
    parser.add_argument('--max-tokens', type = int, default = 4000)
    parser.add_argument('--shift-prob', type = float, default = 0.99)
    parser.add_argument('--max-shift', type = int, default = 128)

    parser.add_argument('--hidden-dim', type = int, default = 512)
    parser.add_argument('--nhead', type = int, default = 8)
    parser.add_argument('--feedforward-dim', type = int, default = 2048)
    parser.add_argument('--num-layers', type = int, default = 6)
    parser.add_argument('--dropout', type = float, default = 0.3)
    parser.add_argument('--word-dropout', type = float, default = 0.3)
    parser.add_argument('--attention-dropout', type = float, default = 0.2)
    parser.add_argument('--activation-dropout', type = float, default = 0.2)
    parser.add_argument('--max-len', type = int, default = 256)
    parser.add_argument('--no-share-embedding', action = 'store_true')

    parser.add_argument('--label-smoothing', type = float, default = 0.0)
    parser.add_argument('--lr', type = float, default = 0.0001)
    parser.add_argument('--weight-decay', type = float, default = 0.01)
    parser.add_argument('--max-grad-norm', type = float, default = 1.0)
    parser.add_argument('--scheduler', default = 'linexp')
    parser.add_argument('--warmup-steps', type = int, default = 4000)
    parser.add_argument('--start-factor', type = float, default = 1.0)

    parser.add_argument('--epochs', type = int, default = 200)
    parser.add_argument('--step-interval', type = int, default = 1)
    parser.add_argument('--save-interval', type = int, default = 10)
    return parser.parse_args()


def load_dataset(name):
    data = Seriejo('data/{}'.format(name))
    dataset = Dataset(data)
    return dataset


def load_loaders(vocab, args):
    train_dataset = load_dataset('train')
    valid_dataset = load_dataset('valid')
    train_sampler = RandomSampler(train_dataset, args.max_tokens)
    valid_sampler = FixedSampler(valid_dataset, args.max_tokens)
    train_collator = TrainingCollator(
            vocab,
            args.shift_prob,
            args.max_shift)
    valid_collator = Collator(vocab)
    train_loader = DataLoader(
            train_dataset,
            batch_sampler = train_sampler,
            collate_fn = train_collator)
    valid_loader = DataLoader(
            valid_dataset,
            batch_sampler = valid_sampler,
            collate_fn = valid_collator)
    return train_loader, valid_loader




def train_step(criterion, optimizer, scheduler, clip_norm, model, batch):
    batch.cuda()
    pred = model(batch)
    pred = pred.view(-1, pred.size(-1))
    loss = criterion(pred, batch.outputs.view(-1))
    optimizer.zero_grad()
    loss.backward()
    if clip_norm > 0:
        grad = nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    else:
        grad = None
    optimizer.step()
    scheduler.step()
    return loss, grad


def valid_step(criterion, model, batch):
    batch.cuda()
    with torch.no_grad():
        pred = model(batch)
        pred = pred.view(-1, pred.size(-1))
        loss = criterion(pred, batch.outputs.view(-1))
    return loss


def step_log(scheduler, accum, batch, loss, grad = None):
    lr = scheduler.get_last_lr()[0]
    accum.update(batch, loss, lr, grad)
    logger.info(accum.step_log())


def save_checkpoint(model, epoch):
    Path('checkpoints').mkdir(parents = True, exist_ok = True)
    path = 'checkpoints/lm.{}.pt'.format(epoch)
    torch.save(model.state_dict(), path)
    logger.info('| checkpoint | saved to {}'.format(path))


def training(
        epochs,
        save_interval,
        train_criterion,
        valid_criterion,
        optimizer,
        scheduler,
        clip_norm,
        train_loader,
        valid_loader,
        model):

    num_steps = 0
    for epoch in range(1, epochs + 1):

        model.train()
        accum = Accumulator('train', epoch, len(train_loader))
        for step, batch in enumerate(train_loader):
            loss, grad = train_step(train_criterion, optimizer, scheduler, clip_norm, model, batch)
            num_steps += 1
            step_log(scheduler, accum, batch, loss, grad)
        logger.info(accum.epoch_log(num_steps))
        if epoch % save_interval == 0:
            save_checkpoint(model, epoch)

        model.eval()
        accum = Accumulator('valid', epoch, len(valid_loader))
        for step, batch in enumerate(valid_loader):
            loss = valid_step(valid_criterion, model, batch)
            step_log(scheduler, accum, batch, loss)
        logger.info(accum.epoch_log(num_steps))


def main():
    args = parse_args()
    print(args)

    vocab = load_vocab(args.vocab)
    train_loader, valid_loader = load_loaders(vocab, args)

    model = get_lm_model(vocab, args)
    model = model.cuda()
    logger.info('#params : {} ({})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    opter = Opter(
            model,
            args.lr,
            max_grad_norm = args.max_grad_norm,
            scheduler = args.scheduler,
            warmup_steps = args.warmup_steps,
            start_factor = args.start_factor,
            weight_decay = args.weight_decay)
    losscalc = PonaptLossCalc(label_smoothing = args.label_smoothing)

    trainer = Trainer(
            train_loader,
            valid_loader,
            model,
            opter,
            losscalc,
            args.epochs,
            args.step_interval,
            args.save_interval)

    trainer.run()

