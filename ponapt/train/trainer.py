from pathlib import Path
import torch

from ponapt.accumulator import Accumulator
from ponapt.batch import Batch
from ponapt.log import init_logging
from logging import getLogger
init_logging()
logger = getLogger(__name__)

class Trainer:

    def __init__(
            self,
            train_loader,
            valid_loader,
            model,
            opter,
            losscalc,
            max_epochs,
            step_interval,
            save_interval):

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.opter = opter
        self.losscalc = losscalc
        self.losscalc.set_trainer(self)

        self.max_epochs = max_epochs
        self.epoch = 0
        self.step = 0
        self.step_interval = step_interval
        self.num_accum = 0
        self.grad_to_init = True
        self.save_interval = save_interval
        self.train_accum = Accumulator('train')

    def train(self):
        self.model.train()
        num_steps = (self.num_accum + len(self.train_loader)) // self.step_interval
        this_step = 0
        for index, batch in enumerate(self.train_loader):

            if self.grad_to_init:
                self.opter.zero_grad()
                self.grad_to_init = False

            with torch.cuda.amp.autocast():
                loss = self.losscalc.for_train(batch)
                loss_val = loss.item()
                loss = loss / self.step_interval
            self.opter.scaler.scale(loss).backward()
            self.num_accum += 1
            self.train_accum.update(batch, loss_val, self.opter.get_lr())

            if self.num_accum == self.step_interval:
                this_step += 1
                self.step += 1
                self.num_accum = 0
                self.opter.step()
                self.grad_to_init = True
                logger.info(self.train_accum.step_log(
                    self.epoch,
                    num_steps,
                    grad = self.opter.total_grad_norm))

        logger.info(self.train_accum.epoch_log(self.epoch, num_steps))

    def valid(self):
        self.model.eval()

        accum = Accumulator('valid')
        for index, batch in enumerate(self.valid_loader):
            with torch.no_grad():
                loss = self.losscalc.for_valid(batch)
            accum.update(batch, loss.item(), self.opter.get_lr())
        accum.step_log(self.epoch, len(self.valid_loader))
        logger.info(accum.epoch_log(self.epoch))

    def save_checkpoint(self):
        Path('checkpoints').mkdir(parents = True, exist_ok = True)
        path = 'checkpoints/{}.pt'.format(self.epoch)
        torch.save(self.model.state_dict(), path)
        logger.info('| checkpoint | saved to {}'.format(path))

    def run(self):
        for _ in range(self.max_epochs):
            self.epoch += 1
            self.train()
            self.valid()
            if self.epoch % self.save_interval == 0:
                self.save_checkpoint()

