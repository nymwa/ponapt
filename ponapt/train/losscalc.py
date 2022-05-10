import torch
import torch.nn as nn

class LossCalc:

    def set_trainer(self, trainer):
        self.trainer = trainer

    def for_train(self, batch):
        pred, target = self.get_pred_and_target(batch)
        loss = self.train_criterion(pred, target)
        return loss

    def for_valid(self, batch):
        pred, target = self.get_pred_and_target(batch)
        loss = self.valid_criterion(pred, target)
        return loss


class PonaptLossCalc(LossCalc):

    def __init__(self, label_smoothing = 0.0):
        self.train_criterion = nn.CrossEntropyLoss(
            ignore_index = -100,
            label_smoothing = label_smoothing)
        self.valid_criterion = nn.CrossEntropyLoss(
            ignore_index = -100)

    def get_pred_and_target(self, batch):
        batch.cuda()
        pred = self.trainer.model(batch)
        pred = pred.view(-1, pred.size(-1))
        target = batch.outputs.view(-1)
        return pred, target

