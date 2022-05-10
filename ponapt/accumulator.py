class Accumulator:

    def __init__(self, name):
        self.name = name

        self.clear_epoch()
        self.clear_tmp()

    def update(self, batch, loss, lr):
        self.tmp_loss_list.append(loss)
        self.tmp_wpb_list.append(batch.get_num_tokens())
        self.tmp_spb_list.append(len(batch))
        self.tmp_lr_list.append(lr)

    def step_tmp(self):
        loss = sum(self.tmp_loss_list) / len(self.tmp_loss_list)
        wpb = sum(self.tmp_wpb_list)
        spb = sum(self.tmp_spb_list)
        lr = sum(self.tmp_lr_list) / len(self.tmp_lr_list)

        self.loss_list.append(loss)
        self.wpb_list.append(wpb)
        self.spb_list.append(spb)
        self.lr_list.append(lr)
        return loss, wpb, spb, lr

    def clear_tmp(self):
        self.tmp_loss_list = []
        self.tmp_wpb_list = []
        self.tmp_spb_list = []
        self.tmp_lr_list = []

    def clear_epoch(self):
        self.loss_list = []
        self.wpb_list = []
        self.spb_list = []
        self.lr_list = []

    def step_log(self, epoch, num_steps, grad = None):
        loss, wpb, spb, lr = self.step_tmp()
        self.clear_tmp()

        line = '| {}-inner'.format(self.name)
        line += ' | epoch {}, {}/{}'.format(
                epoch,
                len(self.spb_list),
                num_steps)
        line += ' | loss {:.4f}'.format(loss)
        line += ' | lr {:.8f}'.format(lr)
        if grad:
            line += ' | grad {:.4f}'.format(grad)
        line += ' | w/b {}'.format(wpb)
        line += ' | s/b {}'.format(spb)

        return line

    def avg(self, lst):
        num_examples = sum(self.spb_list)
        return sum([n * x for n, x in zip(self.spb_list, lst)]) / num_examples

    def epoch_log(self, epoch, num_steps = None):
        line = '| {}'.format(self.name)
        line += ' | epoch {}'.format(epoch)
        line += ' | loss {:.4f}'.format(self.avg(self.loss_list))
        line += ' | lr {:.8f}'.format(self.avg(self.lr_list))
        line += ' | w/b {:.1f}'.format(self.avg(self.wpb_list))
        line += ' | s/b {:.1f}'.format(self.avg(self.spb_list))
        if num_steps is not None:
            line += ' | steps {}'.format(num_steps)
        self.clear_epoch()
        return line

