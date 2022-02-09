class Batch:

    def __init__(
            self,
            inputs,
            outputs = None,
            lengths = None,
            position = None,
            padding = None,
            mask = None,
            misc = None):

        self.inputs = inputs
        self.outputs = outputs
        self.lengths = lengths
        self.position = position
        self.padding = padding
        self.mask = mask
        self.misc = misc

    def __len__(self):
        return self.inputs.shape[1]

    def get_num_tokens(self):
        return sum(self.lengths)

    def cuda(self):
        self.inputs = self.inputs.cuda()

        if self.outputs is not None:
            self.outputs = self.outputs.cuda()

        if self.position is not None:
            self.position = self.position.cuda()

        if self.padding is not None:
            self.padding = self.padding.cuda()

        if self.mask is not None:
            self.mask = self.mask.cuda()

        return self

