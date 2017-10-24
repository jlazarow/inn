import collections
import numpy as np
import pdb

class TrainingSummarizer(object):
    def __init__(self, ops):
        self.ops = ops

    def step(self, fetches):
        raise ValueError("should implement step()")

    def iteration(self, steps):
        raise ValueError("should implement iteration")

class LossAndAccuracySummarizer(TrainingSummarizer):
    Summary = collections.namedtuple("LossAndAccuracySummary", ["loss", "accuracy"])
    
    def __init__(self, accuracy_op):
        # note that (for now) loss is _always_ included in ops
        # as the first fetch.
        super(LossAndAccuracySummarizer, self).__init__([accuracy_op])

    def step(self, fetches):
        loss, accuracy = fetches[:2]

        return LossAndAccuracySummarizer.Summary(loss=loss, accuracy=accuracy)

    def reduce(self, steps):
        number_steps = float(len(steps))
        average_loss = sum(map(lambda r: r.loss, steps)) / number_steps
        average_accuracy = sum(map(lambda r: r.accuracy, steps)) / number_steps

        return LossAndAccuracySummarizer.Summary(loss=average_loss, accuracy=average_accuracy)
