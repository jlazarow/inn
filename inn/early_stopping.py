import collections

from inn.util import render_summary

class EarlyStoppingCriterion(object):
    def __init__(self, stop_fn, at_least_n=0):
        self.stop_fn = stop_fn
        self.at_least_n = at_least_n
        self.current_n = 0

    def __call__(self, value):
        self.current_n += 1

        if self.current_n < self.at_least_n:
            return False
        
        return self.stop_fn(value)

    @property
    def summary(self):
        values = [
            ["at least n", self.at_least_n]
        ]

        return render_summary(values, title="Early Stopping Criterion")    

class WindowedEarlyStoppingCriterion(EarlyStoppingCriterion):
    def __init__(self, stop_fn, window_size=10):
        super(WindowedEarlyStoppingCriterion, self).__init__(stop_fn, at_least_n=window_size)
        
        self.window_size = window_size
        self.history = collections.deque(maxlen=self.window_size)

    def __call__(self, value):
        self.history.append(value) 
        average_value = sum(self.history) / float(len(self.history))

        return super(WindowedEarlyStoppingCriterion, self).__call__(average_value)

    @property
    def summary(self):
        values = [
            ["at least n", str(self.at_least_n)],
            ["window size", str(self.window_size)]
        ]

        return render_summary(values, title="(Windowed) Early Stopping Criterion")
    
