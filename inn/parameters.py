import tensorflow as tf

class DiscriminativeClass(object):
    """
    Represents a class (e.g. a class of digits) that should be used within training.


    Args:
      positives_initializer_fn: A function that returns some sort of ImageProvider. This function should take no arguments.
      negatives_initializer_fn: A function that takes a single argument, the number of images, and returns sort of ImageProvider.
      prior_probability: A float in (0, 1.0) that denotes the prior probability of drawing an example from this class.
    
    """
    def __init__(self, name, label, positives_initializer_fn, negatives_initializer_fn, prior_probability, synthesis_initializer_fn=None):
        self.name = name
        self.label = label
        self.positives_initializer_fn = positives_initializer_fn
        self.negatives_initializer_fn = negatives_initializer_fn
        self.synthesis_initializer_fn = synthesis_initializer_fn
        self.prior_probability = prior_probability
