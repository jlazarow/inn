import numpy as np
import pdb
import tensorflow as tf

from inn.networks.dcgan import dcgan
from inn.util import shape_as_string

PIXEL_MINIMUM = 0.0
PIXEL_MAXIMUM = 255.0

class Network(object):
    def __init__(self, name, input_size=None):
        self.name = name
        self._input_size = input_size

    def __call__(self, inputs, is_training=False):
        raise NotImplementedError("__call__")

    def cross_entropy_loss(self, labels, output):
        return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output)

    def classification_accuracy(self, labels, output):
        correct_training_predictions = tf.equal(
            tf.argmax(output, 1),
            tf.cast(labels, tf.int64))

        return tf.cast(correct_training_predictions, tf.float32)

    def count_number_parameters(self):
        count = 0
        trainable_variables = tf.trainable_variables()

        for variable in trainable_variables:
            shape = variable.get_shape()
            count += np.prod(shape.as_list())

        return count

    @property
    def input_size(self):
        return self._input_size

    @property
    def input_colorspace(self):
        return "RGB"

class DCGAN(Network):
    def __init__(self, input_size, number_classes, number_initial_filters=64):
        super(DCGAN, self).__init__("DCGAN", input_size)

        self.number_classes = number_classes
        self.number_initial_filters = number_initial_filters

    def __call__(self, inputs, is_training=False):
        return dcgan(
            inputs,
            self.number_initial_filters,
            number_classes=self.number_classes,
            is_training=is_training)
