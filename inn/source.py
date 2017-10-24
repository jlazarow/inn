import collections
import numpy as np
import pdb

from inn.batch import concatenate_batches, set_of_labels, InputBatch
from inn.util import float_equals

def sums_to_one(a):
    return float_equals(np.sum(a), 1.0)

class TrainingDataHistory(object):
    def __init__(self, batch, name=None):
        self.batches = [batch]
        self.name = name
        self.size = batch.size

    @property
    def total_batch(self):
        return concatenate_batches(self.batches)

    @property
    def age(self):
        return len(self.batches)

    @property
    def most_recent(self):
        return self.batches[-1]

    @property
    def nbytes(self):
        total_bytes = 0
        for batch in self.batches:
            total_bytes += batch.nbytes

        return total_bytes

    def total_batch_with_labels(self, label):
        specific_labels = set_of_labels(label, self.size)

        return concatenate_batches(self.batches, labels=specific_labels)

    def __len__(self):
        return self.size        

    def add(self, batch, label=None):
        if not (label is None):
            batch = batch.relabel(label)
            
        self.batches.append(batch)
        self.size += batch.size

    def sampled(self, count_per_age):
        number_samples = np.sum(count_per_age)
        sampled = InputBatch(
            images=np.zeros((number_samples,) + self.batches[0].image_shape, dtype=np.float32),
            names=np.array(number_samples * ["placeholder"], dtype=object),
            labels=set_of_labels(-1, number_samples))

        current_index = 0
        for history_index in range(self.age):
            batch_at_age = self.batches[history_index]
            count_at_age = count_per_age[history_index]

            if count_at_age > batch_at_age.size:
                choices_at_age = np.random.choice(np.arange(batch_at_age.size, dtype=np.int64), count_at_age)
                sampled.insert(batch_at_age[choices_at_age], current_index)
            else:
                permutation_at_age = np.random.permutation(batch_at_age.size)
                sampled.insert(batch_at_age[permutation_at_age[:count_at_age]], current_index)

            current_index += count_at_age

        return sampled

    # also a bad function for big datasets. currently only used
    # for positives, however.
    def chosen(self, count):
        # note that this won't work for classes that grow.
        number_first_batch_examples = self.batches[0].size
        indexes = np.random.choice(np.arange(number_first_batch_examples, dtype=np.int64),
                                   count,
                                   replace=(count > number_first_batch_examples))

        return self.batches[0][indexes]

class TrainingDataClass(object):
    def __init__(self, positives, negatives, synthesis, probability, positive_label, negative_label):
        self.positives = positives
        self.negatives = negatives
        self.synthesis = synthesis
        self.probability = probability
        self.positive_label = positive_label
        self.negative_label = negative_label

    @property
    def nbytes(self):
        positives_nbytes = self.positives.nbytes
        negatives_nbytes = self.negatives.nbytes
        synthesis_nbytes = self.synthesis.nbytes

        return positives_nbytes + negatives_nbytes + synthesis_nbytes
    
    def add(self, synthesized, suffix, next_synthesis=None):
        if not (self.negatives is None):
            self.negatives.add(synthesized, label=self.negative_label)

        # by default, use the previous synthesized here.
        if next_synthesis is None:
            next_synthesis = synthesized.suffixed(suffix)

        self.synthesis = TrainingDataHistory(
            next_synthesis,
            self.synthesis.name)

def create_evenly_split_training_source(class_, negative_label):
    print "reading positives of class {0} ({1})".format(class_.name, class_.label)
    positives_initializer = class_.positives_initializer_fn()
    positives = TrainingDataHistory(
        batch=InputBatch(
            images=positives_initializer.images,
            names=positives_initializer.names,
            labels=set_of_labels(class_.label, positives_initializer.number_images)),
        name="{0}-positives".format(class_.label))

    # match the number of positives for now.
    print "reading negatives of class {0} ({1})".format(class_.name, negative_label)
    negatives_initializer = class_.negatives_initializer_fn(len(positives), )
    negatives = TrainingDataHistory(
        batch=InputBatch(
            images=negatives_initializer.images,
            names=negatives_initializer.names,
            labels=set_of_labels(negative_label, negatives_initializer.number_images)),
        name="{0}-negatives".format(class_.label))            

    # note that the synthesis carries the true positive label.
    # this will be modified once a synthesis is added to the negatives.
    training_class = TrainingDataClass(
        positives=positives,
        negatives=negatives,
        synthesis=TrainingDataHistory(
            batch=negatives.total_batch.relabel(class_.label).suffixed("round0"),
            name="{0}-synthesis".format(class_.label)),
        probability=class_.prior_probability,
        positive_label=class_.label,
        negative_label=negative_label)

    return training_class
