import numpy as np
import pdb
import resource

from inn.util import float_equals, evenly_divided_by

def set_of_labels(label, count):
    labels = np.zeros((count,), dtype=np.int64)
    labels.fill(label)

    return labels

SUFFIX_SEPARATOR =  "@"

class InputBatch(object):
    __slots__ = ["images", "names", "labels"]

    def __init__(self, images, names, labels):
        self.images = images
        self.names = names
        self.labels = labels

    def __getitem__(self, key):
        return InputBatch(
            images=self.images.__getitem__(key),
            names=self.names.__getitem__(key),
            labels=self.labels.__getitem__(key))
    
    def __add__(self, b):
        return InputBatch(
            images=np.concatenate((self.images, b.images)),
            names=np.concatenate((self.names, b.names)),
            labels=np.concatenate((self.labels, b.labels)))

    @property
    def nbytes(self):
        return self.images.nbytes + self.names.nbytes + self.labels.nbytes

    def insert(self, batch, index):
        self.images[index:(index + batch.size)] = batch.images
        self.names[index:(index + batch.size)] = batch.names
        self.labels[index:(index + batch.size)] = batch.labels

    def resize(self, count):
        self.images.resize((count,) + self.images.shape[1:])
        self.names.resize((count,))
        self.labels.resize((count,))

    def relabel(self, label, copy=False):
        images = self.images
        names = self.names

        if copy:
            images = np.copy(images)
            names = np.copy(names)
            
        return InputBatch(
            images=images,
            names=names,
            labels=set_of_labels(label, self.size))

    def suffixed(self, suffix):
        # do this the slow way for now.
        names = list(self.names)
        suffixed_names = []
        for name in names:
            suffix_index = name.rfind(SUFFIX_SEPARATOR)
            if suffix_index < 0:
                suffixed_name = name + SUFFIX_SEPARATOR + suffix
            else:
                suffixed_name = name[:suffix_index] + SUFFIX_SEPARATOR + suffix

            suffixed_names.append(suffixed_name)
                                
        return InputBatch(
            images=self.images,
            names=np.array(suffixed_names),
            labels=self.labels)
    
    def copy(self):
        return InputBatch(
            images=np.copy(self.images),
            names=np.copy(self.names),
            labels=np.copy(self.labels))
    
    @property
    def size(self):
        return np.shape(self.images)[0]

    @property
    def image_shape(self):
        return np.shape(self.images)[1:]

def concatenate_batches(batches, labels=None):
    return InputBatch(
        images=np.concatenate(map(lambda b: b.images, batches)),
        names=np.concatenate(map(lambda b: b.names, batches)),
        labels=labels if (labels is not None) else np.concatenate(map(lambda b: b.labels, batches)))

# abstract "producer" of batches
class BatchProducer(object):
    def __init__(self, ctx, shuffle=True, exact_number_batches=False, total_batch=None):
        self.ctx = ctx
        self.shuffle = shuffle
        self.exact_number_batches = exact_number_batches
        
        self.total_batch = total_batch
        self.permutation = None
        self.current_index = 0
        self.prepare_total_batch()

    @property
    def size(self):
        if not (self.permutation is None):
            return len(self.permutation)
        
        return self.total_batch.size
    
    def evenly_divided_by(self, batch_size):
        return (self.size % batch_size) == 0

    def prepare_total_batch(self):
        raise NotImplementedError("should be overriden")

    def number_batches_for_size(self, batch_size):
        number_examples = self.size
        
        return int(np.ceil(number_examples / float(batch_size)))

    def __call__(self, batch_size):
        if self.current_index >= self.size:
            raise ValueError("cannot fulfill request, out of examples")

        # we'll allow for looping around slightly if we still have a few examples left.
        if self.current_index + batch_size > self.size:
            if self.exact_number_batches:
                raise ValueError("exact required and batch size does not evenly fit into the number of examples")
            
            number_remaining = self.size - self.current_index
            number_from_beginning = batch_size - number_remaining

            # we might want to just generate a default permutation to put this in one path.
            if not (self.permutation is None):
                batch = self.total_batch[self.permutation[self.current_index:]] + self.total_batch[self.permutation[:number_from_beginning]]
            else:
                batch = self.total_batch[self.current_index:] + self.total_batch[:number_from_beginning]
            
            # but don't let the user ask for any more batches.
            self.current_index = number_examples
        else:
            if not (self.permutation is None):
                batch = self.total_batch[self.permutation[self.current_index:(self.current_index + batch_size)]]
            else:
                batch = self.total_batch[self.current_index:(self.current_index + batch_size)]
            self.current_index += batch_size

        # we could additionally shuffle here.
        return batch

class TrainingBatchProducer(BatchProducer):
    def __init__(self, total_batch, classes, positive_probability, ctx, shuffle=True, exact_number_batches=True):
        self.classes = classes
        self.positive_probability = positive_probability
        if self.positive_probability <= 0.0 or self.positive_probability >= 1.0:
            raise ValueError("positive probability should be in the range (0, 1)")

        self.verify_distribution()

        super(TrainingBatchProducer, self).__init__(ctx, shuffle, exact_number_batches, total_batch=total_batch)

    def verify_distribution(self):
        distribution_sum = sum(map(lambda c: c.probability, self.classes))
        if not float_equals(distribution_sum, 1.0):
            raise ValueError("prior distribution of batch does not sum to 1")
        
    def select_counts_per_age(self, training_class, count):
        age_of_negatives = training_class.negatives.age

        # this is our (sub)sampling strategy for negatives. we want:
        #   - get a biased sample towards young examples
        #   - still get _some_ examples for each age.
        likelihood_of_age = np.logspace(age_of_negatives, 1, num=age_of_negatives, base=0.5)
        probability_of_age = likelihood_of_age / np.sum(likelihood_of_age)
        number_negatives_per_age = np.random.multinomial(count, probability_of_age)

        return number_negatives_per_age

    def prepare_total_batch(self):
        if self.total_batch is None:
            raise ValueError("total batch needs to be preallocated")

        current_index = 0
        for training_class in self.classes:
            number_examples = training_class.probability * self.size
            if float_equals(number_examples, 0.0) or not number_examples.is_integer():
                raise ValueError("class prior produced either zero examples or a non-integer amount {0}".format(number_examples))

            number_examples = int(number_examples)
            number_positives = int(self.positive_probability * number_examples)
            number_negatives = int((1.0 - self.positive_probability) * number_examples)

            if number_examples != (number_positives + number_negatives):
                raise ValueError("number of examples per class is not the number of positives plus negatives exactly")

            self.total_batch.insert(training_class.positives.chosen(number_positives), current_index)
            current_index += number_positives

            number_negatives_per_age = self.select_counts_per_age(training_class, number_negatives)
            self.ctx.batch_age_counts(number_negatives_per_age)
                
            self.total_batch.insert(training_class.negatives.sampled(number_negatives_per_age), current_index)
            current_index += number_negatives

        if current_index != self.total_batch.size:
            raise ValueError("expected to match total batch {0} and classes {1}".format(self.total_batch.size, current_index))

        if self.shuffle:
            self.permutation = np.random.permutation(self.total_batch.size)
        else:
            self.permutation = None
        
class SynthesisBatchProducer(BatchProducer):
    def __init__(self, total_batch, classes, ctx):
        self.classes = classes

        # remember where each class started.
        self.start_indexes = []

        super(SynthesisBatchProducer, self).__init__(ctx, shuffle=False, exact_number_batches=True, total_batch=total_batch)

    def prepare_total_batch(self):
        if self.total_batch is None:
            raise ValueError("total batch needs to be preallocated")

        current_index = 0
        for training_class in self.classes:
            self.start_indexes.append(current_index)
            
            number_synthesis = training_class.synthesis.size
            previous_synthesis_for_class = training_class.synthesis.most_recent
            self.total_batch.insert(previous_synthesis_for_class, current_index)
            
            current_index += number_synthesis

        if current_index != self.total_batch.size:
            raise ValueError("expected to match total batch {0} and classes {1}".format(self.total_batch.size, current_index))
            
