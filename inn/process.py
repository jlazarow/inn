import bdb
import collections
import numpy as np
import os
import pdb
import resource
import tensorflow as tf

from tensorflow.python import debug as tfdbg

from inn.batch import InputBatch
from inn.util import shape_as_tuple, float_equals

TrainingStepResult = collections.namedtuple("TrainingStepResult", ["loss", "accuracy"])
SynthesisStepResult = collections.namedtuple("SynthesisStepResult", ["loss"])
SynthesisBatchResult = collections.namedtuple("SynthesisBatchResult", ["images", "names", "labels"])

AuxiliaryInput = collections.namedtuple("AuxiliaryInput", ["images", "names", "labels"])

def average_training_steps(results):
    average_loss = sum(map(lambda r: r.loss, results)) / float(len(results))
    average_accuracy = sum(map(lambda r: r.accuracy, results)) / float(len(results))

    return TrainingStepResult(loss=average_loss, accuracy=average_accuracy)
            
class BatchedInput(object):
    def __init__(self, batch_size, image_batch_shape):
        self.batch_size = batch_size
        
        self.images = tf.Variable(np.zeros((batch_size,) + image_batch_shape, dtype=np.float32), name="images")
        self.labels = tf.Variable(np.zeros((batch_size,), dtype=np.int64), name="labels")
        self.names = tf.Variable([""] * batch_size, dtype=tf.string, name="names")

    # invoking the input retrieves
    def __call__(self, session):
        current_images, current_labels, current_names = session.run([self.images, self.labels, self.names])

        return InputBatch(
            images=current_images,
            names=current_names,
            labels=current_labels)

class BatchLoader(object):
    def __init__(self, input):
        self.input = input
        self.images_placeholder = tf.placeholder(
            dtype=tf.float32, shape=self.input.images.get_shape())
        self.labels_placeholder = tf.placeholder(
            dtype=tf.int64, shape=self.input.labels.get_shape())
        self.names_placeholder = tf.placeholder(
            dtype=tf.string, shape=self.input.names.get_shape())

        self.assign_images = self.input.images.assign(self.images_placeholder)
        self.assign_labels = self.input.labels.assign(self.labels_placeholder)
        self.assign_names = self.input.names.assign(self.names_placeholder)
        self.batch_number = 0

    def __call__(self, session, images, labels, names):
        session.run([self.assign_images, self.assign_labels, self.assign_names],
                    feed_dict={
                        self.images_placeholder: images,
                        self.labels_placeholder: labels,
                        self.names_placeholder: names})
        self.batch_number += 1

class ProduceAndLoadBatch(object):
    def __init__(self, producer, loader, batch_size, batch_image_size):
        self.producer = producer
        self.loader = loader
        self.batch_size = batch_size
        self.batch_image_size = batch_image_size

    def __call__(self, session, shuffle=False):
        batch = self.producer(self.batch_size)
        self.loader(session, batch.images, batch.labels, batch.names)
        del batch

class DiscriminativeTrainingStep(object):
    def __init__(self, loss_fn, optimizer, optimize_variables, optimizer_step, summarizer):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optimize_variables = optimize_variables
        self.optimizer_step = optimizer_step

        self.optimize_operation = self.optimizer.minimize(
            self.loss_fn,
            global_step=self.optimizer_step,
            var_list=self.optimize_variables)
        self.summarizer = summarizer

    def __call__(self, session, ctx, step_number, denormalize_fn=None):
        operations = []    

        # the latter summand might seem odd, but it's clear that it uses a consistent order.
        operations += [self.optimize_operation, self.loss_fn] + self.summarizer.ops
        results = session.run(operations)

        return self.summarizer.step(results[1:])

class DiscriminativeTrainingIteration(object):
    def __init__(self, load_fn, train_fn, summarizer, denormalize_fn=None):
        self.load_fn = load_fn
        self.train_fn = train_fn
        self.summarizer = summarizer
        self.denormalize_fn = denormalize_fn
        self.batch_number = 0

    def __call__(self, session, ctx, retain_batch_n_times=1):
        # the "number steps" might be unusual, but there can be additional
        # stochasticity introduced by auxiliary processors.
        self.load_fn(session, shuffle=True)
        ctx.batch_loaded_info(self.batch_number, self.load_fn.loader.input, self.denormalize_fn)

        steps = []
        step_number = 0
        while step_number < retain_batch_n_times:
            step = self.train_fn(session, ctx, step_number + self.batch_number * retain_batch_n_times, self.denormalize_fn)
            steps.append(step)
            
            step_number += 1

        self.batch_number += 1
        return self.summarizer.reduce(steps)

class DiscriminativeTrainingEpoch(object):
    def __init__(self, iteration_fn):
        self.iteration_fn = iteration_fn

    def __call__(self, session, number_batches, ctx, retain_batch_n_times=1):
        iterations = []

        for batch_number in range(number_batches):
            # this will load a new batch.
            iteration = self.iteration_fn(session, ctx, retain_batch_n_times)
            ctx.batch_training_summary(batch_number, iteration)
            
            iterations.append(iteration)

        return iterations

class DiscriminativeEvaluationIteration(object):
    def __init__(self, load_fn, loss_fn, summarizer, denormalize_fn=None):
        self.load_fn = load_fn
        self.loss_fn = loss_fn
        self.summarizer = summarizer
        self.denormalize_fn = denormalize_fn
        self.batch_number = 0

    def __call__(self, session, ctx, retain_batch_n_times=1):
        self.load_fn(session, shuffle=False)
        ctx.batch_loaded_info(self.batch_number, self.load_fn.loader.input, self.denormalize_fn)

        steps = []
        step_number = 0
        while step_number < retain_batch_n_times:
            fetches = session.run([self.loss_fn] + self.summarizer.ops)
            step = self.summarizer.step(fetches)
            steps.append(step)
            step_number += 1

        self.batch_number += 1

        return self.summarizer.reduce(steps)
    
# this should really take a validation set at some point.
class DiscriminativeTrainingEvaluation(object):
    def __init__(self, iteration_fn):
        self.iteration_fn = iteration_fn

    def __call__(self, session, number_batches, ctx, retain_batch_n_times=1):
        iterations = []

        for batch_number in range(number_batches):
            iteration = self.iteration_fn(session, ctx, retain_batch_n_times)
            ctx.batch_testing_summary(batch_number, iteration)

            iterations.append(iteration)

        return iterations

class DiscriminativeSynthesisStep(object):
    def __init__(self, synthesis_fn, loss_fn, clip_fn=None):
        self.synthesis_fn = synthesis_fn
        self.loss_fn = loss_fn
        self.clip_fn = clip_fn

    def __call__(self, session):
        _, current_loss = session.run([self.synthesis_fn, self.loss_fn])
        # unsure if we can combine this with the previous run.
        if not (self.clip_fn is None):
            session.run(self.clip_fn)

        return SynthesisStepResult(loss=current_loss)

class LangevinSynthesisStep(object):
    def __init__(self, synthesis_fn, loss_fn, noise_fn):
        self.synthesis_fn = synthesis_fn
        self.loss_fn = loss_fn
        self.noise_fn = noise_fn

    def __call__(self, session):
        session.run(self.synthesis_fn)
        _, current_loss = session.run([self.noise_fn, self.loss_fn])

        return SynthesisStepResult(loss=current_loss)
        
class DiscriminativeSynthesisBatch(object):
    def __init__(self, load_batch_fn, synthesis_step_fn, retrieve_batch_fn, denormalize_fn=None, debug=False):
        self.load_batch_fn = load_batch_fn
        self.synthesis_step_fn = synthesis_step_fn
        self.retrieve_batch_fn = retrieve_batch_fn
        self.denormalize_fn = denormalize_fn
        self.debug = debug

    def __call__(self, session, batch_number, number_steps, ctx, early_stopping=None):
        self.load_batch_fn(session)

        for step_number in range(number_steps):
            step_result = self.synthesis_step_fn(session)

            if (step_number % 10) == 0:
                print "{0} loss at step {1}".format(step_result.loss, step_number)

            ctx.step_stats(step_number, step_result)                
            if early_stopping and early_stopping(step_result.loss):
                print "stopping synthesis early at step {0} and loss {1}".format(step_number, step_result.loss)
                ctx.stopping_early(step_number, step_result)
                break

        images, labels, names = self.retrieve_batch_fn(session)
        ctx.synthesized_info(images, labels, names, self.denormalize_fn)

        return SynthesisBatchResult(images=images, labels=labels, names=names)
