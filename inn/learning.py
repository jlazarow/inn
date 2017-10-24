import bdb
import gc
import math
import numpy as np
import os
import resource
import tensorflow as tf

from inn.batch import *
from inn.context import *
from inn.discriminator import Discriminator
from inn.offline import LearningSummaryDescription, EvaluationSummaryDescription, SynthesisSummaryDescription
from inn.process import *
from inn.source import TrainingDataHistory, TrainingDataClass
from inn.task import GraphBasedTask
from inn.util import ensure_shape_3d, shape_as_string, managed_or_debug_session, evenly_divided_by

NUMBER_SNAPSHOTS = 3

class LearnDiscriminatively(GraphBasedTask):
    MAIN_SCOPE_NAME = "learn_discriminatively"
    
    def __init__(self, class_definitions, network, batch_size, batch_image_size,
                 create_learn_epoch_fn, create_evaluate_epoch_fn, create_synthesis_fn,
                 experiment, negative_labels=None, auxiliary_processors=None,
                 config_proto=None, graph=None, device_string=None):
        """
        Create a learner of a distribution using discriminative methods.

        create_learn_epoch_fn: a function that takes a Discriminator and returns a callable capable of learning
                               an epoch with arguments of the form: "batch_loading_fn, number_iterations, tf.Session, ctx"

        """
        super(LearnDiscriminatively, self).__init__(
            experiment.run_path, config_proto, graph, device_string)

        # the base network parameters: what to learn and on which architecture to learn it.
        self.network = network
        self.batch_size = batch_size
        self.batch_image_size = ensure_shape_3d(batch_image_size)
        self.experiment = experiment
        self.offline_file = self.experiment.file
        self.negative_labels = negative_labels
        self.auxiliary_processors = auxiliary_processors or []
        self.auxiliary_inputs = []

        self.class_definitions = class_definitions
        self.classes = self.create_training_classes(self.class_definitions, self.negative_labels)
                
        with tf.name_scope(LearnDiscriminatively.MAIN_SCOPE_NAME):
            self.global_step = tf.train.create_global_step(graph=self.graph)
            
            self.input = self.create_input()

            auxiliary_images = self.input.images
            auxiliary_labels = self.input.labels

            # we go through the auxiliary processors in the given order. this means
            # the innermost part of the composition is the first element, etc.
            for auxiliary_processor in self.auxiliary_processors:
                auxiliary_images, auxiliary_labels, auxiliary_names = auxiliary_processor(auxiliary_images, auxiliary_labels)

                self.auxiliary_inputs.append(
                    AuxiliaryInput(images=auxiliary_images, labels=auxiliary_labels, names=auxiliary_names))
            
            self.discriminator = self.create_discriminator(self.network, auxiliary_images, auxiliary_labels)

            with tf.name_scope("input_loader"):
                self.batch_loader = self.create_batch_loader(self.input)

            self.scaffold = tf.train.Scaffold(                    
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.local_variables_initializer())
            self.saver = self.discriminator.saver()

            self.learn_epoch = create_learn_epoch_fn(self.discriminator, self.auxiliary_inputs)
            self.evaluate_epoch = create_evaluate_epoch_fn(self.discriminator, self.auxiliary_inputs)
            self.synthesize = create_synthesis_fn(self.discriminator, self.input)

    @property
    def batch_shape(self):
        return (self.batch_size,) + self.batch_image_size

    def create_input(self):
        return BatchedInput(self.batch_size, self.batch_image_size)

    def create_positives_for_class(self, class_definition):
        return class_definition.positives_initializer_fn()

    def create_negatives_for_class(self, class_definition, number_negatives):
        return class_definition.negatives_initializer_fn(number_negatives)

    def create_synthesis_for_class(self, class_definition, number_negatives):
        return class_definition.synthesis_initializer_fn(number_negatives)    

    def create_training_class(self, class_definition, negative_label):
        print "reading positives of class {0} ({1})".format(class_definition.name, class_definition.label)

        initialized_positives = self.create_positives_for_class(class_definition)
        positives = TrainingDataHistory(
            batch=InputBatch(
                images=initialized_positives.images,
                names=initialized_positives.names,
                labels=set_of_labels(class_definition.label, initialized_positives.number_images)),
            name="{0}-positives".format(class_definition.label))

        # match the number of positives for now.
        print "reading negatives of class {0} ({1})".format(class_definition.name, negative_label)
        initialized_negatives = self.create_negatives_for_class(class_definition, len(positives))
        negatives = TrainingDataHistory(
            batch=InputBatch(
                images=initialized_negatives.images,
                names=initialized_negatives.names,
                labels=set_of_labels(negative_label, initialized_negatives.number_images)),
            name="{0}-negatives".format(class_definition.label))

        if not (class_definition.synthesis_initializer_fn is None):
            initialized_synthesis = self.create_synthesis_for_class(class_definition, len(positives))
            synthesis = TrainingDataHistory(
                batch=InputBatch(
                    images=initialized_synthesis.images,
                    names=initialized_synthesis.names,
                    labels=set_of_labels(class_definition.label, initialized_synthesis.number_images)),
                name="{0}-synthesis".format(class_definition.label))
        else:
            synthesis = TrainingDataHistory(
                batch=negatives.total_batch.relabel(class_definition.label).suffixed("round0"),
                name="{0}-synthesis".format(class_definition.label))

        # note that the synthesis carries the true positive label.
        # this will be modified once a synthesis is added to the negatives.
        training_class = TrainingDataClass(
            positives=positives,
            negatives=negatives,
            synthesis=synthesis,
            probability=class_definition.prior_probability,
            positive_label=class_definition.label,
            negative_label=negative_label)

        return training_class
        
    def create_training_classes(self, class_definitions, negative_labels=None):
        if negative_labels is None:
            negative_labels = len(class_definitions) * [len(class_definitions)]

        if len(negative_labels) != len(class_definitions):
            raise ValueError("expected negative labels and class definitions to be of equal size")
            
        training_classes = []
        
        for definition_index, definition in enumerate(class_definitions):
            training_class = self.create_training_class(definition, negative_labels[definition_index])
            training_classes.append(training_class)

        return training_classes

    def create_discriminator(self, network, images, labels):
        return Discriminator(network, images, labels)

    def create_batch_loader(self, input):
        return BatchLoader(input)

    def create_or_resize_training_batches(self, training_batches, maximum_learning_batches):
        training_batches_size = 0
        
        for training_class in self.classes:
            number_negatives = training_class.negatives.size

            # technically it's always the case that we add as many positives as we do negatives.
            number_examples_for_class = 2 * number_negatives
            training_batches_size += number_examples_for_class

        if not (maximum_learning_batches is None):
            training_batches_size = min(self.batch_size * maximum_learning_batches, training_batches_size)

        if training_batches is None:
            return InputBatch(
                images=np.zeros((training_batches_size,) + self.batch_image_size, dtype=np.float32),
                names=np.array(training_batches_size * ["placeholder"], dtype=object),
                labels=set_of_labels(-1, training_batches_size))

        # resize.
        training_batches.resize(training_batches_size)
        return training_batches

    def create_or_resize_evaluation_batches(self, evaluation_batches, number_evaluation_batches):
        number_images = number_evaluation_batches * self.batch_size
        if evaluation_batches is None:
            return InputBatch(
                images=np.zeros((number_images,) + self.batch_image_size, dtype=np.float32),
                names=np.array(number_images * ["placeholder"], dtype=object),
                labels=set_of_labels(-1, number_images))

        # resize.
        evaluation_batches.resize(number_images)
        return evaluation_batches
    
    def create_or_resize_synthesis_batches(self, synthesis_batches):
        if synthesis_batches is None:
            iteration_size = 0

            for synthesis_class in self.classes:
                number_synthesis = synthesis_class.synthesis.size
                number_examples_for_class = number_synthesis
                iteration_size += number_examples_for_class
            
                return InputBatch(
                    images=np.zeros((iteration_size,) + self.batch_image_size, dtype=np.float32),
                    names=np.array(iteration_size * ["placeholder"], dtype=object),
                    labels=set_of_labels(-1, iteration_size))

        # do nothing since the iteration size is fixed between rounds.
        return synthesis_batches    

    def create_training_batch_producer(self, positive_probability, shuffle, ctx, total_batch=None):
        return TrainingBatchProducer(
            total_batch, self.classes, positive_probability=positive_probability, ctx=ctx, shuffle=shuffle)

    def create_evaluation_batch_producer(self, positive_probability, shuffle, ctx, total_batch=None):
        return TrainingBatchProducer(
            total_batch, self.classes, positive_probability=positive_probability, ctx=ctx, shuffle=shuffle)   

    def create_synthesis_batch_producer(self, ctx, total_batch=None):
        return SynthesisBatchProducer(total_batch, self.classes, ctx=ctx)   
    
    def create_batch_loading_fn(self, producer, loader, batch_size):
        return ProduceAndLoadBatch(producer, loader, batch_size, self.batch_image_size)

    def __call__(self, session,
                 number_learning_epochs, learning_early_stopping_fn,
                 evaluate_every_n, number_evaluation_batches,
                 number_synthesis_epochs, synthesis_early_stopping_fn,
                 maximum_learning_batches=None, retain_batch_n_times=1, checkpoint_every_n=1,
                 collapse_every_n=1, debug=True):
        super(LearnDiscriminatively, self).__call__(session)

        # everything should be done by now.
        self.graph.finalize()

        reinitialize_collapse = False
        if not (collapse_every_n is None) and collapse_every_n > 1:
            checkpoint_every_n = collapse_every_n

        training_batches = None
        evaluation_batches = None
        synthesis_batches = None
        
        learning_summary_description = LearningSummaryDescription(
            number_epochs=number_learning_epochs,
            summarizer_type=self.learn_epoch.summarizer.Summary)
        
        with LearningProcessContext(session, self.offline_file, debug) as process_ctx:
            print "using {0} requiring {1:.1f}M parameters".format(
                self.network.name,
                self.network.count_number_parameters() / 1e6)
            round_number = 0

            # until the user cancels.            
            while True:
                try:
                    # this means never collapse.
                    if collapse_every_n is None:
                        collapse_number = round_number + 1
                    else:
                        collapse_number = (round_number % collapse_every_n) + 1
                        
                    training_batches = self.create_or_resize_training_batches(training_batches, maximum_learning_batches)
                    synthesis_batches = self.create_or_resize_synthesis_batches(synthesis_batches)

                    # add a new batch to evaluate each round.
                    evaluation_batches = self.create_or_resize_evaluation_batches(evaluation_batches, number_evaluation_batches)

                    print "using {0} MB".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
                    process_ctx.classes_info(self.classes, round_number)
                    session.run(self.learn_epoch.reset_iteration_number)

                    eval_summary_description = EvaluationSummaryDescription(
                        number_epochs=int(math.ceil(number_learning_epochs / evaluate_every_n)),
                        summarizer_type=self.evaluate_epoch.summarizer.Summary)
                    synthesis_summary_description = SynthesisSummaryDescription(
                        number_batches=(np.shape(synthesis_batches.images)[0] / self.batch_size),
                        summarizer_type=None)

                    with process_ctx.round_ctx(
                            round_number, learning_summary_description, eval_summary_description, synthesis_summary_description) as round_ctx:
                        with round_ctx.learning_ctx() as learning_ctx: 
                            learning_early_stopping = learning_early_stopping_fn()
                            evaluation_number = 0

                            for epoch_number in range(number_learning_epochs):
                                with learning_ctx.epoch_ctx(epoch_number) as epoch_ctx:
                                    # generate our batches for this set.
                                    training_batch_producer = self.create_training_batch_producer(
                                        positive_probability=0.5, shuffle=True, ctx=epoch_ctx, total_batch=training_batches)
                                    number_batches_per_epoch = training_batch_producer.number_batches_for_size(
                                        self.batch_size)
                                    training_batch_loading_fn = self.create_batch_loading_fn(
                                        training_batch_producer, self.batch_loader, self.batch_size)

                                    epoch_summary = self.learn_epoch.summarizer.reduce(self.learn_epoch(
                                        training_batch_loading_fn, number_batches_per_epoch, session, epoch_ctx, retain_batch_n_times))
                                    learning_ctx.epoch_training_summary(epoch_number, epoch_summary)

                                is_final_epoch = epoch_number + 1 == number_learning_epochs
                                if not evenly_divided_by(epoch_number, evaluate_every_n) or is_final_epoch:
                                    continue
                                
                                # report test statistics for the epoch.
                                with learning_ctx.evaluation_ctx(evaluation_number) as evaluation_ctx:
                                    evaluation_batch_producer = self.create_evaluation_batch_producer(
                                        positive_probability=0.5, shuffle=True, ctx=evaluation_ctx, total_batch=evaluation_batches)
                                    evaluation_ctx.batch_producer_info(evaluation_batch_producer)
                                                
                                    evaluation_batch_loading_fn = self.create_batch_loading_fn(
                                        evaluation_batch_producer, self.batch_loader, self.batch_size)
                                    evaluation_summary = self.evaluate_epoch.summarizer.reduce(self.evaluate_epoch(
                                        evaluation_batch_loading_fn, number_evaluation_batches, session, evaluation_ctx))
                                    learning_ctx.epoch_evaluation_summary(evaluation_number, evaluation_summary)
                                    
                                    evaluation_number += 1

                                # also, perform an evaluation to determine early stopping.
                                if epoch_number > 1 and learning_early_stopping(evaluation_summary):
                                    learning_ctx.stopping_early(epoch_number, evaluation_summary)
                                    break

                            if evenly_divided_by(round_number + 1, checkpoint_every_n):
                                self.saver.save(
                                    session, os.path.join(self.experiment.checkpoints_path, "model.ckpt"), round_number)

                        with round_ctx.synthesis_ctx() as synthesis_ctx:
                            synthesis_batch_producer = self.create_synthesis_batch_producer(
                                synthesis_ctx, synthesis_batches)
                            if not synthesis_batch_producer.evenly_divided_by(self.batch_size):
                                raise ValueError("cannot evenly divide total batches {0} by {1}".format(
                                    synthesis_batch_producer.size, self.batch_size))

                            number_synthesis_batches = synthesis_batch_producer.number_batches_for_size(
                                self.batch_size)
                            synthesis_batch_loading_fn = self.create_batch_loading_fn(
                                synthesis_batch_producer, self.batch_loader, self.batch_size)
                            synthesized_batch = self.synthesize(
                                synthesis_batch_loading_fn, synthesis_early_stopping_fn, number_synthesis_batches,
                                number_synthesis_epochs * collapse_number, session, synthesis_ctx)
                                
                            # the results will be in the relatively correct order.
                            synthesis_start_indexes = synthesis_batch_producer.start_indexes
                            for class_index, start_index in enumerate(synthesis_start_indexes):
                                class_at_index = self.classes[class_index]
                                examples_start = start_index

                                # the last class should eat the remainder.
                                if (class_index + 1) == len(synthesis_start_indexes):
                                    examples_end = synthesized_batch.size
                                else:
                                    examples_end = synthesis_start_indexes[class_index + 1]
                                                           
                                # if we're collapsing this round, generate new starting synthesis examples.
                                next_synthesis = None
                                if (collapse_every_n is None) or (collapse_number != collapse_every_n):
                                    if reinitialize_collapse or (collapse_every_n is None):
                                        initialized_next_synthesis = self.create_negatives_for_class(
                                            self.class_definitions[class_index], examples_end - examples_start)
                                        next_synthesis = InputBatch(
                                            images=initialized_next_synthesis.images,
                                            names=initialized_next_synthesis.names,
                                            labels=set_of_labels(self.class_definitions[class_index].label,
                                                                 initialized_next_synthesis.number_images))
                                    else:
                                        next_synthesis = self.classes[class_index].negatives.batches[-collapse_number].relabel(
                                            self.classes[class_index].positive_label)
                                        
                                    next_synthesis = next_synthesis.suffixed("gen@round" + str(round_number + 1))

                                # put these back under the right label.
                                batch_for_class = synthesized_batch[examples_start:examples_end]
                                for snapshot_index in range(NUMBER_SNAPSHOTS):
                                    snapshot_image, snapshot_name = (batch_for_class.images[snapshot_index],
                                                                     batch_for_class.names[snapshot_index])

                                    # figure out how to get denormalize_fn in here.
                                    self.experiment.add_snapshot(
                                        snapshot_name,
                                        (snapshot_image + 1.0) * 127.5)

                                class_at_index.add(batch_for_class, "round" + str(round_number + 1), next_synthesis)
                except (KeyboardInterrupt, bdb.BdbQuit):
                    print "Training stopping on account of CTRL+C"
                    break

                round_number += 1

        return self.classes
                
