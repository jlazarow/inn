import bdb
import gc
import numpy as np
import os
import resource
import tensorflow as tf

from inn.batch import *
from inn.context import *
from inn.discriminator import Discriminator
from inn.process import *
from inn.source import TrainingDataHistory, TrainingDataClass
from inn.task import GraphBasedTask
from inn.util import ensure_shape_3d, shape_as_string, managed_or_debug_session, evenly_divided_by

NUMBER_SNAPSHOTS = 3

class SynthesizeDiscriminatively(GraphBasedTask):
    MAIN_SCOPE_NAME = "synthesize_discriminatively"
    
    def __init__(self, class_definitions, number_synthesis_per_class, network,
                 batch_size, batch_image_size, create_synthesis_fn, experiment,
                 auxiliary_processors=None, config_proto=None, graph=None, device_string=None):
        """
        Synthesize a sample from a sequence of classifiers learned.

        """
        super(SynthesizeDiscriminatively, self).__init__(
            experiment.run_path, config_proto, graph, device_string)

        # the base network parameters: what to learn and on which architecture to learn it.
        self.network = network
        self.batch_size = batch_size
        self.batch_image_size = ensure_shape_3d(batch_image_size)
        self.experiment = experiment
        self.offline_file = experiment.file
        self.auxiliary_processors = auxiliary_processors or []
        self.auxiliary_inputs = []

        self.class_definitions = class_definitions
        self.number_synthesis_per_class = number_synthesis_per_class
        self.classes = self.create_synthesis_classes(self.class_definitions, self.number_synthesis_per_class)
                
        with tf.name_scope(SynthesizeDiscriminatively.MAIN_SCOPE_NAME):
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

            self.synthesize = create_synthesis_fn(self.discriminator, self.input)

    @property
    def batch_shape(self):
        return (self.batch_size,) + self.batch_image_size

    def load_checkpoints(self, checkpoints_path):
        checkpoints = tf.train.get_checkpoint_state(checkpoints_path)
        if checkpoints is None:
            raise ValueError("no checkpoints found at {0}".format(checkpoints_path))

        return list(checkpoints.all_model_checkpoint_paths)

    def create_input(self):
        return BatchedInput(self.batch_size, self.batch_image_size)

    def create_synthesis_for_class(self, class_definition, number_synthesis):
        return class_definition.synthesis_initializer_fn(number_synthesis)

    def create_synthesis_class(self, class_definition, number_synthesis):
        print "reading (initial) synthesis of class {0} ({1})".format(class_definition.name, class_definition.label)

        initialized_synthesis = self.create_synthesis_for_class(class_definition, number_synthesis)
        # we set these up with the _desired_ class label.
        synthesis = TrainingDataHistory(
            batch=InputBatch(
                images=initialized_synthesis.images,
                names=initialized_synthesis.names,
                labels=set_of_labels(class_definition.label, initialized_synthesis.number_images)),
            name="{0}-synthesis".format(class_definition.label))
        training_class = TrainingDataClass(
            positives=None,
            negatives=None,
            synthesis=synthesis,
            probability=class_definition.prior_probability,
            positive_label=class_definition.label,
            negative_label=None)

        return training_class
        
    def create_synthesis_classes(self, class_definitions, number_synthesis_per_class):
        synthesis_classes = []
        
        for definition_index, definition in enumerate(class_definitions):
            synthesis_class = self.create_synthesis_class(definition, number_synthesis_per_class[definition_index])
            synthesis_classes.append(synthesis_class)

        return synthesis_classes

    def create_discriminator(self, network, images, labels):
        return Discriminator(network, images, labels)

    def create_batch_loader(self, input):
        return BatchLoader(input)

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

    def create_synthesis_batch_producer(self, ctx, total_batch=None):
       return SynthesisBatchProducer(total_batch, self.classes, ctx=ctx)

    def create_batch_loading_fn(self, producer, loader, batch_size):
        return ProduceAndLoadBatch(producer, loader, batch_size, self.batch_image_size)

    def clip_batch(self, batch):
        return InputBatch(np.clip(batch.images, -1.0, 1.0), batch.names, batch.labels)

    def __call__(self, session, model_path, number_synthesis_epochs,
                 synthesis_early_stopping_fn, synthesis_schedule_fn=None, debug=True):
        super(SynthesizeDiscriminatively, self).__call__(session)

        checkpoints_path = os.path.join(model_path, "checkpoints")
        model_data_path = os.path.join(model_path, "data")
        
        checkpoints = self.load_checkpoints(checkpoints_path)
        if not (synthesis_schedule_fn is None):
            checkpoints = synthesis_schedule_fn(checkpoints)

        # everything should be done by now.
        self.graph.finalize()

        synthesis_batches = None
        with SynthesisProcessContext(session, self.offline_file, debug) as process_ctx:
            for round_number in range(len(checkpoints)):
                try:
                    self.saver.restore(session, checkpoints[round_number])
                    
                    synthesis_batches = self.create_or_resize_synthesis_batches(synthesis_batches)
                    print "at round {0}, we're using {1} MB".format(
                        round_number,
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)

                    with process_ctx.round_ctx(round_number) as round_ctx:
                        with round_ctx.synthesis_ctx() as synthesis_ctx:
                            synthesis_batch_producer = self.create_synthesis_batch_producer(synthesis_ctx, synthesis_batches)
                            if not synthesis_batch_producer.evenly_divided_by(self.batch_size):
                                raise ValueError("cannot evenly divide total batches {0} by {1}".format(
                                    synthesis_batch_producer.size, self.batch_size))

                            number_synthesis_batches = synthesis_batch_producer.number_batches_for_size(
                                self.batch_size)
                            synthesis_batch_loading_fn = self.create_batch_loading_fn(
                                synthesis_batch_producer, self.batch_loader, self.batch_size)

                            synthesized_batch = self.synthesize(
                                synthesis_batch_loading_fn, synthesis_early_stopping_fn, number_synthesis_batches,
                                number_synthesis_epochs, session, synthesis_ctx)
                                
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
                                                           
                                batch_for_class = synthesized_batch[examples_start:examples_end]
                                for snapshot_index in range(min(NUMBER_SNAPSHOTS, self.number_synthesis_per_class[class_index])):
                                    snapshot_image, snapshot_name = batch_for_class.images[snapshot_index], batch_for_class.names[snapshot_index]

                                    # figure out how to get denormalize_fn in here.
                                    self.experiment.add_snapshot(
                                        snapshot_name,
                                        (snapshot_image + 1.0) * 127.5)

                                class_at_index.add(batch_for_class, "round" + str(round_number + 1))

                            del synthesis_batch_loading_fn
                            del synthesis_batch_producer
                            del synthesized_batch                    
                except (KeyboardInterrupt, bdb.BdbQuit):
                    print "Synthesis stopping on account of CTRL+C"
                    break

                # for good measure.
                gc.collect()

        # add ourselves to the given learned model.
        try:
            learned_offline_file = OfflineRunDataFile(os.path.join(model_data_path, "index.h5"), mode="r+")
            learned_offline_file.syntheses.add(
                self.experiment.run_name, os.path.join(self.experiment.data_path, "index.h5"))
            learned_offline_file.close()
        except:
            print "warning: unable to add reference to synthesis to model (this won't affect anything), probably already open."

        return self.classes
