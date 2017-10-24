import tensorflow as tf

from inn.process import *

class LearningEpochStage(object):
    DEFAULT_GRADIENT_DESCENT_RATE = 0.01
    DEFAULT_ADAM_LEARNING_RATE = 0.0001
    DEFAULT_ADAM_BETA1 = 0.5    
    
    def __init__(self, discriminator, summarizer, optimizer_fn=None, auxiliary_inputs=None, denormalize_fn=None):
        self.discriminator = discriminator
        self.summarizer = summarizer
        self.optimizer_fn = optimizer_fn
        self.auxiliary_inputs = auxiliary_inputs
        self.denormalize_fn = denormalize_fn

        self.iteration_number = tf.Variable(0, name="training_iteration_number", trainable=False)
        self.reset_iteration_number = self.iteration_number.assign(0)

        self.iterations_per_epoch = tf.Variable(0, name="iterations_per_epoch", trainable=False)
        self.iterations_per_epoch_placeholder = tf.placeholder(tf.int32, shape=())
        self.assign_iterations_per_epoch = self.iterations_per_epoch.assign(
            self.iterations_per_epoch_placeholder)

        self.training_step_fn = self.create_training_step_fn()

    def create_optimizer(self):
        if not (self.optimizer_fn is None):
            return self.optimizer_fn(self.iteration_number, self.iterations_per_epoch)
        
        return tf.train.GradientDescentOptimizer(learning_rate=LearningEpochStage.DEFAULT_GRADIENT_DESCENT_RATE)

    def create_training_step_fn(self):
        optimizer = self.create_optimizer()

        return DiscriminativeTrainingStep(
            self.discriminator.training_loss,
            optimizer,
            self.discriminator.network_variables,
            self.iteration_number,
            self.summarizer)

    def create_training_iteration_fn(self, batch_loading_fn, training_step_fn):
        return DiscriminativeTrainingIteration(
            load_fn=batch_loading_fn,
            train_fn=training_step_fn,
            summarizer=self.summarizer,
            denormalize_fn=self.denormalize_fn)

    def create_training_epoch_fn(self, training_iteration_fn):
        return DiscriminativeTrainingEpoch(iteration_fn=training_iteration_fn)

    def __call__(self, batch_loading_fn, number_iterations, session, ctx, retain_batch_n_times=1):
        session.run(self.assign_iterations_per_epoch,
                    feed_dict={ self.iterations_per_epoch_placeholder: (number_iterations * retain_batch_n_times)  })
        
        training_iteration_fn = self.create_training_iteration_fn(
            batch_loading_fn,
            self.training_step_fn)
        training_epoch_fn = self.create_training_epoch_fn(
            training_iteration_fn)
        
        return training_epoch_fn(session, number_iterations, ctx, retain_batch_n_times)

class LearningEvaluationStage(object):
    def __init__(self, discriminator, summarizer, auxiliary_inputs=None, denormalize_fn=None):
        self.discriminator = discriminator
        self.summarizer = summarizer
        self.auxiliary_inputs = auxiliary_inputs
        self.denormalize_fn = denormalize_fn        

    def create_evaluation_iteration_fn(self, batch_loading_fn, evaluation_step_fn):
        return DiscriminativeEvaluationIteration(
            batch_loading_fn,
            evaluation_step_fn,
            self.summarizer,
            self.denormalize_fn)

    def create_evaluation_epoch_fn(self, evaluation_iteration_fn):
        return DiscriminativeTrainingEvaluation(iteration_fn=evaluation_iteration_fn)
        
    def __call__(self, batch_loading_fn, number_batches_per_evaluation, session, ctx):
        evaluate_iteration_fn = self.create_evaluation_iteration_fn(
            batch_loading_fn,
            self.discriminator.testing_loss)
        evaluate_epoch_fn = self.create_evaluation_epoch_fn(
            evaluate_iteration_fn)

        return evaluate_epoch_fn(session, number_batches_per_evaluation, ctx)            

class SynthesisStage(object):
    DEFAULT_ADAM_LEARNING_RATE = 0.005
    DEFAULT_ADAM_BETA1 = 0.5
    
    def __init__(self, discriminator, input, optimizer_fn=None, denormalize_fn=None, debug=True):
        self.discriminator = discriminator
        self.input = input
        self.optimizer_fn = optimizer_fn
        self.denormalize_fn = denormalize_fn
        self.debug = debug
        self.iteration_number = tf.Variable(0, name="synthesis_iteration_number", trainable=False)
        self.reset_iteration_number = self.iteration_number.assign(0)

        self.synthesize_step_fn = self.create_synthesize_step_fn()
        self.retrieve_batch_fn = self.create_retrieve_batch_fn()

    def create_optimizer(self):
        if not (self.optimizer_fn is None):
            return self.optimizer_fn(self.iteration_number)
        
        return tf.train.AdamOptimizer(
            learning_rate=SynthesisStage.DEFAULT_ADAM_LEARNING_RATE, beta1=SynthesisStage.DEFAULT_ADAM_BETA1)

    def create_synthesize_step_fn(self):
        with tf.variable_scope("sampling_optimizer") as optimizer_scope:
            optimizer = self.create_optimizer()
            optimize_variables = [self.input.images]
            optimize_gradients = optimizer.compute_gradients(
                self.discriminator.sampling_loss, var_list=optimize_variables)
            synthesis_fn = optimizer.apply_gradients(
                optimize_gradients, global_step=self.iteration_number)

            optimizer_own_variables = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if optimizer_scope.name in v.name]
            if len(optimizer_own_variables) > 0:
                print "synthesis optimizer has {0} variables. these will be reset on each batch.".format(len(optimizer_own_variables))

            self.reinitialize_optimizer = tf.variables_initializer(
                optimizer_own_variables,
                name="reinitialize_sampling_optimizer")

        return DiscriminativeSynthesisStep(
            synthesis_fn=synthesis_fn,
            loss_fn=self.discriminator.sampling_loss,
            clip_fn=self.input.images.assign(tf.clip_by_value(self.input.images, -1.0, 1.0)))

    def create_retrieve_batch_fn(self):
        return lambda session: session.run([self.input.images, self.input.labels, self.input.names])

    def create_synthesize_batch_fn(self, batch_loading_fn, synthesize_step_fn, retrieve_batch_fn, denormalize_fn=None, debug=False):
        return DiscriminativeSynthesisBatch(
            batch_loading_fn,
            synthesize_step_fn,
            retrieve_batch_fn,
            denormalize_fn,
            debug)

    def __call__(self, batch_loading_fn, early_stopping_fn, number_batches, number_epochs, session, ctx):
        images = []
        labels = []
        names = []

        synthesize_batch_fn = self.create_synthesize_batch_fn(
            batch_loading_fn,
            self.synthesize_step_fn,
            self.retrieve_batch_fn,
            self.denormalize_fn,
            self.debug)

        for batch_number in range(number_batches):
            session.run([self.reset_iteration_number, self.reinitialize_optimizer])

            synthesized_batch = None
            early_stopping = early_stopping_fn()

            with ctx.batch_ctx(batch_number) as batch_ctx:
                try:
                    synthesized_batch = synthesize_batch_fn(
                        session, batch_number, number_epochs, batch_ctx, early_stopping)
                    
                    images.append(synthesized_batch.images)
                    labels.append(synthesized_batch.labels)
                    names.append(synthesized_batch.names)
                except IndexError:
                    pass                                            

        # combine all the synthesized images.
        images = np.concatenate(images)
        labels = np.concatenate(labels)
        names = np.concatenate(names)

        return InputBatch(images, names, labels)
