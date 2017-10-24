import datetime
import pdb
import os

from inn.offline import *
from inn.profiler import Profiler

class Context(object):
    def __init__(self, session, debug):
        self.session = session
        self.debug = debug
        self.profiler = Profiler()
        
    def __enter__(self):
        self.profiler.__enter__()

        self.start()

        return self

    def start(self):
        pass

    def end(self):
        pass

    def __exit__(self, type, value, tb):
        self.profiler.__exit__(type, value, tb)

        self.end()

class LearningProcessContext(Context):
    def __init__(self, session, offline_file, debug=False):
        super(LearningProcessContext, self).__init__(session, debug)
        self.offline_file = offline_file

    def classes_info(self, classes, before_round_number):
        pass

    def round_ctx(self, round_number, learning_desc, evaluation_desc, synthesis_desc):
        offline_round = self.offline_file.rounds.add(learning_desc, evaluation_desc, synthesis_desc)
        
        return RoundContext(round_number, self.session, offline_round,  self.debug)

    def start(self):
        print "(learning) process started"

    def end(self):
        print "process finished after {0} seconds".format(
            self.profiler.elapsed)

class SynthesisProcessContext(Context):
    def __init__(self, session, offline_file, debug=False):
        super(SynthesisProcessContext, self).__init__(session, debug)
        self.offline_file = offline_file

    def round_ctx(self, round_number):
        offline_round = self.offline_file.rounds.add()
        
        return RoundContext(round_number, self.session, offline_round, self.debug)

    def start(self):
        print "(synthesis) process started"

    def end(self):
        print "process finished after {0} seconds".format(
            self.profiler.elapsed)
        
class RoundContext(Context):
    def __init__(self, round_number, session, offline_round, debug=False):
        super(RoundContext, self).__init__(session, debug)

        self.round_number = round_number
        self.offline_round = offline_round

    def learning_ctx(self):
        return LearningContext(self, self.session, self.offline_round.learning, self.debug)

    def synthesis_ctx(self):
        return SynthesisContext(self, self.session, self.offline_round.synthesis, self.debug)

    def start(self):
        print "starting round {0}".format(self.round_number)
    
    def end(self):
        print "round {0} finished after {1} seconds".format(
            self.round_number,
            self.profiler.elapsed)

        self.offline_round.close()

class LearningEpochContext(Context):
    def __init__(self, learning_ctx, epoch_number, session, offline_epoch, debug=False):
        super(LearningEpochContext, self).__init__(session, debug)

        self.learning_ctx = learning_ctx
        self.round_number = self.learning_ctx.round_number
        self.epoch_number = epoch_number
        self.offline_epoch = offline_epoch

    def start(self):
        # do nothing to reduce some chatter.
        pass

    def batch_training_summary(self, batch_number, summary):
        # we can throttle this off if desired.
        if not self.debug:
            return

        self.offline_epoch.batch_summary(batch_number, summary)

    def batch_age_counts(self, counts):
        if not self.debug:
            return
        
        self.offline_epoch.age_counts.add(counts)
        
    def batch_producer_info(self, producer):
        if not self.debug:
            return

    def batch_loaded_info(self, batch_number, input, denormalize_fn=None):
        if not self.debug:
            return

        # ask for the current batch.
        batch = input(self.session)
        batch_images = batch.images

        if not (denormalize_fn is None):
            batch_images = denormalize_fn(batch_images)

        self.offline_epoch.training_batch(batch_images, batch.labels, batch.names)

    def auxiliary_input_info(self, step, auxiliary_inputs, denormalize_fn=None):
        return
        
        # for now, only consider the first.
        auxiliary_input = auxiliary_inputs[0]
        if not (denormalize_fn is None):
            auxiliary_images = denormalize_fn(auxiliary_input.images)
        
        self.offline_epoch.auxiliary_batch(auxiliary_images, auxiliary_input.labels, auxiliary_input.names)

    def end(self):
        print "epoch {0} finished after {1} seconds".format(
            self.epoch_number,
            self.profiler.elapsed)

class LearningEvaluationContext(Context):
    def __init__(self, learning_ctx, evaluation_number, session, offline_evaluation, debug=False):
        super(LearningEvaluationContext, self).__init__(session, debug)

        self.learning_ctx = learning_ctx
        self.round_number = self.learning_ctx.round_number
        self.evaluation_number = evaluation_number
        self.offline_evaluation = offline_evaluation

    def start(self):
        print "starting evaluation {0}".format(self.evaluation_number)

    def batch_testing_summary(self, batch_number, summary):
        if not self.debug:
            return
        
        self.offline_evaluation.batch_summary(batch_number, summary)

    def batch_age_counts(self, counts):
        if not self.debug:
            return
        
        self.offline_evaluation.age_counts.add(counts)        

    def batch_producer_info(self, producer):
        if not self.debug:
            return

    def batch_loaded_info(self, batch_number, input, denormalize_fn=None):
        if not self.debug:
            return

        # ask for the current batch.
        batch = input(self.session)
        batch_images = batch.images

        if not (denormalize_fn is None):
            batch_images = denormalize_fn(batch_images)

        self.offline_evaluation.training_batch(batch_images, batch.labels, batch.names)        
        
    def end(self):
        print "evaluation {0} finished after {1} seconds".format(
            self.evaluation_number,
            self.profiler.elapsed)

class LearningContext(Context):        
    def __init__(self, round_ctx, session, offline_learning, debug=False):
        super(LearningContext, self).__init__(session, debug)

        self.round_ctx = round_ctx
        self.round_number = self.round_ctx.round_number
        self.offline_learning = offline_learning

    def epoch_ctx(self, epoch_number):
        if len(self.offline_learning.epochs) > epoch_number:
            offline_epoch = self.offline_learning.epochs[epoch_number]
        else:
            offline_epoch = self.offline_learning.epochs.add()
            
        return LearningEpochContext(self, epoch_number, self.session, offline_epoch, self.debug)

    def evaluation_ctx(self, evaluation_number):
        if len(self.offline_learning.evaluations) > evaluation_number:
            offline_evaluation = self.offline_learning.evaluations[evaluation_number]
        else:
            offline_evaluation = self.offline_learning.evaluations.add()
            
        return LearningEvaluationContext(self, evaluation_number, self.session, offline_evaluation, self.debug)

    def start(self):
        print "learning started"

    def epoch_training_summary(self, epoch_number, summary):
        print "trained at {0} loss".format(summary.loss)
        self.offline_learning.training_summary(epoch_number, summary)

    def epoch_evaluation_summary(self, evaluation_number, summary):
        print "evaluated at {0} loss".format(summary.loss)
        self.offline_learning.evaluation_summary(evaluation_number, summary)
        
    def stopping_early(self, epoch_number, stopping_result):
        print "stopping training early (epoch {0} and accuracy {1})".format(
            epoch_number, stopping_result.accuracy)
        self.offline_learning.stopped_early(epoch_number, stopping_result.accuracy)

    def end(self):
        print "learning finished after {0} seconds".format(
            self.profiler.elapsed)

class SynthesisBatchContext(Context):
    def __init__(self, synthesis_ctx, batch_number, session, offline_batch, debug=False):
        super(SynthesisBatchContext, self).__init__(session, debug)

        self.synthesis_ctx = synthesis_ctx
        self.round_number = self.synthesis_ctx.round_number
        self.batch_number = batch_number
        self.offline_batch = offline_batch

    def start(self):
        print "synthesizing batch {0}".format(self.batch_number)

    def step_stats(self, step_number, step_result):
        self.offline_batch.stats.add(step_number, step_result.loss)

    def synthesized_info(self, images, labels, names, denormalize_fn=None):
        if not (denormalize_fn is None):
            images = denormalize_fn(images)
        
        self.offline_batch.synthesized_batch(images, labels, names)

    def stopping_early(self, step_number, stopping_result):
        print "stopping synthesis of batch {0} early (step {1} and loss {2})".format(
            self.batch_number, step_number, stopping_result.loss)

        self.offline_batch.stopped_early(step_number, stopping_result.loss)
        
    def end(self):
        print "synthesis of batch {0} finished after {1} seconds".format(
            self.batch_number,
            self.profiler.elapsed)

class SynthesisContext(Context):
    def __init__(self, round_ctx, session, offline_synthesis,  debug=False):
        super(SynthesisContext, self).__init__(session, debug)

        self.round_ctx = round_ctx
        self.round_number = self.round_ctx.round_number
        self.offline_synthesis = offline_synthesis

    def batch_ctx(self, batch_number):
        if len(self.offline_synthesis.batches) > batch_number:
            batch = self.offline_synthesis.batches[batch_number]
        else:
            batch = self.offline_synthesis.batches.add()
            
        return SynthesisBatchContext(self, batch_number, self.session, batch, self.debug)

    def start(self):
        print "synthesis started"

    def synthesized_info(self, images, labels, names):
        pass

    def end(self):
        print "entire synthesis finished after {0} seconds".format(
            self.profiler.elapsed)
