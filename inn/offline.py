import collections
import h5py
import numpy as np
import os
import pdb

from inn.summarizer import LossAndAccuracySummarizer

class HierarchicalObject(object):
    def __init__(self, parent):
        self.parent = parent
        self.root = None

    def flush(self):
        pass

class HierarchicalDataset(object):
    def __init__(self, parent, name, shape, dtype, compression=None, flatten=False):
        self.parent = parent
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.compression = compression
        self.flatten = flatten

        if self.name in self.parent.keys():
            self.root = self.parent[self.name]
        else:
            # replace None's in the given shape with zeros.
            corrected_shape = []
            max_shape = []
            for index, val in enumerate(self.shape):
                if val is None:
                    corrected_shape.append(0)
                    max_shape.append(None)
                else:
                    corrected_shape.append(val)
                    max_shape.append(val)

            self.shape = tuple(corrected_shape)
            self.root = self.parent.require_dataset(
                self.name, self.shape, dtype=self.dtype,
                maxshape=tuple(max_shape),
                compression=compression)

        self.size = self.root.shape[0]

    def __getitem__(self, key):        
        return self.root.__getitem__(key)

    def flush(self):
        self.root.flush()

    def refresh(self):
        self.root.id.refresh()

    def add(self, value):
        number_added = 1
        if isinstance(value, np.ndarray):
            # this is a tad tricky for images. we'll match up other dimensions that are none.
            if self.flatten:
                number_added = np.shape(value)[0]
                resize_shape = (self.size + number_added,) + np.shape(value)[1:]
                self.root.resize(resize_shape)
                self.root[self.size:(self.size + number_added)] = value                
            else:
                resize_shape = (self.size + 1,) + np.shape(value)
                self.root.resize(resize_shape)
                self.root[self.size] = value
        else:
            self.root.resize(self.size + 1, axis=0)
            self.root[self.size] = value

        # perhaps not the most performant.
        self.flush()
        
        self.size += number_added
        
class HierarchicalGroup(object):
    def __init__(self, parent):
        self.parent = parent
        self.root = None
        self.instances = []

    def __len__(self):
        return self.instances.__len__()

    def __getitem__(self, key):        
        return self.instances.__getitem__(key)

    def __iter__(self):
        return self.instances.__iter__()

    def get_sorted_keys(self):
        group_integer_keys = filter(lambda k: k.isdigit(), self.root.keys())
        return sorted(group_integer_keys, key=lambda k: int(k))

    def get_next_insertion_key(self):
        sorted_keys = self.get_sorted_keys()

        return "0" if not sorted_keys else str(int(sorted_keys[-1]) + 1)

    def flush(self):
        self.root.flush()

class SavedBatch(HierarchicalObject):
    def __init__(self, parent, key, flatten=False):
        super(SavedBatch, self).__init__(parent)
        self.root = self.parent.require_group(key)
        self.flatten = flatten
        self.read()

    def read(self):
        # assume RGB images.
        # should we store floats here? interesting choice.
        self.identifiers = HierarchicalDataset(
            self.root, "identifiers", (None,) if self.flatten else (None, None),
            dtype=np.uint32, compression="gzip", flatten=self.flatten)        
        self.images = HierarchicalDataset(
            self.root, "images", (None, None, None, None) if self.flatten else (None, None, None, None, None),
            dtype=np.uint8, compression="gzip", flatten=self.flatten)
        self.labels = HierarchicalDataset(
            self.root, "labels", (None,) if self.flatten else (None, None), dtype=np.uint8, flatten=self.flatten)
        self.names = HierarchicalDataset(
            self.root, "names", (None,) if self.flatten else (None, None), dtype=h5py.special_dtype(vlen=unicode),
            flatten=self.flatten)

    @property
    def present_size(self):
        return min(max(self.images.size, self.identifiers.size), self.labels.size, self.names.size)

    @property
    def batch_size(self):
        if self.present_size > 0:
            if self.identifiers.size > 0:
                return self.identifiers.root.shape[1]
            
            if self.images.size > 0:
                return self.images.root.shape[1]
            
        return 0

class SummaryStatistics(HierarchicalObject):
    def __init__(self, parent, key, type_name):
        super(SummaryStatistics, self).__init__(parent)

        self.root = self.parent.require_group(key)
        self.type_name = type_name

    def read(self):
        self.step = HierarchicalDataset(
            self.root, "step", (None,), dtype=np.int64)        

    def add(self, step_, summary):
        if not ("type" in self.root.attrs):
            self.root.attrs["type"] = self.type_name
        
        self.step.add(step_)
        self.step.flush()

    def refresh(self):
        self.step.refresh()
            
    @property
    def last_nonzero_step(self):
        return np.nonzero(self.step[:])[0][-1]

LEARNING_STATISTICS_TYPE_NAME = "LearningStatistics"    
class LearningStatistics(SummaryStatistics):
    def __init__(self, parent, key, type_name=LEARNING_STATISTICS_TYPE_NAME):
        super(LearningStatistics, self).__init__(parent, key, type_name)
        self.read()

    def read(self):
        super(LearningStatistics, self).read()
        
        self.loss = HierarchicalDataset(
            self.root, "loss", (None,), dtype=np.float32)
        self.accuracy = HierarchicalDataset(
            self.root, "accuracy", (None,), dtype=np.float32)

    def add(self, step_, summary):
        super(LearningStatistics, self).add(step_, summary)
        
        self.loss.add(summary.loss)
        self.accuracy.add(summary.accuracy)
        self.loss.flush()
        self.accuracy.flush()

    def refresh(self):
        super(LearningStatistics, self).refresh()

        self.loss.refresh()
        self.accuracy.refresh()

STATISTICS_SUMMARY_MAPPING = {
    LEARNING_STATISTICS_TYPE_NAME: LearningStatistics,
}

SUMMARIZER_SUMMARY_MAPPING = {
    "LossAndAccuracySummary": LearningStatistics,
}
        
class LearningEarlyStopping(HierarchicalObject):
    def __init__(self, parent):
        super(LearningEarlyStopping, self).__init__(parent)

        self.root = self.parent.require_group("early_stopping")

    @property
    def step(self):
        return self.root.attrs["step"]

    @step.setter
    def step(self, value):
        self.root.attrs["step"] = value

    @property
    def accuracy(self):
        return self.root.attrs["accuracy"]

    @accuracy.setter
    def accuracy(self, value):
        self.root.attrs["accuracy"] = value

    @property
    def has_data(self):
        return "step" in self.root.attrs

def summary_type_for_summarizer(summarizer):
    if summarizer is None:
        return None
    
    summary_type = SUMMARIZER_SUMMARY_MAPPING[summarizer.__name__]

    return summary_type

def read_summary_type(root, key, default_type=None):
    summary_type = default_type
    if summary_type is None:
        summary_type = LearningStatistics

    if key in root:
        stats_obj = root[key]

        if "type" in stats_obj.attrs:
            summary_type = STATISTICS_SUMMARY_MAPPING[stats_obj.attrs["type"]]

    return summary_type
         
class LearningEpoch(HierarchicalObject):
    def __init__(self, parent, store, number, summary_type=None):
        super(LearningEpoch, self).__init__(parent)

        self.store = store
        self.number = number
        self.summary_type = summary_type            
        self.root = self.parent.require_group(str(self.number))

        if self.summary_type is None:
            self.summary_type = read_summary_type(self.root, "stats", LearningStatistics)

        self.read()

    def batch_summary(self, step, summary):
        self.stats.add(step, summary)

    def training_batch(self, images, labels, names):
        # since this will grow, we'll store identifiers from our store instead (just in time).
        identifiers = np.array(self.store.store(images, labels, names), dtype=np.uint32)

        self.batch.identifiers.add(identifiers)
        self.batch.labels.add(labels)
        self.batch.names.add(names)

    def auxiliary_batch(self, images, labels, names):
        self.auxiliary.images.add(images)
        self.auxiliary.labels.add(labels)
        self.auxiliary.names.add(names)

    def read(self):
        self.stats = self.summary_type(self.root, "stats")
        self.batch = SavedBatch(self.root, "batch", flatten=False)
        self.age_counts = HierarchicalDataset(
            self.root, "age_counts", (None, None), dtype=np.int32)
        self.auxiliary = SavedBatch(self.root, "auxiliary", flatten=False)

class LearningEvaluations(HierarchicalGroup):
    def __init__(self, parent, store):
        super(LearningEvaluations, self).__init__(parent)

        self.store = store
        self.root = self.parent.require_group("evaluation")

        self.read()

    def read(self):
        evaluation_keys = self.get_sorted_keys()

        for evaluation_key in evaluation_keys:
            try:
                self.instances.append(LearningEpoch(self.root, self.store, int(evaluation_key)))
            except:
                pass

    def add(self, summary_type=None):
        evaluation_insertion_key = self.get_next_insertion_key()
        evaluation  = LearningEpoch(self.root, self.store, int(evaluation_insertion_key), summary_type)
        self.instances.append(evaluation)

        return evaluation
       
class LearningEpochs(HierarchicalGroup):
    def __init__(self, parent, store):
        super(LearningEpochs, self).__init__(parent)

        self.store = store
        self.root = self.parent.require_group("epoch")

        self.read()

    def read(self):
        epoch_keys = self.get_sorted_keys()

        for epoch_key in epoch_keys:
            try:
                # for now this might happen if we stopped early before this epoch.
                self.instances.append(LearningEpoch(self.root, self.store, int(epoch_key)))
            except:
                pass

    def add(self, summary_type=None):
        epoch_insertion_key = self.get_next_insertion_key()
        epoch = LearningEpoch(self.root, self.store, int(epoch_insertion_key), summary_type)
        self.instances.append(epoch)

        return epoch

class LearningStage(HierarchicalObject):
    def __init__(self, parent, store, learning_summary_type=None, eval_summary_type=None):
        super(LearningStage, self).__init__(parent)
        
        self.root = self.parent.require_group("learning")
        self.store = store
        self.learning_summary_type = learning_summary_type
        self.eval_summary_type = eval_summary_type

        if self.learning_summary_type is None:
            self.learning_summary_type = read_summary_type(self.root, "training_stats", LearningStatistics)

        if self.eval_summary_type is None:
            self.eval_summary_type = read_summary_type(self.root, "evaluation_stats", LearningStatistics)
        
        self.read()

    def stopped_early(self, step, accuracy):
        self.early_stopping.step = step
        self.early_stopping.accuracy = accuracy

    def training_summary(self, step, summary):
        self.training_stats.add(step, summary)

    def evaluation_summary(self, step, summary):
        self.evaluation_stats.add(step, summary)        
        
    def read(self):
        self.epochs = LearningEpochs(self.root, self.store)
        self.evaluations = LearningEvaluations(self.root, self.store)

        # averages over the entire round.
        self.training_stats = self.learning_summary_type(self.root, "training_stats")
        self.evaluation_stats = self.eval_summary_type(self.root, "evaluation_stats")
        self.early_stopping = LearningEarlyStopping(self.root)

    @property
    def actual_last_epoch(self):
        return self.training_stats.last_nonzero_step

    @property
    def actual_last_evaluation(self):
        return self.evaluation_stats.last_nonzero_step    

class SynthesisBatchStatistics(HierarchicalObject):
    def __init__(self, parent):
        super(SynthesisBatchStatistics, self).__init__(parent)

        self.root = self.parent.require_group("stats")
        self.read()

    def read(self):
        self.step = HierarchicalDataset(
            self.root, "step", (None,), dtype=np.int64)
        self.loss = HierarchicalDataset(
            self.root, "loss", (None,), dtype=np.float32)

    def add(self, step_, loss_):
        self.step.add(step_)
        self.loss.add(loss_)

class SynthesisEarlyStopping(HierarchicalObject):
    def __init__(self, parent):
        super(SynthesisEarlyStopping, self).__init__(parent)

        self.root = self.parent.require_group("early_stopping")

    @property
    def step(self):
        return self.root.attrs["step"]

    @step.setter
    def step(self, value):
        self.root.attrs["step"] = value

    @property
    def loss(self):
        return self.root.attrs["loss"]

    @loss.setter
    def loss(self, value):
        self.root.attrs["loss"] = value
   
class SynthesisBatch(HierarchicalObject):
    def __init__(self, parent, number, summary_type=None):
        super(SynthesisBatch, self).__init__(parent)

        self.number = number
        self.summary_type = summary_type
        self.root = self.parent.require_group(str(self.number))
        self.synthesized = None

        self.read()

    def synthesized_batch(self, images, labels, names):
        self.synthesized.images.add(images)
        self.synthesized.labels.add(labels)
        self.synthesized.names.add(names)
        self.synthesized.flush()
        
    def stopped_early(self, step, loss):
        self.early_stopping.step = step
        self.early_stopping.loss = loss
        self.early_stopping.flush()

    def read(self):
        self.stats = SynthesisBatchStatistics(self.root)
        self.early_stopping = SynthesisEarlyStopping(self.root)
        self.synthesized = SavedBatch(self.root, "synthesized", flatten=True)

class SynthesisBatches(HierarchicalGroup):
    def __init__(self, parent, summary_type=None):
        super(SynthesisBatches, self).__init__(parent)

        self.root = self.parent.require_group("batch")
        self.summary_type = summary_type
        
        self.read()

    def read(self):
        batch_keys = self.get_sorted_keys()

        for batch_key in batch_keys:
            try:
                self.instances.append(SynthesisBatch(self.root, int(batch_key)))
            except:
                pass

    def add(self):
        batch_insertion_key = self.get_next_insertion_key()
        batch = SynthesisBatch(self.root, int(batch_insertion_key), self.summary_type)
        self.instances.append(batch)

        return batch

class SynthesisStage(HierarchicalObject):
    def __init__(self, parent, summary_type=None):
        super(SynthesisStage, self).__init__(parent)

        self.root = self.parent.require_group("synthesis")
        self.summary_type = summary_type
        
        self.read()

    def read(self):
        self.batches = SynthesisBatches(self.root, self.summary_type)
        
class Round(HierarchicalObject):
    def __init__(self, parent, number, store, learning_summary_type=None, eval_summary_type=None, synthesis_summary_type=None):
        super(Round, self).__init__(parent)

        self.number = number
        self.store = store
        self.learning_summary_type = learning_summary_type
        self.eval_summary_type = eval_summary_type
        self.synthesis_summary_type = synthesis_summary_type
        
        self.root = self.parent.require_group(str(self.number))

        self.read()

    def read(self):
        self.learning = LearningStage(self.root, self.store, self.learning_summary_type, self.eval_summary_type)
        self.synthesis = SynthesisStage(self.root, self.synthesis_summary_type)

    def close(self):
        if isinstance(self.parent, h5py.File):
            self.parent.close()

LearningSummaryDescription = collections.namedtuple("LearningSummaryDescription", ["number_epochs", "summarizer_type"])
EvaluationSummaryDescription = collections.namedtuple("EvaluationSummaryDescription", ["number_epochs", "summarizer_type"])
SynthesisSummaryDescription = collections.namedtuple("SynthesisSummaryDescription", ["number_batches", "summarizer_type"])

class Rounds(HierarchicalGroup):
    ROUND_FILENAME = "round_{0}.h5"
    
    def __init__(self, parent, store, streaming=True):
        super(Rounds, self).__init__(parent)
        
        self.root = self.parent.require_group("round")
        self.paths = HierarchicalDataset(
            self.root, "paths", (None,), dtype=h5py.special_dtype(vlen=bytes))
        self.store = store
        self.streaming = streaming

        if not self.streaming:
            self.read()

    def read(self):
        round_keys = self.get_sorted_keys()

        for round_key in round_keys:
            self.instances.append(Round(self.root, int(round_key)))

    def add(self, learning_desc=None, evaluation_desc=None, synthesis_desc=None):
        round_insertion_key = str(self.paths.size)
        round_filename = Rounds.ROUND_FILENAME.format(round_insertion_key)
        self.paths.add(round_filename)
        self.paths.flush()

        round_path = os.path.join(os.path.dirname(self.parent.filename), round_filename)
        round_file = h5py.File(round_path, mode="w", libver="latest")
       
        if self.streaming:
            learning_summary_type, eval_summary_type, synthesis_summary_type = None, None, None
            if not (learning_desc is None):
                learning_summary_type = summary_type_for_summarizer(learning_desc.summarizer_type)
            if not (evaluation_desc is None):
                eval_summary_type = summary_type_for_summarizer(evaluation_desc.summarizer_type)
            if not (synthesis_desc is None):
                synthesis_summary_type = summary_type_for_summarizer(synthesis_desc.summarizer_type)
                
            round = Round(round_file, int(round_insertion_key), self.store,
                          learning_summary_type, eval_summary_type, synthesis_summary_type)
                          
            # hack to populate these beforehand.
            if not (learning_desc is None):
                for epoch_number in range(learning_desc.number_epochs):
                    round.learning.epochs.add()

            if not (evaluation_desc is None):
                for epoch_number in range(evaluation_desc.number_epochs):
                    round.learning.evaluations.add()
                    
            if not (synthesis_desc is None):
                for batch_number in range(synthesis_desc.number_batches):
                    round.synthesis.batches.add()
                
            # turn on the magic.
            round_file.flush()
            round_file.swmr_mode = True
            
            self.instances.append(round)            
        else:
            round = Round(round_file, int(round_insertion_key), self.store)
            self.instances.append(round)

        return round

    def write_external_links(self):
        if not self.streaming:
            return
        
        for index, instance in enumerate(self.instances):
            self.root[str(instance.number)] = h5py.ExternalLink(self.paths[index], "/{0}".format(instance.number))

# note that these our "new" syntheses.
class Syntheses(HierarchicalGroup):
    def __init__(self, parent):
        super(Syntheses, self).__init__(parent)
        
        self.root = self.parent.require_group("syntheses")

        self.read()

    def read(self):
        keys = self.get_sorted_keys()

        for key in keys:
            self.instances.append(OfflineRunDataFile(self.root[key], mode="r"))

    def add(self, name, synthesis_data_file_path):
        self.root[name] = h5py.ExternalLink(synthesis_data_file_path, "/")

class ImageStore(HierarchicalObject):
    def __init__(self, parent):
        super(ImageStore, self).__init__(parent)

        self.root = self.parent.require_group("store")
        self.names_mapping = {}

        self.read()

    @property
    def next_identifier(self):
        return self.images.size

    def read(self):
        self.images = HierarchicalDataset(
            self.root, "images", (None, None, None, None),
            dtype=np.uint8, compression="gzip", flatten=True)

    def store(self, images, labels, names):
        next_identifier_internal = self.next_identifier
        identifiers = []
        missed_indexes = []
        
        for index in range(np.shape(images)[0]):
            name = names[index].lower()
            if name in self.names_mapping:
                identifiers.append(self.names_mapping[name])
            else:
                # we didn't find it, store it (delayed).
                self.names_mapping[name] = next_identifier_internal
                missed_indexes.append(index)

                identifiers.append(next_identifier_internal)
                next_identifier_internal += 1

        if len(missed_indexes) > 0:
            self.images.add(images[missed_indexes])

        return identifiers
    
    def close(self):
        pass

class OfflineRunDataFile(object):
    def __init__(self, path, mode="w", attrs=None, swmr=False, open_rounds=True):
        super(OfflineRunDataFile, self).__init__()

        self.mode = mode
        self.swmr = swmr
        self.open_rounds = open_rounds        

        if isinstance(path, h5py.File):
            self.file = path
            self.path = self.file.filename
        else:
            self.path = path

            # prevent accidental truncation.
            if os.path.exists(self.path) and self.mode == "w":
                raise ValueError("{0} already exists".format(self.path))

            self.file = h5py.File(self.path, self.mode, libver="latest", swmr=(self.swmr and self.mode != "w"))            

        # we only set SWMR here if it's requested and we're not the writer (which gets set _after_
        # we create the needed
        self.root = self.file
        self.comment = None
        self.type = None
        self.rounds = None
        self.syntheses = None

        self.read()
        
        # every group/dataset should be created by now.
        if mode == "w":
            if attrs:
                for k, v in attrs.iteritems():
                    self.root.attrs[k] = v

            # cannot set swmr_mode to false.
            if self.swmr:
                self.file.swmr_mode = True

    def read(self):
        self.store = ImageStore(self.root)

        if "comment" in self.root.attrs:
            self.comment = self.root.attrs["comment"]

        if "type" in self.root.attrs:
            self.type = self.root.attrs["type"]

        if self.open_rounds:
            self.rounds = Rounds(self.root, self.store, self.swmr)

            if "syntheses" in self.root or (self.mode != "r"):            
                self.syntheses = Syntheses(self.root)
            
    def close(self):
        if self.mode == "w" and self.swmr:
            self.rounds.write_external_links()
            
        if self.file:
            self.file.close()
