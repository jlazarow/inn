import os
import numpy as np
import pdb
import sys
import tensorflow as tf

from PIL import Image

from inn.auxiliary import *
from inn.early_stopping import EarlyStoppingCriterion
from inn.experiment import ExperimentRun
from inn.image_provider import *
from inn.network import DCGAN
from inn.parameters import DiscriminativeClass
from inn.learning import LearnDiscriminatively
from inn.stage import *
from inn.summarizer import *
from inn.util import managed_or_debug_session

GPU_NUMBER = 0
EXPERIMENT_DIRECTORY = "results"
TEXTURE_FILENAME = sys.argv[1]
TEXTURE_NAME = TEXTURE_FILENAME.split(".")[0]
TEXTURE_PATH = os.path.join("datasets", "textures", TEXTURE_FILENAME)

with Image.open(TEXTURE_PATH) as texture_file:
    TEXTURE_SIZE = texture_file.size

INPUT_SIZE = (64, 64)
PATCH_SIZE = (64, 64)
NUMBER_NEGATIVES = 200
BATCH_SIZE = 100
NUMBER_PATCHES = 1000
DF_DIM = 64
RETAIN_FACTOR = 1
MAXIMUM_LEARNING_BATCHES = 30
DEBUG = True

if not os.path.exists(EXPERIMENT_DIRECTORY):
    os.mkdir(EXPERIMENT_DIRECTORY)

network = DCGAN(INPUT_SIZE, 2, DF_DIM)

# the noise will be prenormalized.
noise_fn = lambda width, height: np.clip(gaussian_noise((width, height), sigma=0.3), -1.0, 1.0)
normalize_fn = lambda im: (im / 127.5) - 1.0
denormalize_fn = lambda im: (im + 1.0) * 127.5
expanded_image_size = get_expanded_image_size(TEXTURE_SIZE, PATCH_SIZE)

def create_texture_class(auditor=None):
    texture = DiscriminativeClass(
        TEXTURE_NAME,
        label=1,
        positives_initializer_fn=lambda: DuplicateImageProvider(
            LoadFromImageFileProvider(TEXTURE_PATH, TEXTURE_SIZE, "RGB", resize_size=expanded_image_size, normalize_fn=normalize_fn), NUMBER_NEGATIVES),
        negatives_initializer_fn=lambda number_negatives: LoadFromNoiseProvider(
            number_negatives, expanded_image_size, noise_fn, normalize_fn=None,
            name_prefix="neg"),
        synthesis_initializer_fn=lambda number_negatives: LoadFromNoiseProvider(
            number_negatives, expanded_image_size, noise_fn, normalize_fn=None,
            name_prefix="syn"),
        prior_probability=1.0)

    return texture

NAME = "learn_{0}_texture".format(TEXTURE_NAME)
COMMENT = "DCGAN + other interesting comments"
    
with ExperimentRun(EXPERIMENT_DIRECTORY, NAME, comment=COMMENT, type="learning") as experiment:
    texture = create_texture_class()
    patch_processor = RandomImagePatchesProcessor(TEXTURE_SIZE, NUMBER_PATCHES, PATCH_SIZE, network.input_size)

    train_summarizer_fn = lambda d: LossAndAccuracySummarizer(d.training_accuracy)
    eval_summarizer_fn = lambda d: LossAndAccuracySummarizer(d.testing_accuracy)

    learning_optimizer_fn = lambda step, n_iter: tf.train.GradientDescentOptimizer(
        learning_rate=.01)
    synthesis_optimizer_fn = lambda step: tf.train.AdamOptimizer(
        learning_rate=0.01, beta1=0.5)

    learn_texture = LearnDiscriminatively(
        class_definitions=[texture],
        network=network,
        batch_size=BATCH_SIZE,
        batch_image_size=tuple(reversed(expanded_image_size)),
        create_learn_epoch_fn=lambda d, aux: LearningEpochStage(d, train_summarizer_fn(d), learning_optimizer_fn, aux, denormalize_fn),
        create_evaluate_epoch_fn=lambda d, aux: LearningEvaluationStage(d, eval_summarizer_fn(d), aux, denormalize_fn),
        create_synthesis_fn=lambda d, i: SynthesisStage(d, i, synthesis_optimizer_fn, denormalize_fn),
        experiment=experiment,
        negative_labels=[0],
        auxiliary_processors=[patch_processor],
        graph=tf.get_default_graph(),
        device_string="/gpu:{0}".format(GPU_NUMBER))

    with managed_or_debug_session(hooks=[], scaffold=learn_texture.scaffold, config=learn_texture.config_proto, debug=DEBUG) as session:
        learn_texture(session, number_learning_epochs=30,
                      learning_early_stopping_fn=lambda: EarlyStoppingCriterion(lambda s: s.loss <= 0.005),
                      evaluate_every_n=5, number_evaluation_batches=10, number_synthesis_epochs=500,
                      synthesis_early_stopping_fn=lambda: EarlyStoppingCriterion(lambda v: v <= 0.62),
                      maximum_learning_batches=MAXIMUM_LEARNING_BATCHES, retain_batch_n_times=RETAIN_FACTOR, debug=DEBUG)

