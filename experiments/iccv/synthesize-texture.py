import os
import numpy as np
import pdb
import sys
import tensorflow as tf

from scipy.misc import imsave

from inn.auxiliary import *
from inn.early_stopping import EarlyStoppingCriterion
from inn.experiment import ExperimentRun
from inn.image_provider import *
from inn.network import DCGAN
from inn.parameters import DiscriminativeClass
from inn.synthesis import SynthesizeDiscriminatively
from inn.stage import *
from inn.util import managed_or_debug_session, prompt_for_run

GPU_NUMBER = 0
EXPERIMENT_DIRECTORY = "results"
TEXTURE_FILENAME = sys.argv[1]
TEXTURE_NAME = TEXTURE_FILENAME.split(".")[0]
MODEL_NAME = "learn_{0}_texture".format(TEXTURE_NAME)
# will be prompted unless manually specified here.
RUN_NAME = None
# note, set this to the desired synthesis size
# it does not need to be the same as the original texture.
TEXTURE_SIZE = (256, 256)
INPUT_SIZE = (64, 64)
PATCH_SIZE = (64, 64)
NUMBER_TO_SYNTHESIZE = 1
# generally, it's best to match the average number of patches
# that would have been selected per image * number of synthesis.
# here we use 10 because in the corresponding "learn-texture.py" file
# a batch size of 100 with 1000 patches created uniformly at random means
# that on average, a single image should see 10 patches selected from it.
NUMBER_PATCHES = 10
DF_DIM = 64
DEBUG = True

network = DCGAN(INPUT_SIZE, 2, DF_DIM)

# the noise will be prenormalized.
noise_fn = lambda width, height: np.clip(gaussian_noise((width, height), sigma=0.3), -1.0, 1.0)
normalize_fn = lambda im: (im / 127.5) - 1.0
denormalize_fn = lambda im: (im + 1.0) * 127.5
expanded_image_size = get_expanded_image_size(TEXTURE_SIZE, PATCH_SIZE)

def create_texture_class():
    texture = DiscriminativeClass(
        TEXTURE_NAME,
        label=1,
        positives_initializer_fn=None,
        negatives_initializer_fn=None,
        synthesis_initializer_fn=lambda number_synthesis: LoadFromNoiseProvider(
            number_synthesis, expanded_image_size, noise_fn, normalize_fn=None,
            name_prefix="syn"),
        prior_probability=1.0)

    return texture

if RUN_NAME is None:
    model_path = os.path.join(EXPERIMENT_DIRECTORY, MODEL_NAME, prompt_for_run(os.path.join(EXPERIMENT_DIRECTORY, MODEL_NAME)))
else:
    model_path = os.path.join(EXPERIMENT_DIRECTORY, MODEL_NAME, RUN_NAME)

NAME = "synthesize_{0}_texture".format(TEXTURE_NAME)
# optionally set this to limit the number of "stages" used.
# one possible measure of this is when the synthesis process is no longer able to produce a positive, on average.
LAST_ROUND = None # e.g. 37
with ExperimentRun(EXPERIMENT_DIRECTORY, NAME, comment="synthesizing the {0} texture".format(TEXTURE_NAME), type="synthesis") as experiment:
    texture = create_texture_class()
    patch_processor = RandomImagePatchesProcessor(TEXTURE_SIZE, NUMBER_PATCHES, PATCH_SIZE, network.input_size)

    synthesis_optimizer_fn = lambda step: tf.train.AdamOptimizer(
        learning_rate=0.01, beta1=0.5)    
    synthesize_texture = SynthesizeDiscriminatively(
        class_definitions=[texture],
        number_synthesis_per_class=[NUMBER_TO_SYNTHESIZE],
        network=network,
        batch_size=min(NUMBER_TO_SYNTHESIZE, 100),
        batch_image_size=tuple(reversed(expanded_image_size)),
        create_synthesis_fn=lambda d, i: SynthesisStage(d, i, synthesis_optimizer_fn, denormalize_fn),
        experiment=experiment,
        auxiliary_processors=[patch_processor],
        graph=tf.get_default_graph(),
        device_string="/gpu:{0}".format(GPU_NUMBER))

    with managed_or_debug_session(hooks=[], scaffold=synthesize_texture.scaffold, config=synthesize_texture.config_proto, debug=DEBUG) as session:
        class_syntheses = synthesize_texture(session, model_path, number_synthesis_epochs=500,
                                             synthesis_early_stopping_fn=lambda: EarlyStoppingCriterion(lambda v: v <= 0.62),
                                             synthesis_schedule_fn=lambda ckpts: ckpts[:LAST_ROUND],
                                             debug=DEBUG)

        syntheses_path = os.path.join(experiment.run_path, "syntheses")
        if not os.path.exists(syntheses_path):
            os.mkdir(syntheses_path)
        
        # write the final syntheses.
        for synthesis_class in class_syntheses:
            final_synthesis = synthesis_class.synthesis.most_recent

            for synthesis_index in range(final_synthesis.size):
                image_at_index = clip_expanded_image(denormalize_fn(final_synthesis.images[synthesis_index]), PATCH_SIZE)

                # clip if needed.
                image_filename = "class{0}_{1}.png".format(synthesis_class.positive_label, final_synthesis.names[synthesis_index])
                imsave(os.path.join(syntheses_path, image_filename), image_at_index)

