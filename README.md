# Overview

This is the code to accompany the paper [Introspective Generative Modeling: Decide Discriminatively](https://arxiv.org/abs/1704.07820)

# Introspective Neural Networks
Although the paper goes into more detail on this, the high level overview of the ideas seen here are:

1. Attempt to model some "class" (in our case, textures) of examples
2. Consider this in a "supervised" setting by introduce a positive (1) label to said examples
3. Introduce a negative label (0) to label all non-examples of our texture e.g. Gaussian noise is in this class
4. Train a discriminator (CNN) to discriminate between the positive and negative examples
5. Use this to synthesize examples of negative label to ones considered to be positive by the discriminator.
6. Augment the negative training set with these new examples.
7. Repeat

Eventually (for example approximately 35 stages for the "pebbles" texture), step (5) will start
to produce visually similar syntheses to those in the positive class.

Now that the model has been trained, new syntheses (without repeating the training process)
can be produced by initializing a new image form noise and only repeating the synthesis part (step 5) of
the previous steps.

# Running the code
## Prerequisites
1. TensorFlow 1.3+
2. h5py (pip install h5py)
3. numpy (recent version)
4. Pillow or some form of PIL (pip install Pillow)

You should add the checked out repository to your PYTHONPATH.

Two corresponding scripts are provided at:

1. experiments/iccv/learn-texture.py
2. experiments/iccv/synthesize-texture.py

Both take a filename (not the full path) of example textures in datasets/textures.

For example:

`
python experiments/iccv/learn-texture.py diamond.png
`

will start training the diamond texture.

For the given script, it's best to run these from the root of the repositories or
otherwise modify the experiment paths inside to better suit your environment.

Each execution of the script will create a timestamped folder beneath
"results/your_experiment_name". You can inspect this (especially the "snapshots" folder)
for more insight onto the generation process. Additionally, syntheses will be placed under
"syntheses".

# Extending the code

The code is written to be extremely flexible. Not only can arbitrary
discriminator models i.e. other CNN architectures be added (see network.py), but the
processes themselves can be extended quite easily.

# Other issues

Feel free to email me at mygithubusername@ucsd.edu for specific details or questions.
