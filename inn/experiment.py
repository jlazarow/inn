from __future__ import absolute_import

import datetime
import inspect
import os
import pdb
import shutil

from scipy.misc import imsave

from inn.offline import OfflineRunDataFile
from inn.summarizer import *

# the existence of this file means the experiment is running.
RUNNING_LOCK = "running.{0}"
SNAPSHOTS_NAME = "snapshots"

class ExperimentRun(object):
    def __init__(self, base_directory, experiment_name, comment, type):
        self.base_directory = base_directory
        self.experiment_name = experiment_name
        self.comment = comment
        self.type = type
        self.run_name = None
        self.running_lock_name = None

    @property
    def run_path(self):
        return os.path.join(self.base_directory, self.experiment_name, self.run_name)

    @property
    def data_path(self):
        return os.path.join(self.run_path, "data")

    @property
    def checkpoints_path(self):
        return os.path.join(self.run_path, "checkpoints")

    @property
    def snapshots_path(self):
        return os.path.join(self.run_path, "snapshots")

    def __enter__(self):
        self.run_name = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")

        if not os.path.exists(self.run_path):
            os.makedirs(self.run_path)

        os.mkdir(self.data_path)
        os.mkdir(self.checkpoints_path)

        # save the calling script.
        current_frame = inspect.currentframe()
        outermost_frame = inspect.getouterframes(current_frame, 2)[-1]
        calling_script = os.path.join(os.getcwd(), outermost_frame[1])
        shutil.copy(calling_script, self.run_path)

        # initialize the HDF file (in streaming mode).
        attrs = { "name": self.run_name, "type": self.type }

        if self.comment:
            attrs["comment"] = self.comment

        self.file = OfflineRunDataFile(os.path.join(self.data_path, "index.h5"), "w", attrs, swmr=True)
        self.running_lock_name = RUNNING_LOCK.format(os.getpid())
        with open(os.path.join(self.run_path, self.running_lock_name), "a"):
            pass

        print "experiment started in {0}".format(self.run_path)

        return self

    def add_snapshot(self, name, image):
        if not os.path.exists(self.snapshots_path):
            os.mkdir(self.snapshots_path)

        imsave(os.path.join(self.snapshots_path, name + ".png"), image)

    def __exit__(self, type, value, tb):
        os.remove(os.path.join(self.run_path, self.running_lock_name))

        self.file.close()            
        
        # here we should do some post-processing e.g. zipping the run.
        print "experiment concluded in {0}".format(self.run_path)
