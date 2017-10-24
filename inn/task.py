import os
import pdb
import tensorflow as tf

class GraphBasedTask(object):    
    def __init__(self, log_path, config_proto=None, graph=None, device_string=None):
        self.config_proto = config_proto
        if self.config_proto is None:
            self.config_proto = tf.ConfigProto(allow_soft_placement=True)
            self.config_proto.gpu_options.allow_growth = True
            
        self.graph = graph
        self.graph_ctx = None

        if self.graph is None:
            self.graph = tf.Graph()

            self.graph_ctx = self.graph.as_default()
            self.graph_ctx.enforce_nesting = False
            self.graph_ctx = self.graph_ctx.__enter__()

        if device_string is None:
            device_string = "/gpu:0"
            
        self.device_ctx = tf.device(device_string)
        self.device_ctx = self.device_ctx.__enter__()

        if not (log_path is None):
            self.summary_writer = tf.summary.FileWriter(logdir=log_path)
        else:
            self.summary_writer = None

    def __call__(self, session):
        # everything in the graph needs to be created before this.
        if self.summary_writer is not None:
            # if we own the graph.
            self.summary_writer.add_graph(self.graph)

    def __enter__(self):
        return self

    def __exit__(self, exc, value, tb):
        result = None
        
        if not (self.device_ctx is None):
            result = self.device_ctx.__exit__(exc, value, tb)

        if not (self.graph_ctx is None):
            result = self.graph_ctx.__exit__(exc, value, tb)    

        return result
