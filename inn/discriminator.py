import pdb
import tensorflow as tf

from inn.util import shape_as_string, render_summary

class Discriminator(object):
    def __init__(self, network, images, labels, batch_reducer=tf.reduce_mean):
        self.network = network
        self.images = images
        self.labels = labels
        self.batch_reducer = batch_reducer
        self.network_scope = None

        with tf.variable_scope("discriminator"):
            self.create_network()
            self.create_classifier()
            self.create_sampler()

    def create_network(self):
        with tf.variable_scope("network") as network_scope:
            self.training_output = self.network(self.images, is_training=True)
            network_scope.reuse_variables()
            self.testing_output = self.network(self.images, is_training=False)
            
            self.network_scope = network_scope
            
    def create_classifier(self):
        with tf.name_scope("classification"):
            possible_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if possible_regularizers:
                print "NOTE: adding {0} regularizers".format(len(possible_regularizers))
                
            with tf.name_scope("training"):
                self.training_loss = tf.reduce_mean(
                    self.network.cross_entropy_loss(self.labels, self.training_output),
                    name="xentropy_training_loss")
                    
                if possible_regularizers:
                    self.training_loss = self.training_loss + tf.add_n(possible_regularizers)
                    
                self.training_accuracy = tf.reduce_mean(self.network.classification_accuracy(
                    self.labels, self.training_output), name="training_accuracy")
                        
            with tf.name_scope("testing"):
                self.testing_loss = tf.reduce_mean(
                    self.network.cross_entropy_loss(self.labels, self.testing_output),
                    name="xentropy_testing_loss")
                    
                self.testing_accuracy = tf.reduce_mean(self.network.classification_accuracy(
                    self.labels, self.testing_output))
                                
    def create_sampler(self):
        with tf.name_scope("sampling"):
            self.sampling_loss_per_input = self.network.cross_entropy_loss(
                self.labels, self.testing_output)
            self.sampling_loss = self.batch_reducer(self.sampling_loss_per_input)
                
    @property
    def network_variables(self):
        # TensorFlow likes to hide these everywhere... just bring in all globals under the network scope.
        # return (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.network_scope.name) +
        #         tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, scope=self.network_scope.name))

        network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.network_scope.name)
        if len(network_vars) == 0:
            print "weird scoping. using backup"
            return [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.startswith("learn_discriminatively/discriminator/network")]

        return network_vars

    def saver(self, max_to_keep=500):
        saveable_variables = self.network_variables
        saver = tf.train.Saver(saveable_variables, max_to_keep=max_to_keep)

        return saver   

    def get_variable(self, name):
        for variable in self.network_variables:
            normalized_name = variable.name.split(":")[0]
            
            if normalized_name.endswith(name):
                return variable

        raise ValueError("variable {0} not found".format(name))
    
