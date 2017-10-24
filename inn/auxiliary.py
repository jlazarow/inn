import pdb
import tensorflow as tf

def get_expanded_image_size(image_size, patch_size):
    if image_size == patch_size:
        return image_size

    # otherwise, we should add a bit of padding
    patch_width, patch_height = patch_size
    
    if (patch_width % 2) != 0 or (patch_height % 2) != 0:
        raise ValueError("expected patch size to be even")

    image_width, image_height = image_size

    return (image_width + patch_width, image_height + patch_height)

def clip_expanded_image(image, patch_size):
    patch_width, patch_height = patch_size
    
    # clip a half width/height on either size.
    half_patch_width = int(patch_width / 2)
    half_patch_height = int(patch_height / 2)
    
    return image[half_patch_height:-half_patch_height, half_patch_width:-half_patch_width]
                                                  
class AuxiliaryPatchProcessor(object):
    def __init__(self, requested_image_size, number_patches, patch_size, resize_size=None):
        self.requested_image_size = requested_image_size
        self.number_patches = number_patches
        self.patch_size = patch_size
        self.resize_size = resize_size

        self.image_index_for_patch = None
        self.box_for_patch = None
        self.name_for_patch = None
        self.label_for_patch = None

    @property
    def expanded_image_size(self):
        # determine if the caller needs to expand the given image or not.
        if self.patch_size == self.requested_image_size:
            # no expansion, the requested and input size are equal.
            return self.requested_image_size

        # otherwise, we should add a bit of padding
        if self.patch_size[0] % 2 != 0 or self.patch_size[1] % 2 != 0:
            raise ValueError("expected patch size to be event")

        return (self.requested_image_size[0] + self.patch_size[0],
                self.requested_image_size[1] + self.patch_size[1])

    @property
    def crop_image_box(self):
        if self.patch_size == self.requested_image_size:
            # no expansion was done.
            return (0, 0, self.patch_size[1], self.patch_size[0])

        # otherwise, remove the padding
        half_patch_width = self.patch_size[0] / 2
        half_patch_height = self.patch_size[1] / 2
        expanded_image_size = self.expanded_image_size

        return (half_patch_height,
                half_patch_width,
                expanded_image_size[1] - half_patch_height,
                expanded_image_size[0] - half_patch_width)

    def __call__(self, images, labels, number_patches):
        raise NotImplementedError("AuxiliaryProcessor.__call__")

class RandomImagePatchesProcessor(AuxiliaryPatchProcessor):
    # override this to customize how patch positions are selected.n
    def select_random_images(self, count, images, labels):
        batch_shape = tf.shape(images)
        batch_size = batch_shape[0]

        with tf.name_scope("random_images"):
            self.random_images = tf.random_uniform((count,), minval=0, maxval=batch_size, dtype=tf.int32)

            return self.random_images
        
    def select_random_positions(self, count, images, labels):
        batch_shape = tf.shape(images)
        image_height = batch_shape[1]
        image_width = batch_shape[2]        
        patch_width, patch_height = self.patch_size
        
        # the default just selects random (x, y) independently (and with respect to the image chosen).
        with tf.name_scope("random_normalized_positions"):
            random_x = tf.random_uniform((count, 1), maxval=(image_width - patch_width + 1), dtype=tf.int32)            
            random_y = tf.random_uniform((count, 1), maxval=(image_height - patch_height + 1), dtype=tf.int32)
    
        return tf.concat([random_y, random_x], 1)

    def __call__(self, images, labels):
        with tf.name_scope("random_image_patches"):
            # first, we randomly pick which image each patch should be assigned to.
            batch_shape = tf.shape(images)
            batch_size = batch_shape[0]

            image_height = batch_shape[1]
            image_width = batch_shape[2]
            patch_width, patch_height = self.patch_size

            resize_width, resize_height = patch_width, patch_height

            if not (self.resize_size is None):
                if not (self.resize_size[0] is None):
                    resize_width = self.resize_size[0]

                if not (self.resize_size[1] is None):
                    resize_height = self.resize_size[1]
            else:
                # unsure if this makes the resize part a no-op.
                resize_width = patch_width
                resize_height = patch_height
        
            normalized_width = tf.div(tf.cast(patch_width, tf.float32), tf.cast(image_width, tf.float32))
            normalized_height = tf.div(tf.cast(patch_height, tf.float32), tf.cast(image_height, tf.float32))

            self.image_index_for_patch = self.select_random_images(self.number_patches, images, labels)
            self.position_for_patch = self.select_random_positions(self.number_patches, images, labels)

            self.position_for_patch = tf.div(
                tf.cast(self.position_for_patch, tf.float32),
                [tf.cast(image_height, tf.float32), tf.cast(image_width, tf.float32)])
            self.box_for_patch = tf.concat(
                [self.position_for_patch, self.position_for_patch + [normalized_height, normalized_width]], 1)
            self.name_for_patch = tf.string_join(
                [tf.reduce_join(tf.as_string(self.box_for_patch, precision=2), 1, separator=","),
                 tf.as_string(tf.squeeze(self.image_index_for_patch))],
                separator=":")
            self.label_for_patch = tf.gather(labels, self.image_index_for_patch)

            self.crop_and_resize = tf.image.crop_and_resize(
                images, self.box_for_patch, self.image_index_for_patch, [resize_height, resize_width])

            return (self.crop_and_resize, self.label_for_patch, self.name_for_patch)
