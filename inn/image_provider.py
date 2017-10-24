import glob
import numpy as np
import os
import pdb
from PIL import Image

from inn.util import shape_as_string

PNG_EXTENSION = ".png"
REORDER_CACHE = {
        "BGRRGB" : (2, 1, 0),
        "RGBBGR" : (2, 1, 0),
}

def gaussian_noise(image_size, sigma, mu=0.0, channels=3):
    width, height = image_size
    
    return np.random.normal(mu, sigma, size=(1, height, width, channels))

def uniform_noise(image_size, low, high, channels=3):
    width, height = image_size
    
    return np.random.uniform(low, high, size=(1, height, width, channels))

def denormalize_from_range_around_origin(image, half_length):
    clipped = np.clip(image, -half_length, half_length) + half_length
    factor_by = 255.0 / (2.0 * half_length)

    return image * factor_by

def read_image_filenames_at_path(path):
    if os.path.isdir(path):
        matching_filenames = []
        for ext in [".png", ".jpg"]:
            matching_filenames += glob.glob(os.path.join(path, "*" + ext))
    else:
        matching_filenames = [path]

    return matching_filenames

def read_images_at_path(path, image_size, is_color, resize_size=None):
    # we want to H x W.
    array_image_size = tuple(reversed(image_size))

    if not (resize_size is None):
        array_image_size = tuple(reversed(resize_size))
    
    if is_color and len(array_image_size) == 2:
        array_image_size = array_image_size + (3,)

    matching_filenames = read_image_filenames_at_path(path)
    number_images = len(matching_filenames)

    if number_images <= 0:
        raise ValueError("found no images at path {0}".format(path))

    images_at_path = np.zeros((number_images,) + array_image_size, dtype=np.float32)
    names_at_path = np.zeros((number_images,), dtype=np.object_)
    for index, filename in enumerate(matching_filenames):
        image_at_path = Image.open(filename)

        # we can relax this to allow slightly different image sizes if desired
        # later.
        if (resize_size is None) and (image_at_path.size != image_size):
            raise ValueError("image size mismatch for {0}".format(filename))
        
        if is_color:
            image_at_path = image_at_path.convert("RGB")

        if not (resize_size is None):
            image_at_path = image_at_path.resize(tuple(resize_size))

        image_bytes = np.array(image_at_path)
        
        images_at_path[index, :] = image_bytes
        names_at_path[index] = "".join(os.path.basename(filename).split(".")[:-1])

    return (names_at_path, images_at_path)

def reorder_image_channels(image, from_order="RGB", to_order=None):
    if to_order is None:
        return image

    if from_order.upper() == to_order.upper():
        return image

    permutation = REORDER_CACHE[from_order.upper() + to_order.upper()]
    return image[:, :, permutation]

class ImageProvider(object):
    def __init__(self):
        self._images = None
        self._names = None

    @property
    def number_images(self):
        return np.shape(self.images)[0]

    @property
    def image_shape(self):
        return np.shape(self.images)[1:]

    @property
    def images(self):
        if self._images is None:
            self.read()

        return self._images

    @images.setter
    def images(self, value):
        self._images = value

    @property
    def names(self):
        if self._names is None:
            self.read()
            
        return self._names

    @names.setter
    def names(self, value):
        self._names = value

    def ensure_equal(self):
        number_distinct_names = len(set(self.names))
        number_images = np.shape(self.images)[0]

        if number_distinct_names != number_images:
            repeated_names = list(set([x for x in self.names if self.names.count(x) > 1]))
            for repeated_name in repeated_names:
                print "{0} is repeated {1} time(s)".format(repeated_name, self.names.count(repeated_name))
            
            raise ValueError("passthrough has unequal images and names: {0} vs {1}".format(number_images, number_distinct_names))

    def ensure_multiple(self, batch_size):
        if (self.number_images % batch_size) != 0:
            raise ValueError("expected batch size to divide number of images")
        
class PassthroughProvider(ImageProvider):
    def __init__(self, names, images):
        super(PassthroughProvider, self).__init__()

        self.names = list(names)
        self.images = np.copy(images)        
        self.ensure_equal()

    def read(self):
        pass
        
class LoadFromImageFileProvider(ImageProvider):
    def __init__(self, path, image_size, channel_order, at_most_n=None, resize_size=None, normalize_fn=None):
        super(LoadFromImageFileProvider, self).__init__()
        self.path = path
        self.image_size = image_size
        self.channel_order = channel_order
        self.at_most_n = at_most_n
        self.resize_size = resize_size
        self.normalize_fn = normalize_fn

    def read(self):
        is_color = len(self.channel_order) >= 3
        self.names, self.images = read_images_at_path(            
            self.path, self.image_size, is_color, self.resize_size)
        self.images = reorder_image_channels(self.images, to_order=self.channel_order)

        if not (self.at_most_n is None):
            number_images_read = np.shape(self.images)[0]

            if number_images_read > self.at_most_n:
                self.names = self.names[:self.at_most_n]
                self.images = self.images[:self.at_most_n]

        if not (self.normalize_fn is None):
            self.images = self.normalize_fn(self.images)
                    
class LoadFromNoiseProvider(ImageProvider):
    def __init__(self, number_images, image_size, noise_fn, normalize_fn=None, name_prefix="noise"):
        super(LoadFromNoiseProvider, self).__init__()

        self._number_images = number_images
        self.image_size = image_size
        self.noise_fn = noise_fn
        self.normalize_fn = normalize_fn
        self.name_prefix = name_prefix

    def read(self):
        random_images = []
        random_names = []

        for image_index in range(self._number_images):
            generated_image = self.noise_fn(self.image_size[0], self.image_size[1])

            if self.normalize_fn:
                generated_image = self.normalize_fn(generated_image)

            random_images.append(generated_image)
            random_names.append(self.name_prefix + str(image_index))

        self.images = np.concatenate(random_images)
        self.names = np.array(random_names)

class LimitImageProvider(ImageProvider):
    def __init__(self, provider, limit_images):
        super(LimitImageProvider, self).__init__()
        self.provider = provider
        self.limit_images = limit_images

    def read(self):
        unfiltered_images = self.provider.images
        unfiltered_names = self.provider.names

        if self.provider.number_images < self.limit_images:
            raise ValueError("cannot filter a smaller number of images")

        self.images = unfiltered_images[:self.limit_images]
        self.names = unfiltered_names[:self.limit_images]

class DuplicateImageProvider(ImageProvider):
    def __init__(self, provider, number_times):
        super(DuplicateImageProvider, self).__init__()
        self.provider = provider
        if self.provider.number_images != 1:
            raise ValueError("can only duplicate a single image")

        self.number_times = number_times

    def read(self):
        tile_dfn = np.ones(self.provider.images.ndim, dtype=np.int64)
        tile_dfn[0] = self.number_times

        self.images = np.tile(self.provider.images, tile_dfn)

        given_name = self.provider.names[0]
        duplicated_names = []

        for duplicate_index in range(self.number_times):
            duplicated_names.append("{0}-{1}".format(given_name, duplicate_index))

        self.names = np.array(duplicated_names)

class PatchesProvider(ImageProvider):
    def __init__(self, provider, number_patches, patch_size, resize_size=None):
        super(PatchesProvider, self).__init__()
        self.provider = provider
        self.number_patches = number_patches
        self.patch_size = patch_size
        self.resize_size = resize_size

        with tf.name_scope("create_patches"):
            self.generate_source = tf.placeholder(tf.float32, (None,) + self.provider.image_shape)
            self.generate_number = tf.placeholder(tf.int32)
            self.generate_patches, self.generate_patch_names = random_image_patches(
                self.generate_source, self.generate_number, self.patch_size, resize_size=self.resize_size)

    def read(self):
        tf_config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=tf_config) as session:
            generated_images, generated_names = session.run(
                [self.generate_patches, self.generate_patch_names],
                feed_dict={self.generate_source: self.provider.images, self.generate_number: self.number_patches})

            self.images = np.clip(generated_images, 0.0, 255.0).astype(np.uint8).astype(np.float32)
            self.names = generated_names
        
