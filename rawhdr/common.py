import pathlib
import tempfile

import imageio
import numpy as np
import rawpy


def load_image(path):
    """Load a raw image file.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    image : ndarray
        Loaded image data in linear color space.
    """
    path = str(path)
    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(gamma=(1, 1),
                                  no_auto_bright=True,
                                  use_camera_wb=True,
                                  output_bps=16)
        return rgb.astype('float32') / np.float32(2**16)
    except rawpy._rawpy.LibRawNonFatalError:
        rgb = imageio.imread(path)
        if rgb.dtype.kind != 'f':
            raise RuntimeError('only RAW or floating point images are support')
        if rgb.ndim == 3 and rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
        return rgb.astype('float32')


def save_image(path, image):
    imageio.imsave(path, image)


def principal_component(image):
    if image.ndim == 2:
        return image

    mu = np.mean(image, axis=(0, 1), keepdims=True)
    b = np.reshape(image - mu, (-1, image.shape[2]))
    cov = (b.T @ b) / (b.shape[0] - 1)
    w, v = np.linalg.eig(cov)
    i = np.argmax(w)
    return np.reshape(b @ v[:, i], image.shape[:2])


def reduce_color_dimension(image, pca=False):
    if image.ndim == 2:
        return image
    if pca:
        return principal_component(image)
    return np.sqrt(np.sum(image**2, axis=2))


class _TemporaryArrayList:
    class _Slice:
        def __init__(self, parent, range_):
            self._parent = parent
            self._range = range_

        def __len__(self):
            return self._stop - self._start

        def __getitem__(self, index):
            if isinstance(index, slice):
                return type(self)(self.parent, self._range[index])
            return self._parent[self._range[index]]

        def __setitem__(self, index, value):
            self._parent[self._range[index]] = value

        def __iter__(self):
            def generator():
                for i in self._range:
                    yield self._parent[i]

            return generator()

    def __init__(self, iterable=None):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._len = 0
        if iterable is not None:
            for value in iterable:
                self.append(value)

    def __len__(self):
        return self._len

    def _path(self, index):
        if index < 0:
            index = self._len + index
        assert 0 <= index < self._len
        return pathlib.Path(self._tmpdir.name) / f'{index}.npy'

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._Slice(self, range(self._len)[index])
        return np.load(self._path(index), mmap_mode='r+')

    def __setitem__(self, index, value):
        assert isinstance(index, int)
        return np.save(self._path(index), value)

    def append(self, value):
        self._len += 1
        self[-1] = value

    def __iter__(self):
        def generator():
            for i in range(self._len):
                yield self[i]

        return generator()


def temporary_array_list(iterable=None, *, in_memory=False):
    if in_memory:
        return [] if iterable is None else list(iterable)
    return _TemporaryArrayList(iterable)
