import pathlib
import tempfile
import warnings

import imageio
import numpy as np
import rawpy


def load_image(path, return_original_dtype=False):
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
        rgb = rgb.astype('float32') / np.float32(2**16)
        dtype = rgb.dtype
    except rawpy._rawpy.LibRawError:
        rgb = imageio.imread(path)
        dtype = rgb.dtype
        if rgb.dtype.kind != 'f':
            if rgb.dtype.kind != 'u':
                raise RuntimeError(
                    'only RAW, floating point, and unsigned integer images are supported'
                )
            rgb = rgb.astype('float32') / np.float32(np.iinfo(rgb.dtype).max)
        if rgb.ndim == 3 and rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
    mask = np.isnan(rgb)
    if np.any(mask):
        warnings.warn(
            f"fixed {np.count_nonzero(mask)} NaN values in image {path}")
        rgb[mask] = 0

    if return_original_dtype:
        return rgb, dtype
    return rgb


def save_image(path, image, dtype=None):
    if dtype is None:
        path_str = str(path).lower()
        if path_str.endswith('.jpg') or path_str.endswith('.jpeg'):
            dtype = np.dtype('uint8')
        elif path_str.endswith('.png'):
            dtype = np.dtype('uint16')
    if dtype is not None:
        if dtype.kind != 'f':
            if dtype.kind != 'u':
                raise RuntimeError(
                    'only floating point and unsigned integer images are supported'
                )
            image = (np.clip(image, 0, 1) * np.iinfo(dtype).max).astype(dtype)
    imageio.imsave(path, image)


def principal_component(image):
    if image.ndim == 2:
        return image

    mu = np.mean(image, axis=(0, 1), keepdims=True)
    b = np.reshape(image - mu, (-1, image.shape[2]))
    cov = (b.T @ b) / image.dtype.type(b.shape[0] - 1)
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
