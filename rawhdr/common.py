import imageio
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
        return rgb.astype('float32')
