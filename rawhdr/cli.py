"""Command line tool of rawhdr."""

import os

import click
import numpy as np

import rawhdr
from rawhdr.common import load_image, save_image, temporary_array_list, reduce_color_dimension


def print_version(ctx, _, value):
    """Print version information."""
    if not value or ctx.resilient_parsing:
        return
    click.echo('rawhdr version ' + rawhdr.__version__ + '\n'
               'Copyright (C) 2019 – 2021 Felix Thaler')
    ctx.exit()


@click.group()
@click.option('--version',
              '-v',
              is_flag=True,
              callback=print_version,
              expose_value=False,
              is_eager=True)
def main():
    pass


@main.command()
@click.argument('images', nargs=-1, type=click.Path(exists=True))
@click.option('--output',
              '-o',
              type=click.Path(),
              help='File name of the output image.')
@click.option('--save-memory',
              '-s',
              is_flag=True,
              help='Use a less memory-intense algorithm that could lead '
              'to slight quality reduction in some cases.')
@click.option('--mask-width',
              type=float,
              help='Mask width option used for exposure scaling.')
@click.option('--blend-width',
              type=float,
              help='Smoothness option used for blending.')
@click.option('--blend-cap', type=float, help='Cap option used for blending.')
@click.option('--target-gamma',
              type=float,
              help='Gamma correction used in internal computations.')
def exposure_fusion(images, output, save_memory, mask_width, blend_width,
                    blend_cap, target_gamma):
    """Command-line utility for fusing RAW images into a single HDR image.

    All input images must be RAW images. The exposure of the first image is
    taken as reference for the brightness of the resulting HDR image.
    """
    if not images:
        return

    if output is None:
        output = os.path.splitext(images[0])[0] + '-hdr.exr'

    from rawhdr import exposure_fusion

    if save_memory:
        # Load one image after another to save memory
        fused, *other_images = images
        fused = load_image(fused)
        for image in other_images:
            image = load_image(image)
            fused = exposure_fusion.fuse_exposures([fused, image],
                                                   mask_width=mask_width,
                                                   blend_width=blend_width,
                                                   blend_cap=blend_cap,
                                                   target_gamma=target_gamma,
                                                   weight_first=False)
            del image
    else:
        # Load all images at ones and perform fusion
        images = [load_image(image) for image in images]
        fused = exposure_fusion.fuse_exposures(images,
                                               mask_width=mask_width,
                                               blend_width=blend_width,
                                               blend_cap=blend_cap,
                                               target_gamma=target_gamma)

    save_image(output, fused)


@main.command()
@click.argument('images', nargs=-1, type=click.Path(exists=True))
@click.option('--output',
              '-o',
              type=click.Path(),
              help='File name of the output image.')
@click.option('--wavelet-levels', '-w', type=int)
@click.option('--pca/--no-pca', default=True)
@click.option('--stationary/--not-stationary', '-s', default=False)
@click.option('--clip/--dont-clip', '-c', default=False)
def wavelet_fusion(images, output, wavelet_levels, pca, stationary, clip):
    if not images:
        return

    if output is None:
        output = os.path.splitext(images[0])[0] + '-fused.exr'

    from rawhdr import generic_fusion

    if stationary:
        fuse_func = generic_fusion.fuse_stationary_wavelets
    else:
        fuse_func = generic_fusion.fuse_wavelets

    fused, *other_images = images
    fused = load_image(fused)
    for image in other_images:
        image = load_image(image)
        fused = fuse_func(fused,
                          image,
                          levels=wavelet_levels,
                          pca=pca,
                          clip=clip)
        del image

    save_image(output, fused)


@main.command()
@click.argument('images', nargs=-1, type=click.Path(exists=True))
@click.option('--output',
              '-o',
              type=click.Path(),
              help='File name of the output image.')
@click.option('--levels', '-l', type=int)
@click.option('--pca/--no-pca', default=False)
@click.option('--clip/--dont-clip', '-c', default=False)
def pyramid_fusion(images, output, levels, pca, clip):
    if not images:
        return

    if output is None:
        output = os.path.splitext(images[0])[0] + '-fused.exr'

    from rawhdr import generic_fusion

    image = load_image(images[0])
    fused = generic_fusion.laplacian_pyramid(image, levels)
    if clip:
        image_min = np.copy(image)
        image_max = np.copy(image)
    del image
    for image in images[1:]:
        image = load_image(image)
        if clip:
            image_bw = reduce_color_dimension(image, pca)
            mask = image_bw < reduce_color_dimension(image_min, pca)
            image_min[mask, ...] = image[mask, ...]
            mask = image_bw > reduce_color_dimension(image_max, pca)
            image_max[mask, ...] = image[mask, ...]
            del image_bw
            del mask
        image = generic_fusion.laplacian_pyramid(image, levels)
        fused = generic_fusion.fuse_laplacian_pyramids(fused, image, pca=pca)
        del image

    fused = generic_fusion.laplacian_unpyramid(fused)
    if clip:
        fused_bw = reduce_color_dimension(fused, pca)
        mask = fused_bw < reduce_color_dimension(image_min, pca)
        fused[mask, ...] = image_min[mask, ...]
        mask = fused_bw > reduce_color_dimension(image_max, pca)
        fused[mask, ...] = image_max[mask, ...]
        del fused_bw
        del mask

    save_image(output, fused)


@main.command()
@click.argument('images', nargs=-1, type=click.Path(exists=True))
@click.option('--output',
              '-o',
              type=click.Path(),
              help='File name of the output image.')
@click.option('--pca/--no-pca', default=True)
@click.option('--in-memory/--not-in-memory', default=False)
@click.option('--sharpness-sigma', type=float)
@click.option('--weighted-depth-n', type=int)
@click.option('--error-weight', type=float)
@click.option('--sigma-weight', type=float)
@click.option('--weights-smoothing', type=float)
@click.option('--wavelets/--no-wavelets', default=False)
@click.option('--levels', '-l', type=int)
@click.option('--k', '-k', type=int)
def focus_fusion(images, output, pca, in_memory, sharpness_sigma,
                 weighted_depth_n, error_weight, sigma_weight,
                 weights_smoothing, wavelets, levels, k):
    if not images:
        return

    if output is None:
        output = os.path.splitext(images[0])[0] + '-fused.exr'

    from rawhdr import focus_fusion

    if wavelets:
        fuse_func = focus_fusion.fuse_focal_stack_kmax
    else:
        fuse_func = focus_fusion.fuse_focal_stack

    kwargs = dict(pca=pca, in_memory=in_memory)
    if sharpness_sigma is not None:
        kwargs.update(sharpness_sigma=sharpness_sigma)
    if weighted_depth_n is not None:
        kwargs.update(weighted_depth_n=weighted_depth_n)
    if error_weight is not None:
        kwargs.update(error_weight=error_weight)
    if sigma_weight is not None:
        kwargs.update(sigma_weight=sigma_weight)
    if weights_smoothing is not None:
        kwargs.update(weights_smoothing=weights_smoothing)
    if levels is not None:
        kwargs.update(levels=levels)
    if k is not None:
        kwargs.update(k=k)

    loaded_images = temporary_array_list(load_image(image) for image in images)

    fused = fuse_func(loaded_images, **kwargs)

    save_image(output, fused)


@main.command()
@click.argument('images', nargs=-1, type=click.Path(exists=True))
@click.option('--output',
              '-o',
              type=click.Path(),
              help='File name pattern of the output images.')
def alignment(images, output):
    if not images:
        return

    if output is None:
        output = os.path.splitext(images[0])[0] + '-aligned-{:05}.exr'

    from rawhdr import alignment

    template = load_image(images[0])
    save_image(output.format(0), template)
    mat = np.eye(2, 3, dtype=np.float32)
    for i, image in enumerate(images[1:], 1):
        image = load_image(image)
        image, mat = alignment.align(template, image, mat, True)
        save_image(output.format(i), image)
        template = image


if __name__ == '__main__':
    main()
