"""Command line tool of rawhdr."""

import os

import click

import rawhdr
from rawhdr.common import load_image, save_image


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
              help='File name of the output HDR image.')
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

    save_image(output, fused.astype('float32'))


@main.command()
@click.argument('images', nargs=-1, type=click.Path(exists=True))
@click.option('--output',
              '-o',
              type=click.Path(),
              help='File name of the output HDR image.')
@click.option('--wavelet-levels', '-w', type=int)
def generic_fusion(images, output, wavelet_levels):
    if not images:
        return

    if output is None:
        output = os.path.splitext(images[0])[0] + '-fused.exr'

    from rawhdr import generic_fusion

    fused, *other_images = images
    fused = load_image(fused)
    for image in other_images:
        image = load_image(image)
        fused = generic_fusion.fuse_wavelets(fused,
                                             image,
                                             levels=wavelet_levels)
        del image

    save_image(output, fused.astype('float32'))


if __name__ == '__main__':
    main()
