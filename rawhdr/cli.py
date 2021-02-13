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
    from rawhdr import exposure_fusion

    if not images:
        return

    if output is None:
        output = os.path.splitext(images[0])[0] + '-hdr.exr'

    if save_memory:
        # Load one image after another to save memory
        fused, other_images = images[0], images[1:]
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
              help='File name of the stacked output image.')
@click.option('--sigma',
              type=float,
              help='Standard deviation of filter used in merger '
              '(availabilty and exact meaning depends on --merger option).')
@click.option('--levels',
              type=int,
              help='Number of levels used in wavelet transforms '
              '(availabilty and exact meaning depends on --merger option).')
@click.option('--merger',
              type=click.Choice(['highpass', 'wavelet', 'dtcwt', 'wavelet2']),
              help='Kind of merger to use.',
              default='highpass')
def focus_stack(images, output, sigma, levels, merger):
    """Command-line utility for focus stacking of RAW images.

    All input images must be RAW images.
    """
    from rawhdr import focusstack

    if not images:
        return

    if output is None:
        output = os.path.splitext(images[0])[0] + '-stacked.exr'

    def merge(first, second):
        if merger == 'dtcwt':
            return focusstack.merge_dtcwt(first,
                                          second,
                                          levels=levels,
                                          sigma=sigma)
        if merger == 'highpass':
            if levels is not None:
                raise ValueError(
                    'Option --levels is not supported for this merger.')
            return focusstack.merge_highpass(first, second, sigma=sigma)
        elif merger == 'wavelet':
            return focusstack.merge_waveletes(first,
                                              second,
                                              levels=levels,
                                              sigma=sigma)
        elif merger == 'wavelet2':
            if sigma is not None:
                raise ValueError(
                    'Option --sigma is not supported for this merger.')
            return focusstack.merge_waveletes2(first, second, levels=levels)
        else:
            raise ValueError()

    images = list(images)
    stacked = load_image(images.pop())
    while images:
        image = load_image(images.pop())
        stacked = merge(stacked, image)

    save_image(output, stacked.astype('float32'))


if __name__ == '__main__':
    main()
