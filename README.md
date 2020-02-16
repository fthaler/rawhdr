# Rawhdr

![Test](https://github.com/fthaler/rawhdr/workflows/Test/badge.svg)

A simple HDR image merger that converts multiple RAW files into a single HDR image.

## Installation

Installation via the [Python Package Index](https://pypi.org/):

`$> pip install rawhdr`

## Command-Line Usage

### Merging HDR Images

Use the command-line tool _rawhdr_ to merge multiple RAW images into a single high-dynamic-range image.

`$> rawhdr hdr-merge -o merged-hdr.exr base-exposure.RAW under-exposed.RAW over-exposed.RAW`

All common RAW file formats are supported.

Note: if you want to save to OpenEXR format and get the error `ValueError: Could not find a format to write the specified file in mode 'i'` from _imageio_, you might need to install _freeimage_, as documented [here](https://imageio.readthedocs.io/en/stable/format_exr-fi.html#exr-fi).

### Focus Stacking

Focus-stacking works similar to HDR merging:

`$> rawhdr focus-stack --output result.exr image-1.NEF image-2.NEF image-3.NEF …`

## Documentation

See [fthaler.github.io/rawhdr](https://fthaler.github.io/rawhdr) for the full documentation, including Python API.