# Rawhdr

[![Build Status](https://travis-ci.com/fthaler/rawhdr.svg?branch=master)](https://travis-ci.com/fthaler/rawhdr)

A simple HDR image merger that converts multiple RAW files into a single HDR image.


## Installation

Installation via the [Python Package Index](https://pypi.org/):

`$> pip install rawhdr`

## Command-Line Usage

Use the command-line tool _rawhdrmerge_ to merge multiple RAW images into a single high-dynamic-range image.

`$> rawhdrmerge -o merged-hdr.exr base-exposure.RAW under-exposed.RAW over-exposed.RAW`

All common RAW file formats are supported.

## Documentation

See [fthaler.github.io/rawhdr](https://fthaler.github.io/rawhdr) for the full documentation, including Python API.