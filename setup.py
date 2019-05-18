# -*- coding: utf-8 -*-
"""Setup file for rawhdr."""

import os
import re
import setuptools


def long_description():
    """Read long description from README.md."""
    with open('README.md', 'r') as readme_file:
        return readme_file.read()


def version():
    """Parse version info."""
    with open(os.path.join('rawhdr', '__init__.py'), 'r') as init_file:
        content = init_file.read()
    match = re.search(r"__version__ = [']([^']*)[']", content, re.MULTILINE)
    if match:
        return match.group(1)
    raise RuntimeError('Unable to find version string')


setuptools.setup(
    name='rawhdr',
    version=version(),
    author='Felix Thaler',
    author_email='felix.thaler@nummi.ch',
    description='A simple HDR image merger',
    long_description=long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/fthaler/rawhdr',
    packages=setuptools.find_packages(),
    scripts=['bin/rawhdrmerge'],
    install_requires=['click', 'imageio', 'numpy', 'rawpy'],
    classifiers=[
        'Programming Language :: Python', 'License :: OSI Approved :: '
        'GNU General Public License v2 or later (GPLv2+)',
        'Operating System :: OS Independent',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion'
    ],
)
