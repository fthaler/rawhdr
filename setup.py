import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

from rawhdr.version import __version__

setuptools.setup(
    name='rawHDR',
    version=__version__,
    author='Felix Thaler',
    author_email='felix.thaler@nummi.ch',
    description='A simple HDR image merger',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fthaler/rawhdr',
    packages=setuptools.find_packages(),
    scripts=['bin/rawhdr'],
    install_requires=['click', 'imageio', 'numpy', 'rawpy'],
    classifiers=[
        'Programming Language :: Python',
        'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
        'Operating System :: OS Independent',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion'
    ],
)
