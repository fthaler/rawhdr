Quickstart
==========

Installation
------------

Install using pip:

.. code-block:: bash

   pip install rawhdr


Usage
-----

Run `rawhdrmerge` from the command line.
To merge three Nikon NEF files, run for example:

.. code-block:: bash

   rawhdrmerge --output result.exr base.NEF under-exposed.NEF over-exposed.NEF

Any RAW file format known by the `rawpy <https://letmaik.github.io/rawpy/>`_ library is supported as input.
Any HDR file format knwon by `imageio <https://imageio.github.io/>`_ is supported as output.

.. note::

   The exposure of the generated HDR image alway matches the first given RAW image.
