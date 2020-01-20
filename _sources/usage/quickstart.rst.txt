Quickstart
==========

Installation
------------

Install using pip:

.. code-block:: bash

   pip install rawhdr


Usage
-----

Merging HDR Images
~~~~~~~~~~~~~~~~~~

Run `rawhdr` from the command line.
To merge three Nikon NEF files, run for example:

.. code-block:: bash

   rawhdr hdr-merge --output result.exr base.NEF under-exposed.NEF over-exposed.NEF

Any RAW file format known by the `rawpy <https://letmaik.github.io/rawpy/>`_ library is supported as input.
Any HDR file format knwon by `imageio <https://imageio.github.io/>`_ is supported as output.

.. note::

   The exposure of the generated HDR image alway matches the first given RAW image.

.. note::

    For further command line parameters, see ``rawhdr hdr-merge --help``.
    But in many cases the default parameters should work reasonably well.

Focus Stacking
~~~~~~~~~~~~~~

Focus-stacking works similar to HDR merging:

.. code-block:: bash

    rawhdr focus-stack --output result.exr image-1.NEF image-2.NEF image-3.NEF …

.. note::

    For further command line parameters, see ``rawhdr focus-stack --help``.
    But in many cases the default parameters should work reasonably well.