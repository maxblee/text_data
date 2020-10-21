.. highlight:: shell
.. _installation:

Installation
============

Installation
------------

Stable release
--------------

To install :code:`text_data` without the extra visualization
features, run this command in your terminal:

.. code-block:: console

    pip install text_data

If you also want to install the visual features (which
installs :code:`altair` and :code:`pandas`), run
this command in your terminal

.. code-block:: console

    pip install text_data[display]

Or, using :code:`poetry`, run

.. code-block:: console

    poetry add text_data -E display

This is the preferred method to install Text Data, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for Text Data can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    git clone git://github.com/maxblee/text_data

Or download the `tarball`_:

.. code-block:: console

    curl -OJL https://github.com/maxblee/text_data/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    poetry install -E display

Or

.. code-block:: console

    poetry install -E display --no-dev


Additional Tutorial Setup
-------------------------
If you want to follow along with the tutorial, you should
download the `Kaggle State of the Union Corpus <https://www.kaggle.com/rtatman/state-of-the-union-corpus-1989-2017>`_,
unzip it, and put it into a directory called :code:`sotu-data`.

You should also install the latest version of :code:`text_data` with the optional
visualization features. The tutorial shows the process of conducting
an analysis using :code:`text_data` and makes extensive use
of :code:`pandas` and some use of the build-in visualization
features.

.. _Github repo: https://github.com/maxblee/text_data
.. _tarball: https://github.com/maxblee/text_data/tarball/master
