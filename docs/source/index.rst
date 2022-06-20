.. py-wrp documentation master file, created by
   sphinx-quickstart on Fri Jun 17 08:55:55 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

A wave reconstruction and propagation library in python
=======================================================

.. note::

   This project is under active development.

Demo
----

Here's an example of the program in action using the example script in scripts/staticVis.py. 
It loads data which has already been recorded and saved in increments as specified by :doc:`overview`.
This video is meant to demonstrate the predictive ability of the program as it iterates through data but does not capture the ability of the program to also generate real time outputs based on this prediction.

.. image:: ../images/hero.gif
    :width: 600

Installation
------------

The easiest way to run this code locally will be to pull the entire repository directly from GitHub. 
If you already have git and GitHub installed on your computer, this should be as easy as typing::
   git clone git@github.com:doe-fowt-control/py-wrp.git

If you have never used git, you should 

   - make an account on `github<https://github.com>`
   - download and install `git<https://git-scm.com/downloads>`
   - Set up git with username and email in a terminal::
      $ git config --global user.name "Your name here"
      $ git config --global user.email "your_email@example.com"
   (Don’t type the $; that just indicates that you’re doing this at the command line.)
   - execute the following command in a directory where you want the folder to be installed::
      git clone https://github.com/doe-fowt-control/py-wrp.git

This assumes that for new users you will simply want access to the code. To make changes or interact with the repository directly
you will need to set up an SSH key. There is more information available online `here<https://docs.github.com/en/authentication/connecting-to-github-with-ssh/about-ssh>`


.. toctree::
   overview
   background
   theory
   reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
