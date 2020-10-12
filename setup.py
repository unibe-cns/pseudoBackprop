#!/usr/bin/env python
"""
  Setup for pip install using setuptools
"""

from setuptools import setup
import pseudo_backprop


setup(name='pseudo_backprop',
      version=pseudo_backprop.__version__,
      description='',
      author='Akos F. Kungl',
      author_email='fkungl@kip.uni-heidelberg.de',
      url='https://github.com/afkungl/pseudoBackprop',
      packages=["pseudo_backprop", "pseudo_backprop/experiments"],
      package_dir={
          "pseudo_backprop": "pseudo_backprop",
      },
      license="GNUv3",
      install_requires=["matplotlib", "numpy", "torch"],
      package_data={
          "pseudo_backprop": ["defaults/plotstyle"],
          },
      include_package_data=True
      )
