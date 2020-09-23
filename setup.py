 #!/usr/bin/env python

from setuptools import setup, find_packages

version = "0.1"

setup(name='pseudoBackprop',
      version=version,
      description='',
      author='Akos Kungl',
      author_email='afkungl@kip.uni-heidelberg.de',
      url='https://github.com/afkungl/psedoBackprop',
      packages=["pseudoBackprop"],
      package_dir={
          "pseudoBackprop": "pseudoBackprop",
          },
      license="GNUv3",
      install_requires=["matplotlib", "numpy", "torch"],
      )