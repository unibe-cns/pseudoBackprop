# -*- coding: utf-8 -*-
"""
Package to implement and test pseudoackprop

This module is meant to implement pseudo backpropagation. This means that
in the backpropagation algorithm the backward pass flows throught the
pseudoinverse of the forward weight instead of the transpose. The idea is
following the suplementary in Lillicrap (2016), that this pseudo-backprop
approximates a second order optimization of the network. As a comparison, we
use vanilla backpropagation and feedback alignement.

Caution: We keep our experiments simple. The results might depend strongly on
the studied dataset and metaparameters, which change or alter our findings

References:
Lillicrap, T., Cownden, D., Tweed, D. et al. Random synaptic feedback weights
support error backpropagation for deep learning. Nat Commun 7, 13276 (2016).
https://doi.org/10.1038/ncomms13276
"""

# Versioning (the manual way)
MAJOR = 0
MINOR = 1
MICRO = 0
__version__ = f'{MAJOR}.{MINOR}.{MICRO}'
