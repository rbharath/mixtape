# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2013, Stanford University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#   Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import absolute_import, print_function, division
from six import PY2
import numpy as np
from sklearn import cluster

__all__ = ['KMeans', 'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift',
           'SpectralClustering', 'Ward', 'KCenters', 'MultiSequenceClusterMixin']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class MultiSequenceClusterMixin(object):

    # The API for the scikit-learn Cluster object is, in fit(), that
    # they take a single 2D array of shape (n_data_points, n_features).
    #
    # For clustering a collection of timeseries, we need to preserve
    # the structure of which data_point came from which sequence. If
    # we concatenate the sequences together, we lose that information.
    #
    # This mixin is basically a little "adaptor" that changes fit()
    # so that it accepts a list of sequences. Its implementation
    # concatenates the sequences, calls the superclass fit(), and
    # then splits the labels_ back into the sequenced form.

    def fit(self, sequences, y=None):
        """Fit the  clustering on the data

        Parameters
        ----------
        sequences : list of array-like, each of shape [sequence_length, n_features]
            A list of multivariate timeseries. Each sequence may have
            a different length, but they all must have the same number
            of features.

        Returns
        -------
        self
        """
        if len(sequences) > 0 and isinstance(sequences[0], np.ndarray):
            concat = np.concatenate(sequences)
        else:
            # if the input sequences are not numpy arrays, we need to guess
            # how to concatenate them. this operation below works for mdtraj
            # trajectories (which is the use case that I want to be sure to
            # support), but in general the python container protocol doesn't
            # give us a generic way to make sure we merged sequences
            concat = sequences[0].join(sequences[1:])

        lengths = [len(s) for s in sequences]

        # make sure that the concatenation operation worked
        assert sum(lengths) == len(concat)

        s = super(MultiSequenceClusterMixin, self) if PY2 else super()
        s.fit(concat)
        self.labels_ = self._split(self.labels_, lengths)
        return self

    @staticmethod
    def _split(longlist, lengths):
        return [longlist[cl - l: cl] for (cl, l) in zip(np.cumsum(lengths), lengths)]

    def predict(self, sequences):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        sequences : list of array-like, each of shape [sequence_length, n_features]
            A list of multivariate timeseries. Each sequence may have
            a different length, but they all must have the same number
            of features.

        Returns
        -------
        Y : list of arrays, each of shape [sequence_length,]
            Index of the closest center each sample belongs to.
        """
        s = super(MultiSequenceClusterMixin, self) if PY2 else super()
        predictions = []
        for sequence in sequences:
            predictions.append(s.predict(sequence))
        return predictions

    def fit_predict(self, sequences):
        '''Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        sequences : list of array-like, each of shape [sequence_length, n_features]
            A list of multivariate timeseries. Each sequence may have
            a different length, but they all must have the same number
            of features.

        Returns
        -------
        Y : list of ndarray, each of shape [sequence_length, ]
            Cluster labels
        '''
        return self.fit(sequences).labels_


def _replace_labels(doc):
    """Really hacky find-and-replace method that modifies one of the sklearn
    docstrings to change the semantics of labels_ for the subclasses"""
    lines = doc.splitlines()
    labelstart, labelend = None, None
    foundattributes = False
    for i, line in enumerate(lines):
        if 'Attributes' in line:
            foundattributes = True
        if 'labels' in line and not labelstart and foundattributes:
            labelstart = len('\n'.join(lines[:i]))
        if labelstart and line.strip() == '' and not labelend:
            labelend = len('\n'.join(lines[:i+1]))


    replace  = '''\n    `labels_` : list of arrays, each of shape [sequence_length, ]
        `labels_[i]` is an array of the labels of each point in
        sequence `i`. The label of each point is an integer in
        [0, n_clusters).
    '''

    return doc[:labelstart] + replace + doc[labelend:]

#-----------------------------------------------------------------------------
# New "multisequence" versions of all of the clustering algorithims in sklearn
#-----------------------------------------------------------------------------

class KMeans(MultiSequenceClusterMixin, cluster.KMeans):
    __doc__ = _replace_labels(cluster.KMeans.__doc__)

class MiniBatchKMeans(MultiSequenceClusterMixin, cluster.MiniBatchKMeans):
    __doc__ = _replace_labels(cluster.MiniBatchKMeans.__doc__)

class AffinityPropagation(MultiSequenceClusterMixin, cluster.AffinityPropagation):
    __doc__ = _replace_labels(cluster.AffinityPropagation.__doc__)

class MeanShift(MultiSequenceClusterMixin, cluster.MeanShift):
    __doc__ = _replace_labels(cluster.MeanShift.__doc__)

class SpectralClustering(MultiSequenceClusterMixin, cluster.SpectralClustering):
    __doc__ = _replace_labels(cluster.SpectralClustering.__doc__)

class Ward(MultiSequenceClusterMixin, cluster.Ward):
    __doc__ = _replace_labels(cluster.Ward.__doc__)

# This needs to come _after_ MultiSequenceClusterMixin is defined, to avoid
# recursive circular imports
from .kcenters import KCenters
