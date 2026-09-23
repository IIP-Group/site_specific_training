"""PUSCH LS channel estimation without OCC combining.

This module targets Sionna 0.19.2.  The stock
``sionna.nr.PUSCHLSChannelEstimator`` first forms per-resource-element LS
estimates and subsequently combines adjacent estimates in frequency (and, for
double-symbol DM-RS, in time) to separate ports that share one CDM group.

For the configuration used in the accompanying study, the active DM-RS ports
are 0 and 2.  They occupy disjoint type-1 DM-RS combs, and ports 1 and 3 are
inactive.  Hence, no OCC combining is required to separate the active ports.
The class below replaces only this combining step.  Sionna's inherited public
``call`` method still interpolates the per-resource-element estimates according
to ``interpolation_type`` or ``interpolator``.  The companion covariance script
discards those interpolated values and retains only the original even or odd
DM-RS samples of the corresponding layer.

Important: this estimator produces an unbiased per-RE estimate for a stream
only if no other active port occupies the same DM-RS resource elements.  In
particular, explicitly configure ``dmrs_port_set=[0, 2]`` for the two-layer
case.  Sionna's default two-layer port set is ``[0, 1]``; those ports share a
CDM group and require OCC combining.

Author: Nuri Berke Baytekin
"""

from __future__ import annotations

import tensorflow as tf

from sionna.nr import PUSCHLSChannelEstimator
from sionna.ofdm import LSChannelEstimator


class PUSCHLSChannelEstimatorNoOCC(PUSCHLSChannelEstimator):
    """PUSCH LS estimator that skips frequency- and time-domain OCC combining.

    The constructor mirrors the configuration-related arguments of Sionna
    0.19.2's :class:`PUSCHLSChannelEstimator`.  The inherited ``call`` method
    removes nulled subcarriers, gathers the received pilot samples, invokes
    :meth:`estimate_at_pilot_locations`, and then applies the configured
    interpolation exactly as in the stock estimator.

    ``err_var`` is the per-RE LS error variance

        no / abs(pilot)**2,

    expressed in the same channel-amplitude units as ``h_ls``.  No factor
    ``1/2`` is applied because no two LS estimates are averaged.
    """

    def __init__(
        self,
        resource_grid,
        dmrs_length,
        dmrs_additional_position,
        num_cdm_groups_without_data,
        interpolation_type="nn",
        interpolator=None,
        dtype=tf.complex64,
        **kwargs,
    ):
        super().__init__(
            resource_grid=resource_grid,
            dmrs_length=dmrs_length,
            dmrs_additional_position=dmrs_additional_position,
            num_cdm_groups_without_data=num_cdm_groups_without_data,
            interpolation_type=interpolation_type,
            interpolator=interpolator,
            dtype=dtype,
            **kwargs,
        )

    def estimate_at_pilot_locations(self, y_pilots, no):
        """Compute Sionna's raw LS estimate before PUSCH OCC combining.

        Calling the generic Sionna 0.19.2 LS implementation is preferable to
        duplicating its broadcasting logic.  It divides by the complete pilot
        symbols stored in the pilot pattern, including DM-RS power scaling and
        the frequency- and time-domain OCC weights, and returns
        ``no/abs(pilot)**2`` as the corresponding error variance.
        """

        return LSChannelEstimator.estimate_at_pilot_locations(
            self, y_pilots, no
        )

    @property
    def pilot_symbols(self):
        """Pilot vector used for LS division in Sionna's internal ordering."""

        return self._pilot_pattern.pilots

    @property
    def valid_pilot_mask(self):
        """Boolean mask of nonzero pilot entries for every TX stream."""

        magnitudes = tf.abs(tf.convert_to_tensor(self._pilot_pattern.pilots))
        return tf.not_equal(magnitudes, tf.zeros([], magnitudes.dtype))

    @property
    def num_dmrs_symbols(self) -> int:
        """Number of configured DM-RS OFDM symbols."""

        return int(self._num_dmrs_syms)

    @property
    def num_pilots_per_dmrs_symbol(self) -> int:
        """Length of the gathered pilot vector for one DM-RS symbol."""

        return int(self._num_pilots_per_dmrs_sym)
