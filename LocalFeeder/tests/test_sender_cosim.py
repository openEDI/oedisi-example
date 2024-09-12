import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sender_cosim


def test_true_angles():
    for angle in np.random.uniform(-np.pi, np.pi, 100):
        assert (sender_cosim.get_true_phases(angle) - angle) <= np.pi / 6

    for angle in np.linspace(-np.pi, np.pi, 7):
        assert np.isclose(sender_cosim.get_true_phases(angle), angle)
