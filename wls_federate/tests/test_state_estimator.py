import os
import sys

import numpy as np
import scipy.sparse
import pytest
from oedisi.types.data_types import (
    Topology,
    PowersReal,
    PowersImaginary,
    VoltagesMagnitude,
    VoltagesReal,
    VoltagesImaginary,
)
from scipy.sparse import sparray

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(TEST_DIR))

from state_estimator_federate import (
    calculate_jacobian,
    residual,
    get_y,
    AlgorithmParameters,
    get_indices,
)


@pytest.fixture()
def parameters():
    return AlgorithmParameters()


@pytest.fixture()
def topology():
    return Topology.parse_file(os.path.join(TEST_DIR, "topology.json"))


@pytest.fixture()
def measurements():
    return (
        PowersReal.parse_file(os.path.join(TEST_DIR, "power_real.json")),
        PowersImaginary.parse_file(os.path.join(TEST_DIR, "power_imag.json")),
        VoltagesMagnitude.parse_file(os.path.join(TEST_DIR, "voltage_magnitude.json")),
    )


@pytest.fixture()
def sparse_topology():
    return Topology.parse_file(os.path.join(TEST_DIR, "sparse_topology.json"))


@pytest.fixture()
def actuals():
    return (
        VoltagesReal.parse_file(os.path.join(TEST_DIR, "voltage_real.json")),
        VoltagesImaginary.parse_file(os.path.join(TEST_DIR, "voltage_imag.json")),
    )


def inner_args(parameters, topology, measurements):
    P, Q, V = measurements
    knownP = get_indices(topology, P)
    knownQ = get_indices(topology, Q)
    knownV = get_indices(topology, V)
    base_voltages = np.array(topology.base_voltage_magnitudes.values)
    num_node = len(topology.base_voltage_magnitudes.ids)
    base_power = parameters.base_power
    z = np.concatenate(
        (
            V.values / base_voltages[knownV],
            -np.array(P.values) / base_power,
            -np.array(Q.values) / base_power,
        ),
        axis=0,
    )

    Y = get_y(topology.admittance, topology.base_voltage_magnitudes.ids)

    Y = (
        scipy.sparse.diags_array(base_voltages)
        @ Y
        @ scipy.sparse.diags_array(base_voltages)
    ) / (base_power * 1000)
    initial_ang = np.array(topology.base_voltage_angles.values)
    X0 = np.concatenate((initial_ang, np.full(num_node, 1)))
    return X0, z, num_node, knownP, knownQ, knownV, Y


def test_calculate_jacobian(parameters, topology, measurements):
    X0, z, num_node, knownP, knownQ, knownV, Y = inner_args(
        parameters, topology, measurements
    )
    H = calculate_jacobian(X0, z, num_node, knownP, knownQ, knownV, Y)
    assert H.shape == (len(knownP) + len(knownQ) + len(knownV), num_node * 2)
    assert isinstance(H, np.ndarray), f"H has type {type(H)}"
    # This messes up scipy.optimize something fierce
    assert not isinstance(H, np.matrix), f"H has type {type(H)}"


def test_get_y_sparse(sparse_topology):
    base_voltages = np.array(sparse_topology.base_voltage_magnitudes.values)
    base_power = 100
    ids = sparse_topology.base_voltage_magnitudes.ids
    Y = get_y(sparse_topology.admittance, ids)
    assert Y.shape == (len(ids), len(ids))
    assert isinstance(Y, sparray), f"Y has type {type(Y)}"

    Y = (
        scipy.sparse.diags_array(base_voltages)
        @ Y
        @ scipy.sparse.diags_array(base_voltages)
    ) / (base_power * 1000)
    assert Y.shape == (len(ids), len(ids))
    assert isinstance(Y, sparray), f"Y has type {type(Y)}"


def test_calculate_jacobian_sparse(parameters, sparse_topology, measurements):
    X0, z, num_node, knownP, knownQ, knownV, Y = inner_args(
        parameters, sparse_topology, measurements
    )
    assert isinstance(Y, sparray), f"Y has type {type(Y)}"
    H = calculate_jacobian(X0, z, num_node, knownP, knownQ, knownV, Y)
    assert H.shape == (len(knownP) + len(knownQ) + len(knownV), num_node * 2)
    assert isinstance(H, sparray), f"H has type {type(H)}"


def test_residual(parameters, topology, measurements):
    X0, z, num_node, knownP, knownQ, knownV, Y = inner_args(
        parameters, topology, measurements
    )
    h = residual(X0, z, num_node, knownP, knownQ, knownV, Y)
    assert h.shape == (len(knownP) + len(knownQ) + len(knownV),)
    assert isinstance(h, np.ndarray), f"h has type {type(h)}"
    assert not isinstance(h, np.matrix), f"h has type {type(h)}"


def test_residual_sparse(parameters, sparse_topology, measurements):
    X0, z, num_node, knownP, knownQ, knownV, Y = inner_args(
        parameters, sparse_topology, measurements
    )
    h = residual(X0, z, num_node, knownP, knownQ, knownV, Y)
    assert h.shape == (len(knownP) + len(knownQ) + len(knownV),)
    assert isinstance(h, np.ndarray), f"h has type {type(h)}"
    assert not isinstance(h, np.matrix), f"h has type {type(h)}"


def test_residual_against_actuals(parameters, topology, measurements, actuals):
    _, z, num_node, knownP, knownQ, knownV, Y = inner_args(
        parameters, topology, measurements
    )
    voltage_real, voltage_imag = actuals
    true_voltages = np.array(voltage_real.values) + 1j * np.array(voltage_imag.values)
    true_voltages /= np.array(topology.base_voltage_magnitudes.values)

    X0 = np.concatenate((np.angle(true_voltages), np.abs(true_voltages)))
    h = residual(X0, z, num_node, knownP, knownQ, knownV, Y)
    assert np.sum(np.abs(h)) < 1e-2, f"Residuals are too high: {h}"


def get_mean_relative_error(topology, solution, actuals):
    vmagestDecen, vangestDecen = (
        solution[len(solution) // 2 :],
        solution[: len(solution) // 2],
    )

    slack_id = topology.base_voltage_magnitudes.ids.index(topology.slack_bus[0])
    vangestDecen = vangestDecen - vangestDecen[slack_id]

    voltage_mag = vmagestDecen * np.array(topology.base_voltage_magnitudes.values)
    voltage_ang = vangestDecen

    voltage_real, voltage_imag = actuals
    true_voltage = np.array(voltage_real.values) + 1j * np.array(voltage_imag.values)
    estimated_voltage = voltage_mag * np.exp(1j * voltage_ang)

    return np.abs(
        (estimated_voltage - true_voltage)
        / np.array(topology.base_voltage_magnitudes.values)
    ).mean()


def test_least_squares_call(parameters, topology, measurements, actuals):
    X0, z, num_node, knownP, knownQ, knownV, Y = inner_args(
        parameters, topology, measurements
    )

    ls_result = scipy.optimize.least_squares(
        residual,
        X0,
        jac=calculate_jacobian,
        # bounds=(low_limit, up_limit),
        method="trf",
        # method="lm",
        verbose=2,
        ftol=0.0001,
        xtol=0.00001,
        gtol=0.0001,
        args=(z, num_node, knownP, knownQ, knownV, Y),
    )
    solution = ls_result.x

    mean_rel_error = get_mean_relative_error(topology, solution, actuals)
    assert mean_rel_error < 0.1, f"Max relative error too high: {mean_rel_error}"


def test_compare_initial_conditions(
    parameters, topology, sparse_topology, measurements
):
    X0, z, num_node, knownP, knownQ, knownV, Y = inner_args(
        parameters, topology, measurements
    )

    (
        X0_sparse,
        z_sparse,
        num_node_sparse,
        knownP_sparse,
        knownQ_sparse,
        knownV_sparse,
        Y_sparse,
    ) = inner_args(parameters, sparse_topology, measurements)
    assert np.allclose(X0_sparse, X0)
    assert np.allclose(z_sparse, z)
    assert num_node_sparse == num_node
    assert knownP_sparse == knownP
    assert knownQ_sparse == knownQ
    assert knownV_sparse == knownV
    assert np.allclose(Y_sparse.toarray(), Y)


def test_compare_jacobian_residuals(
    parameters, topology, sparse_topology, measurements
):
    X0, z, num_node, knownP, knownQ, knownV, Y = inner_args(
        parameters, topology, measurements
    )

    H = calculate_jacobian(X0, z, num_node, knownP, knownQ, knownV, Y)
    res = residual(X0, z, num_node, knownP, knownQ, knownV, Y)

    X0, z, num_node, knownP, knownQ, knownV, Y = inner_args(
        parameters, sparse_topology, measurements
    )
    H_sparse = calculate_jacobian(X0, z, num_node, knownP, knownQ, knownV, Y)
    res_sparse = residual(X0, z, num_node, knownP, knownQ, knownV, Y)

    assert np.allclose(H_sparse.toarray(), H)
    assert np.allclose(res_sparse, res)


def test_least_squares_call_sparse(parameters, sparse_topology, measurements, actuals):
    X0, z, num_node, knownP, knownQ, knownV, Y = inner_args(
        parameters, sparse_topology, measurements
    )

    ls_result = scipy.optimize.least_squares(
        residual,
        X0,
        jac=calculate_jacobian,
        # bounds=(low_limit, up_limit),
        method="trf",
        # method="lm",
        verbose=2,
        ftol=0.0001,
        xtol=0.00001,
        gtol=0.0001,
        args=(z, num_node, knownP, knownQ, knownV, Y),
    )
    solution = ls_result.x

    mean_rel_error = get_mean_relative_error(sparse_topology, solution, actuals)
    assert mean_rel_error < 0.1, f"Max relative error too high: {mean_rel_error}"


def test_least_squares_with_perfect_initalization(
    parameters, topology, measurements, actuals
):
    _, z, num_node, knownP, knownQ, knownV, Y = inner_args(
        parameters, topology, measurements
    )
    voltage_real, voltage_imag = actuals
    true_voltages = np.array(voltage_real.values) + 1j * np.array(voltage_imag.values)
    true_voltages /= np.array(topology.base_voltage_magnitudes.values)

    X0 = np.concatenate((np.angle(true_voltages), np.abs(true_voltages)))
    ls_result = scipy.optimize.least_squares(
        residual,
        X0,
        jac=calculate_jacobian,
        # bounds=(low_limit, up_limit),
        method="trf",
        # method="lm",
        verbose=2,
        ftol=0.000001,
        xtol=0.000001,
        gtol=0.000001,
        max_nfev=1000,
        args=(z, num_node, knownP, knownQ, knownV, Y),
    )
    assert ls_result.success, f"Least squares failed: {ls_result.message}"

    solution = ls_result.x

    mean_rel_error = get_mean_relative_error(topology, solution, actuals)
    assert mean_rel_error < 1e-4, f"Max relative error too high: {mean_rel_error}"
