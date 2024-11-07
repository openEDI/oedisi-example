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
    estimated_pqv,
    get_zero_injection_indices,
)


@pytest.fixture()
def parameters():
    return AlgorithmParameters()


@pytest.fixture()
def ieee123data():
    return os.path.join(TEST_DIR, "ieee123data")


@pytest.fixture()
def small_smartds_no_tap_time_3():
    return os.path.join(TEST_DIR, "small_smartds_no_tap_time_3")


@pytest.fixture()
def small_smartds_no_tap_time_40():
    return os.path.join(TEST_DIR, "small_smartds_no_tap_time_40")


@pytest.fixture()
def small_smartds_tap_time_3():
    return os.path.join(TEST_DIR, "small_smartds_tap_time_3")


@pytest.fixture()
def small_smartds_tap_time_40():
    return os.path.join(TEST_DIR, "small_smartds_tap_time_3")


INPUT_DATA = list(
    map(
        lambda x: os.path.join(TEST_DIR, x),
        [
            "ieee123data",
            "small_smartds_no_tap_time_3",
            "small_smartds_no_tap_time_40",
            "small_smartds_tap_time_3",
            "small_smartds_tap_time_40",
        ],
    )
)


def get_topology(directory):
    return Topology.parse_file(os.path.join(directory, "topology.json"))


def get_measurements(directory):
    return (
        PowersReal.parse_file(os.path.join(directory, "power_real.json")),
        PowersImaginary.parse_file(os.path.join(directory, "power_imag.json")),
        VoltagesMagnitude.parse_file(os.path.join(directory, "voltage_magnitude.json")),
    )


@pytest.fixture()
def sparse_topology():
    return Topology.parse_file(
        os.path.join(TEST_DIR, "ieee123data", "sparse_topology.json")
    )


def get_actuals(directory):
    return (
        VoltagesReal.parse_file(os.path.join(directory, "voltage_real.json")),
        VoltagesImaginary.parse_file(os.path.join(directory, "voltage_imaginary.json")),
    )


def inner_args(parameters, topology, measurements):
    P, Q, V = measurements
    zero_power_nodes = get_zero_injection_indices(topology)
    knownP = get_indices(topology, P, extra_nodes=zero_power_nodes)
    knownQ = get_indices(topology, Q, extra_nodes=zero_power_nodes)
    knownV = get_indices(topology, V)

    base_voltages = np.array(topology.base_voltage_magnitudes.values)
    num_node = len(topology.base_voltage_magnitudes.ids)
    base_power = parameters.base_power
    # Concat enough zeros to P_array from P.values to make it len(knownP)
    P_array = np.zeros(len(knownP))
    P_array[: len(P.values)] = P.values
    Q_array = np.zeros(len(knownQ))
    Q_array[: len(Q.values)] = Q.values
    z = np.concatenate(
        (
            V.values / base_voltages[knownV],
            -P_array / base_power,
            -Q_array / base_power,
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


def test_get_indices(ieee123data):
    topology = get_topology(ieee123data)
    measurements = get_measurements(ieee123data)
    P, Q, V = measurements
    zero_power_nodes = get_zero_injection_indices(topology)
    knownP = get_indices(topology, P, extra_nodes=zero_power_nodes)
    knownQ = get_indices(topology, Q, extra_nodes=zero_power_nodes)
    assert len(knownP) >= len(P.values)
    assert len(knownQ) >= len(Q.values)

    alternate_P = get_indices(topology, P)
    assert all(
        [knownP[i] == alternate_P[i] for i in range(len(P.values))]
    ), "Measurement indices are different"


def test_zero_power_nodes(ieee123data):
    topology = get_topology(ieee123data)
    measurements = get_measurements(ieee123data)
    zero_power_nodes = get_zero_injection_indices(topology)
    P, Q, V = measurements
    assert all(
        [
            np.abs(P.values[i]) < 0.001
            for i in range(len(P.values))
            if P.ids[i] in zero_power_nodes
        ]
    )
    assert all(
        [
            np.abs(Q.values[i]) < 0.001
            for i in range(len(Q.values))
            if Q.ids[i] in zero_power_nodes
        ]
    )

    Y = get_y(topology.admittance, topology.base_voltage_magnitudes.ids)
    Y = (
        scipy.sparse.diags_array(np.array(topology.base_voltage_magnitudes.values))
        @ Y
        @ scipy.sparse.diags_array(np.array(topology.base_voltage_magnitudes.values))
    ) / (100 * 1000)

    # Get true voltages
    voltage_real, voltage_imag = get_actuals(ieee123data)
    true_voltages = np.array(voltage_real.values) + 1j * np.array(voltage_imag.values)
    true_voltages /= np.array(topology.base_voltage_magnitudes.values)

    S = true_voltages * (Y.conjugate() @ true_voltages.conjugate())
    zero_power_nodes = sorted(list(zero_power_nodes))
    inv_map = {v: i for i, v in enumerate(topology.base_voltage_magnitudes.ids)}
    zero_power_node_idx = [inv_map[node] for node in zero_power_nodes]
    max_calculated_S = np.abs(S[zero_power_node_idx])
    # Get ids with nonzero power from max_calculated_S using np.nonzero
    (nonzero_power_nodes,) = np.nonzero(max_calculated_S > 1e-6)
    nonzero_power_node_ids = [zero_power_nodes[idx] for idx in nonzero_power_nodes]
    assert (
        len(nonzero_power_node_ids) == 0
    ), f"Nonzero power nodes: {nonzero_power_node_ids} with power {max_calculated_S[nonzero_power_nodes]}"


def test_calculate_jacobian(parameters, ieee123data):
    topology = get_topology(ieee123data)
    measurements = get_measurements(ieee123data)
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


def test_calculate_jacobian_sparse(parameters, sparse_topology, ieee123data):
    measurements = get_measurements(ieee123data)
    X0, z, num_node, knownP, knownQ, knownV, Y = inner_args(
        parameters, sparse_topology, measurements
    )
    assert isinstance(Y, sparray), f"Y has type {type(Y)}"
    H = calculate_jacobian(X0, z, num_node, knownP, knownQ, knownV, Y)
    assert H.shape == (len(knownP) + len(knownQ) + len(knownV), num_node * 2)
    assert isinstance(H, sparray), f"H has type {type(H)}"


def test_residual(parameters, ieee123data):
    topology = get_topology(ieee123data)
    measurements = get_measurements(ieee123data)
    X0, z, num_node, knownP, knownQ, knownV, Y = inner_args(
        parameters, topology, measurements
    )
    h = residual(X0, z, num_node, knownP, knownQ, knownV, Y)
    assert h.shape == (len(knownP) + len(knownQ) + len(knownV),)
    assert isinstance(h, np.ndarray), f"h has type {type(h)}"
    assert not isinstance(h, np.matrix), f"h has type {type(h)}"


def test_residual_sparse(parameters, sparse_topology, ieee123data):
    measurements = get_measurements(ieee123data)
    X0, z, num_node, knownP, knownQ, knownV, Y = inner_args(
        parameters, sparse_topology, measurements
    )
    h = residual(X0, z, num_node, knownP, knownQ, knownV, Y)
    assert h.shape == (len(knownP) + len(knownQ) + len(knownV),)
    assert isinstance(h, np.ndarray), f"h has type {type(h)}"
    assert not isinstance(h, np.matrix), f"h has type {type(h)}"


@pytest.mark.parametrize("input_data", INPUT_DATA)
def test_residuals_against_actuals(parameters, input_data):
    topology = get_topology(input_data)
    measurements = get_measurements(input_data)
    actuals = get_actuals(input_data)
    worse_X0, z, num_node, knownP, knownQ, knownV, Y = inner_args(
        parameters, topology, measurements
    )
    voltage_real, voltage_imag = actuals
    true_voltages = np.array(voltage_real.values) + 1j * np.array(voltage_imag.values)
    true_voltages /= np.array(topology.base_voltage_magnitudes.values)

    X0 = np.concatenate((np.angle(true_voltages), np.abs(true_voltages)))
    h = residual(X0, z, num_node, knownP, knownQ, knownV, Y)
    voltage_ids = list(map(lambda x: "voltage_" + x, measurements[2].ids))
    power_real_ids = list(
        map(
            lambda x: "P_" + x,
            set(measurements[0].ids).union(get_zero_injection_indices(topology)),
        )
    )
    power_imag_ids = list(
        map(
            lambda x: "Q_" + x,
            set(measurements[1].ids).union(get_zero_injection_indices(topology)),
        )
    )
    ids = voltage_ids + power_real_ids + power_imag_ids
    assert len(ids) == len(
        h
    ), f"Residuals are of length {len(h)} whereas there are {len(ids)} measurements"

    # baseline_difference = np.abs(
    #     residual(worse_X0, z, num_node, knownP, knownQ, knownV, Y)
    # ) >= np.abs(h)
    # (baseline_exceptions,) = np.nonzero(~baseline_difference)
    # baseline_exception_ids = [ids[idx] for idx in baseline_exceptions]
    # assert np.all(
    #     baseline_difference
    # ), f"Initial set point better for ids {baseline_exception_ids}"

    (idx_above_max,) = np.nonzero(np.abs(h) * parameters.base_power > 10)
    ids_above_max = [ids[idx] for idx in idx_above_max]
    # h has units of base_power.
    max_power_watts = np.max(np.abs(h)) * parameters.base_power
    assert (
        max_power_watts < 10
    ), f"Residuals are too high: max {max_power_watts} at {ids_above_max}"


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


def get_mean_angle_error(topology, solution, actuals):
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

    return np.abs(np.angle(estimated_voltage * true_voltage.conj())).mean()


@pytest.mark.parametrize("input_data", INPUT_DATA)
def test_least_squares_call(parameters, input_data):
    topology = get_topology(input_data)
    measurements = get_measurements(input_data)
    actuals = get_actuals(input_data)
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

    mean_angle_error = get_mean_angle_error(topology, solution, actuals)
    assert (
        mean_angle_error < 3 * np.pi / 180
    ), f"Max angle error too high: {mean_angle_error * 180 / np.pi} degrees"


def test_compare_initial_conditions(parameters, ieee123data, sparse_topology):
    topology = get_topology(ieee123data)
    measurements = get_measurements(ieee123data)
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


def test_compare_jacobian_residuals_vs_sparse(parameters, ieee123data, sparse_topology):
    topology = get_topology(ieee123data)
    measurements = get_measurements(ieee123data)
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


def test_least_squares_call_sparse(parameters, sparse_topology, ieee123data):
    measurements = get_measurements(ieee123data)
    actuals = get_actuals(ieee123data)
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

    mean_angle_error = get_mean_angle_error(sparse_topology, solution, actuals)
    assert (
        mean_angle_error < 3 * np.pi / 180
    ), f"Max angle error too high: {mean_angle_error * 180 / np.pi} degrees"


@pytest.mark.parametrize("input_data", INPUT_DATA)
def test_least_squares_with_perfect_initalization(parameters, input_data):
    topology = get_topology(input_data)
    measurements = get_measurements(input_data)
    actuals = get_actuals(input_data)
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

    mean_angle_error = get_mean_angle_error(topology, solution, actuals)
    assert (
        mean_angle_error < 1 * np.pi / 180
    ), f"Max angle error too high: {mean_angle_error * 180 / np.pi} degrees"


@pytest.mark.parametrize("input_data", INPUT_DATA)
def test_wls_agreement_with_yuqi(parameters, input_data):
    topology = get_topology(input_data)
    measurements = get_measurements(input_data)
    actuals = get_actuals(input_data)
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
    assert ls_result.success, f"Least squares failed: {ls_result.message}"

    solution = ls_result.x

    voltage_real, voltage_imag = actuals
    true_voltage = np.array(voltage_real.values) + 1j * np.array(voltage_imag.values)
    true_voltage /= np.array(topology.base_voltage_magnitudes.values)

    vmagestDecen, vangestDecen = (
        solution[len(solution) // 2 :],
        solution[: len(solution) // 2],
    )

    slack_id = topology.base_voltage_magnitudes.ids.index(topology.slack_bus[0])
    vangestDecen = vangestDecen - vangestDecen[slack_id]

    estimate_voltage = vmagestDecen * np.exp(1j * vangestDecen)

    mean_mag_error = np.mean(np.abs(np.abs(true_voltage) - np.abs(estimate_voltage)))
    mean_angle_error = np.mean(
        np.abs(np.angle(true_voltage) - np.angle(estimate_voltage))
    )

    if input_data == "small_smartds_tap_time_3":
        assert (
            np.abs(mean_mag_error - 0.0273) < 0.00001
        ), f"Max relative error too high: {mean_mag_error}"
        assert np.abs(
            mean_angle_error - 0.0022
        ), f"Max angle error too high: {mean_angle_error * 180 / np.pi} degrees"
    elif input_data == "small_smartds_tap_time_40":
        assert (
            np.abs(mean_mag_error - 0.0307) < 0.00001
        ), f"Max relative error too high: {mean_mag_error}"
        assert (
            np.abs(mean_angle_error - 0.1656) < 0.00001
        ), f"Max angle error too high: {mean_angle_error * 180 / np.pi} degrees"


@pytest.mark.parametrize("input_data", INPUT_DATA)
def test_mean_absolute_error_least_squares(parameters, input_data):
    topology = get_topology(input_data)
    measurements = get_measurements(input_data)
    actuals = get_actuals(input_data)
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
    assert ls_result.success, f"Least squares failed: {ls_result.message}"

    solution = ls_result.x

    voltage_real, voltage_imag = actuals
    true_voltage = np.array(voltage_real.values) + 1j * np.array(voltage_imag.values)
    true_voltage /= np.array(topology.base_voltage_magnitudes.values)

    vmagestDecen, vangestDecen = (
        solution[len(solution) // 2 :],
        solution[: len(solution) // 2],
    )

    slack_id = topology.base_voltage_magnitudes.ids.index(topology.slack_bus[0])
    vangestDecen = vangestDecen - vangestDecen[slack_id]

    estimate_voltage = vmagestDecen * np.exp(1j * vangestDecen)

    mean_mag_error = np.mean(np.abs(np.abs(true_voltage) - np.abs(estimate_voltage)))
    mean_angle_error = np.mean(np.abs(np.angle(true_voltage * estimate_voltage.conj())))

    assert mean_mag_error < 0.04, f"Max relative error too high: {mean_mag_error}"
    assert mean_angle_error < 0.04, f"Max angle error too high: {mean_angle_error}"
