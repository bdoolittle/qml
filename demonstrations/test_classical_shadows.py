import pytest
import pennylane as qml
import numpy as np
import time

from classical_shadows_lib import calculate_classical_shadow, estimate_shadow_obervable, shadow_state_reconstruction
from classical_shadows_utils import operator_2_norm, shadow_bound


@pytest.fixture
def circuit_1_observable(request):
    """Circuit with single layer requiring nqubits parameters"""
    num_qubits = request.param
    dev = qml.device('default.qubit', wires=num_qubits, shots=1)

    @qml.qnode(device=dev)
    def circuit(params, **kwargs):
        observables = kwargs.pop('observable')
        for w in dev.wires:
            qml.Hadamard(wires=w)
            qml.RY(params[w], wires=w)
        return [qml.expval(o) for o in observables]

    param_shape = (None,)
    return circuit, param_shape, num_qubits


@pytest.fixture
def circuit_1_state(request):
    """Circuit with single layer requiring nqubits parameters"""
    num_qubits = request.param
    dev = qml.device('default.qubit', wires=num_qubits)

    @qml.qnode(device=dev)
    def circuit(params, **kwargs):
        for w in dev.wires:
            qml.Hadamard(wires=w)
            qml.RY(params[w], wires=w)
        return qml.state()

    param_shape = (None,)
    return circuit, param_shape, num_qubits


@pytest.fixture
def circuit_2_observable(request):
    """Circuit with multiple layers requiring nqubits*3 parameters"""
    num_qubits = request.param
    dev = qml.device('default.qubit', wires=num_qubits, shots=1)

    @qml.qnode(device=dev)
    def circuit(params, **kwargs):
        observables = kwargs.pop('observable')
        for w in dev.wires:
            qml.Hadamard(wires=w)
            qml.RY(params[w, 0], wires=w)
        for layer in range(2):
            for w in dev.wires:
                qml.RX(params[w, layer + 1], wires=w)
        return [qml.expval(o) for o in observables]

    param_shape = (None, 3)
    return circuit, param_shape, num_qubits


# TODO: Do both circuit_1 and circuit_2 in the same test.
# construct circuit 1 with a different number of qubits.
@pytest.mark.parametrize("circuit_1_observable", [1, 2, 3, 4], indirect=True)
def test_calculate_classical_shadow_circuit_1(circuit_1_observable, shadow_size=10):
    """Test calculating the shadow for a simple circuit with a single layer"""
    circuit_template, param_shape, num_qubits = circuit_1_observable
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    outcomes = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)

    assert all(o in [1.0, -1.0] for o in np.unique(outcomes[:, :num_qubits]))


# construct circuit 1 with a different number of qubits.
@pytest.mark.parametrize("circuit_2_observable", [1, 2, 3, 4], indirect=True)
def test_calculate_classical_shadow_circuit_2(circuit_2_observable, shadow_size=10):
    """Test calculating the shadow for a circuit with multiple layers"""
    circuit_template, param_shape, num_qubits = circuit_2_observable
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    shadow = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)

    assert all(o in [1.0, -1.0] for o in np.unique(shadow[:, :num_qubits]))


@pytest.mark.parametrize("circuit_1_observable, shadow_size", [[2, 10], [2, 100], [2, 1000], [2, 10000]],
                         indirect=['circuit_1_observable'])
def test_calculate_classical_shadow_performance(circuit_1_observable, shadow_size):
    """Performance test calculating the shadow for a circuit with multiple layers"""
    circuit_template, param_shape, num_qubits = circuit_1_observable
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    start = time.time()
    calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)
    delta_time = time.time() - start
    print(f'Elapsed time for {shadow_size} shadows = {delta_time}')


# TODO: create a fixture for the shadow so we only have run it once
@pytest.mark.parametrize("circuit_1_observable, shadow_size", [[2, 10], [2, 100], [2, 1000], [2, 10000]],
                         indirect=['circuit_1_observable'])
def test_estimate_shadow_observable_single(circuit_1_observable, shadow_size):
    """Test calculating an observable with the shadow for a circuit with a single layer"""
    circuit_template, param_shape, num_qubits = circuit_1_observable
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    shadow = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)

    observable = [qml.PauliX(0) @ qml.PauliX(1)]
    expval_shadow = estimate_shadow_obervable(shadow, observable[0])
    assert -1.0 <= expval_shadow <= 1.0
    dev_exact = qml.device('default.qubit', wires=num_qubits)
    # change the simulator to be the exact one.
    circuit_template.device = dev_exact
    expval_exact = circuit_template(params, observable=observable)
    print(f"Shadow : {expval_shadow} - Exact {expval_exact}")


@pytest.mark.parametrize("circuit_2_observable, shadow_size", [[8, 1000], [8, 10000]],
                         indirect=['circuit_2_observable'])
def test_estimate_shadow_observable_multi_1(circuit_2_observable, shadow_size):
    """Test calculating multiple observables with the shadow for a circuit with multiple layers"""
    circuit_template, param_shape, num_qubits = circuit_2_observable
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    shadow = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)

    observable = [qml.PauliX(0) @ qml.PauliZ(num_qubits - 2) @ qml.PauliY(num_qubits - 1)]
    expval_shadow = estimate_shadow_obervable(shadow, observable[0])
    assert -1.0 <= expval_shadow <= 1.0
    dev_exact = qml.device('default.qubit', wires=num_qubits)
    # change the simulator to be the exact one.
    circuit_template.device = dev_exact
    expval_exact = circuit_template(params, observable=observable)
    print(f"Shadow : {expval_shadow} - Exact {expval_exact}")


@pytest.mark.parametrize("circuit_2_observable", [8], indirect=['circuit_2_observable'])
def test_estimate_shadow_observable_shadow_bound(circuit_2_observable):
    """Test calculating multiple observables with the shadowsize shadow_size given by the bound in the paper"""
    circuit_template, param_shape, num_qubits = circuit_2_observable
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    list_of_observables = [qml.PauliX(0) @ qml.PauliZ(2) @ qml.PauliY(4),
                           qml.PauliY(1) @ qml.PauliZ(2),
                           qml.PauliX(0),
                           qml.PauliY(3) @ qml.PauliZ(4) @ qml.PauliY(num_qubits - 1)]
    # Calculate how many shadows we need to get an error of 1e-1 for this set of observables
    shadow_size = shadow_bound(M=len(list_of_observables), max_k=3,
                               error=1e-1, observables=[o.matrix for o in list_of_observables])

    shadow = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)

    expval_shadow = sum(estimate_shadow_obervable(shadow, o) for o in list_of_observables)
    dev_exact = qml.device('default.qubit', wires=num_qubits)
    # change the simulator to be the exact one.
    circuit_template.device = dev_exact
    expval_exact = sum(circuit_template(params, observable=[o, ]) for o in list_of_observables)
    print(f"Shadow : {expval_shadow} - Exact {expval_exact}")
    # from the theoretical bound we know that the variance of this estimator must be within 1e-1
    assert np.isclose(expval_shadow, expval_exact, atol=1e-1)


@pytest.mark.parametrize("circuit_1_observable, circuit_1_state, shadow_size",
                         [[2, 2, 1000], [2, 2, 10000], [2, 2, 50000]], indirect=['circuit_1_observable',
                                                                                 'circuit_1_state'])
def test_shadow_state_reconstruction(circuit_1_observable, circuit_1_state, shadow_size):
    """Test reconstructing the state from the shadow for a circuit with a single layer"""
    circuit_template, param_shape, num_qubits = circuit_1_observable
    circuit_template_state, _, _ = circuit_1_state
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    shadow = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)

    state_shadow = shadow_state_reconstruction(shadow)
    dev_exact = qml.device('default.qubit', wires=num_qubits)
    circuit_template.device = dev_exact
    state_exact = circuit_template_state(params, observable=[qml.state()])
    state_exact = np.outer(state_exact, state_exact.conj())

    print(operator_2_norm(state_shadow - state_exact))
