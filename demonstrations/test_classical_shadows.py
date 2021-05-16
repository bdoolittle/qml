import pytest
import pennylane as qml
import numpy as np
import time

from classical_shadows_lib import calculate_classical_shadow, estimate_shadow_obervable


@pytest.fixture
def circuit_1(request):
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
def circuit_2(request):
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


# @pytest.fixture
def circuit_1_shadow(num_qubits, num_shadows):
    dev = qml.device('default.qubit', wires=num_qubits, shots=1)

    @qml.qnode(device=dev)
    def circuit(params, **kwargs):
        observables = kwargs.pop('observable')
        for w in dev.wires:
            qml.Hadamard(wires=w)
            qml.RY(params[w], wires=w)
        return [qml.expval(o) for o in observables]

    params = np.random.randn(num_qubits)
    return calculate_classical_shadow(circuit, params, num_shadows, num_qubits), params


# construct circuit 1 with a different number of qubits.
@pytest.mark.parametrize("circuit_1", [1, 2, 3, 4], indirect=True)
def test_calculate_classical_shadow_circuit_1(circuit_1, num_shadows=10):
    """Test calculating the shadow for a simple circuit with a single layer"""

    circuit_template, param_shape, num_qubits = circuit_1
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    outcomes = calculate_classical_shadow(circuit_template, params, num_shadows, num_qubits)

    assert all(o in [1.0, -1.0] for o in np.unique(outcomes[:, :num_qubits]))


# construct circuit 1 with a different number of qubits.
@pytest.mark.parametrize("circuit_2", [1, 2, 3, 4], indirect=True)
def test_calculate_classical_shadow_circuit_2(circuit_2, num_shadows=10):
    """Test calculating the shadow for a circuit with multiple layers"""
    circuit_template, param_shape, num_qubits = circuit_2
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    shadow = calculate_classical_shadow(circuit_template, params, num_shadows, num_qubits)

    assert all(o in [1.0, -1.0] for o in np.unique(shadow[:, :num_qubits]))


@pytest.mark.parametrize("circuit_1, num_shadows", [[2, 10], [2, 100], [2, 1000], [2, 10000]], indirect=['circuit_1'])
def test_calculate_classical_shadow_performance(circuit_1, num_shadows):
    """Test calculating the shadow for a circuit with multiple layers"""
    circuit_template, param_shape, num_qubits = circuit_1
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    start = time.time()
    calculate_classical_shadow(circuit_template, params, num_shadows, num_qubits)
    delta_time = time.time() - start
    print(f'Elapsed time for {num_shadows} shadows = {delta_time}')

@pytest.mark.parametrize("circuit_1, num_shadows", [[2, 10], [2, 100], [2, 1000], [2, 10000]], indirect=['circuit_1'])
def test_estimate_shadow_observable_X(circuit_1, num_shadows):
    """Test calculating the shadow for a circuit with multiple layers"""
    circuit_template, param_shape, num_qubits = circuit_1
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    shadow = calculate_classical_shadow(circuit_template, params, num_shadows, num_qubits)

    observable = [(1, i) for i in range(2)]
    expval_shadow = estimate_shadow_obervable(shadow, observable)
    dev_exact = qml.device('default.qubit', wires=num_qubits)
    circuit_template.device = dev_exact
    expval_exact = circuit_template(params, observable=[qml.PauliX(0) @ qml.PauliX(1)])
    print(expval_shadow)
    print(expval_exact)


# if __name__ == '__main__':
#     test_estimate_shadow_obervable(2, 1000)
