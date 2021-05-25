import pennylane as qml
import numpy as np


def calculate_classical_shadow(circuit_template, params, shadow_size: int, num_qubits: int) -> np.ndarray:
    """
    Given a circuit, creates a collection of snapshots U^\dag|b><b| U with the stabilizer description.
    
    Args:
        circuit_template: A Pennylane QNode.
        params: Circuit parameters.
        shadow_size: The number of snapshots in the shadow.
        num_qubits: The number of qubits in the circuit.

    Returns:
        Numpy array containing the outcomes (0, 1) in the first `num_qubits` columns and the sampled Pauli's
        (0,1,2=x,y,z) in the final `num_qubits` columns.

    """
    unitary_ensemble = [qml.PauliX, qml.PauliY, qml.PauliZ]
    # each shadow is one shot, so we set this parameter in the qml.device
    # sample random pauli unitaries uniformly, where 1,2,3 = X,Y,Z
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, num_qubits))
    outcomes = np.zeros((shadow_size, num_qubits))
    for ns in range(shadow_size):
        # for each shadow, add a random Clifford observable at each location
        obs = [unitary_ensemble[int(unitary_ids[ns, i])](i) for i in range(num_qubits)]
        outcomes[ns, :] = circuit_template(params, observable=obs)
    # combine the computational basis outcomes and the sampled unitaries
    return np.concatenate([outcomes, unitary_ids], axis=1)


def estimate_shadow_obervable(shadows, observable) -> float:
    """
    Calculate the estimator E[O] = sum_i Tr{rho_i O} where rho_i is a snapshot in the shadow.

    Args:
        shadows: Numpy array containing the outcomes (0, 1) in the first `num_qubits` columns and the sampled Pauli's
        (0,1,2=x,y,z) in the final `num_qubits` columns.
        observable: Single PennyLane observable consisitng of single Pauli operators e.g. qml.PauliX(0) @ qml.PauliY(1)

    Returns:
        Scalar corresponding to the estimate of the observable.
    """
    map_name_to_int = {'PauliX': 0, 'PauliY': 1, 'PauliZ': 2}
    if isinstance(observable, (qml.PauliX, qml.PauliY, qml.PauliZ)):
        observable_as_list = [(map_name_to_int[observable.name], observable.wires[0])]
    else:
        observable_as_list = [(map_name_to_int[o.name], o.wires[0]) for o in observable.obs]
    num_qubits = shadows.shape[1] // 2
    sum_product, cnt_match = 0, 0
    # loop over the shadows:
    for single_measurement in shadows:
        not_match = 0
        product = 1
        # loop over all the paulis that we care about
        for pauli_XYZ, position in observable_as_list:
            # if the pauli in our shadow does not match, we break and go to the next shadow
            if pauli_XYZ != single_measurement[position + num_qubits]:
                not_match = 1
                break
            product *= single_measurement[position]
        # don't record the shadow
        if not_match == 1: continue

        sum_product += product
        cnt_match += 1
    if cnt_match == 0:
        return 0
    else:
        return sum_product / cnt_match


def snapshot_state(b_list, obs_list):
    """
    Reconstruct a state approximation from a single snapshot in a shadow.

    **Details:**

    Implements Eq. (S44) from https://arxiv.org/pdf/2002.08953.pdf

    Args:
        b_list (array): classical outcomes for a single sample
        obs_list (array): ids for the pauli observable used for each measurement
    """

    num_qubits = len(b_list)

    paulis = [
        qml.Hadamard(0).matrix,

        qml.Hadamard(0).matrix @ np.array([[1, 0],
                                           [0, -1j]], dtype=np.complex),
        qml.Identity(0).matrix
    ]

    zero_state = np.array([[1, 0], [0, 0]])
    one_state = np.array([[0, 0], [0, 1]])

    rho_snapshot = [1]
    for i in range(num_qubits):
        state = zero_state if b_list[i] == 1 else one_state
        U = paulis[int(obs_list[i])]

        local_rho = 3 * (U.conj().T @ state @ U) - np.eye(2, 2)

        rho_snapshot = np.kron(rho_snapshot, local_rho)

    return rho_snapshot


def shadow_state_reconstruction(shadow):
    """
    Reconstruct a state approximation as an average over all snapshots in the shadow.
    """

    num_shadows = shadow.shape[0]
    num_qubits = shadow.shape[1] // 2

    b_lists = shadow[:, 0:num_qubits]
    obs_lists = shadow[:, num_qubits:2 * num_qubits]
    # state approximated from snapshot average
    shadow_rho = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=np.complex)
    for i in range(num_shadows):
        shadow_rho += snapshot_state(b_lists[i], obs_lists[i])

    return shadow_rho / num_shadows
