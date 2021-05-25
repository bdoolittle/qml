r"""
Classical Shadows
=================
.. meta::
    :property="og:description": Learn how to efficiently make predictions about an unkown 
        quantum state using the classical shadow approximation. 
    :property="og:image": https://pennylane.ai/qml/_images/qaoa_layer.png

.. related::

*Authors: Roeland Wiersema & Brian Doolittle (Xanadu Residents).
Posted: 7 May 2021. Last updated: 7 May 2021.*

Estimating properties of unknown quantum states is a key objective of quantum
information science and technology.
For example, one might want to check whether an apparatus prepares a particular target state,
or verify that an unknown system can prepare entangled states.
In principle, any unknown quantum state can be fully characterized by performing `quantum state
tomography <http://research.physics.illinois.edu/QI/Photonics/tomography-files/tomo_chapter_2004.pdf>`_
However this procedure requires one to acquire accurate expectation values for a set of observables
which grows exponentially with the number of qubits.
A potential workaround for these scaling conderns is provided by classical shadow approximation
introduced in `Predicting many properties of a quantum system from very few measurements
<https://arxiv.org/pdf/2002.08953.pdf>`_ [Huang2020]_.

The classical shadow approximation is an efficient protocol for constructing a *classical shadow*
representation of an unknown quantum state.
The classical shadow can then be used to estimate
quantum state fidelity, Hamilton eigenvalues, two-point correlators, and many other properties.

.. figure:: ../demonstrations/classical_shadows/classical_shadow_overview.png
    :align: center
    :width: 80%

    (Image from Huang et al. [Huang2020]_.)

In this demo, we will show how to construct classical shadows and use them to approximate
properties of quantum states.
To begin, we first need to import ``pennylane`` and ``numpy``
"""

import pennylane as qml
import pennylane.numpy as np

##############################################################################
#
# A classical shadow consists of an integer number `N` *snapshots* which are constructed
# via the following proce
#
# 1. A quantum state :math:`\rho` is prepared.
# 2. A randomly selected unitary :math:`U` is applied
# 3. A computational basis measurement is performed.
# 4. The process is repeated :math:`N` times.
#
# The classical shadow is then constructed as a list of measurement outcomes and chosen unitaries.


def classical_shadow(circuit_template, params, num_shadows: int) -> np.ndarray:
    unitary_ensemble = [qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ]
    # each shadow is one shot, so we set this parameter in the qml.device
    device_shadow = qml.device('default.qubit', wires=nqubits, shots=1)
    # sample random unitaries uniformly, where 0 = I and 1,2,3 = X,Y,Z
    randints = np.random.randint(1, 4, size=(number_of_shadows, nqubits))
    unitaries = []
    for ns in range(num_shadows):
        # for each shadow, add a random Clifford observable at each location
        unitaries.extend(unitary_ensemble[int(randints[ns, i])](i) for i in range(nqubits))
    # use the QNodeCollections through qml.map to calculate multiple observables
    # for the same circuit
    samples = qml.map(circuit_template, observables=unitaries,
                      device=device_shadow, measure='sample')(params)
    # reshape from dim 1 vector to matrix, each row now corresponds to a shadow
    samples = samples.reshape((number_of_shadows, nqubits))
    # combine the computational basis outcomes and the sampled unitaries
    return np.concatenate([samples, randints], axis=1)

##############################################################################
#
# Any pauli observable can estimated by using the ``estimate_shadow_observable`` method

def estimate_shadow_obervable(shadows, observable):
    # https://github.com/momohuang/predicting-quantum-properties
    sum_product, cnt_match = 0, 0
    # loop over the shadows:
    for single_measurement in shadows:
        not_match = 0
        product = 1
        # loop over all the paulis that we care about
        for pauli_XYZ, position in observable:
            # if the pauli in our shadow does not match, we break and go to the next shadow
            if pauli_XYZ != single_measurement[position + nqubits]:
                not_match = 1
                break
            product *= single_measurement[position]
        # don't record the shadow
        if not_match == 1: continue

        sum_product += product
        cnt_match += 1

    return sum_product / cnt_match

##############################################################################
# Here, we show a simple example of how the observable :math:`X_0' can be estimated with
# classical whadows.

nqubits = 1
theta = np.random.randn(nqubits, )
device_exact = qml.device('default.qubit', wires=nqubits)
number_of_shadows = 1000

paulis = {0: qml.Identity(0).matrix,
          1: qml.PauliX(0).matrix,
          2: qml.PauliY(0).matrix,
          3: qml.PauliZ(0).matrix}


def circuit(params, wires, **kwargs):
    for i in range(nqubits):
        qml.Hadamard(wires=i)
        qml.RY(params[i], wires=i)


# Estimate Tr{X_0 rho}
exact_observable = qml.map(circuit, observables=[qml.PauliX(0)],
                           device=device_exact, measure='expval')(theta)
print(f"Exact value: {exact_observable[0]}")
shadows = classical_shadow(circuit, theta, number_of_shadows)
# The observable X_0 is passed as [(1,0), ] something like X_0 Y_1 would be  [(1,0),(2,1)]
# TODO: Generalize the mapping from Pauli strings to tuples
shadow_observable = estimate_shadow_obervable(shadows, [(1, 0), ])
print(f"Shadow value: {shadow_observable}")
shadows

##############################################################################
# Next, we consider the multi-qubit observable :math:`X_0 X_1`.


nqubits = 2
theta = np.random.randn(nqubits, )
device_exact = qml.device('default.qubit', wires=nqubits)
number_of_shadows = 3000

exact_observable = qml.map(circuit, observables=[qml.PauliX(0) @ qml.PauliX(1)],
                           device=device_exact, measure='expval')(
    theta)
print(f"Exact value: {exact_observable[0]}")
shadows = classical_shadow(circuit, theta, number_of_shadows)
shadow_observable = estimate_shadow_obervable(shadows, [(1, 0), (1, 1)])
print(f"Shadow value: {shadow_observable}")

exact_observable

##############################################################################
# Comparison of standard observable estimators and classical shadows.


##############################################################################
# State Reconstruction using Classical Shadow
# ###########################################

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
        qml.Identity(0).matrix,
        qml.PauliX(0).matrix,
        qml.PauliY(0).matrix,
        qml.PauliZ(0).matrix
    ]

    zero_state = np.array([[1,0],[0,0]])
    one_state = np.array([[0,0],[0,1]])
    
    rho_snapshot = [1]
    for i in range(num_qubits):
        state = zero_state if b_list[i] == 1 else one_state
        U = paulis[obs_list[i]]
        
        local_rho = 3 * (U.conj().T @ state @ U) - paulis[0]
        
        rho_snapshot = np.kron(rho_snapshot, local_rho)
    
    return rho_snapshot

def shadow_state(shadow):
    """
    Reconstruct a state approximation as an average over all snapshots in the shadow.
    """

    num_shadows = shadow.shape[0]
    num_qubits = int(shadow.shape[1]/2)
    
    b_lists = shadow[:,0:num_qubits]
    obs_lists = shadow[:,num_qubits:2*num_qubits]
    
    # state approximated from snapshot average
    shadow_rho = np.zeros((2**num_qubits, 2**num_qubits))
    for i in range(num_shadows):
        snapshot = snapshot_state(b_lists[i], obs_lists[i])
        shadow_rho = shadow_rho + snapshot
    
    return shadow_rho/num_shadows


##############################################################################
# Example: |+> state is approximated as maximally mixed state

nqubits = 1
theta = [np.pi/4]
number_of_shadows = 500

def one_qubit_RY(params, wires, **kwargs):
    qml.RY(params[0], wires=wires[0])

shadow = classical_shadow(one_qubit_RY, theta, number_of_shadows)

state_reconstruction = shadow_state(shadow)

print("state reconstruction")

print(state_reconstruction)



#TODO: what is a good application?

##############################################################################
# .. [Huang2020] Huang, Hsin-Yuan, Richard Kueng, and John Preskill.
#             "Predicting many properties of a quantum system from very few measurements."
#             Nature Physics 16.10 (2020): 1050-1057.

