import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Define basis states: |0> = Stau, |1> = Free
psi0 = qt.basis(2, 0)  # Stau
psi1 = qt.basis(2, 1)  # Free

# Initial entangled state for 3 qubits: Bell-state extended to GHZ-like for correlated traffic
# (1/sqrt(2)) * (|000> + |111>)
ghz_state = (qt.tensor(psi0, psi0, psi0) + qt.tensor(psi1, psi1, psi1)).unit()

# Apply Hadamard to first qubit for superposition (uncertainty in traffic start)
h_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
h_gate = qt.Qobj(h_matrix, dims=[[2], [2]])
h_on_first = qt.tensor(h_gate, qt.qeye(2), qt.qeye(2))
superposed_state = h_on_first * ghz_state

# Apply CNOT chain: Qubit 1 controls 2, 2 controls 3 (cascade effect)
cnot12_matrix = np.array([[1,0,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0,0],
                          [0,0,1,0,0,0,0,0],
                          [0,0,0,1,0,0,0,0],
                          [0,0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,1],
                          [0,0,0,0,1,0,0,0],
                          [0,0,0,0,0,1,0,0]])  # Custom CNOT 0->1 for 3 qubits
cnot12 = qt.Qobj(cnot12_matrix, dims=[[2,2,2], [2,2,2]])
cnot23_matrix = np.array([[1,0,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0,0],
                          [0,0,1,0,0,0,0,0],
                          [0,0,0,0,0,1,0,0],
                          [0,0,0,0,1,0,0,0],
                          [0,0,0,1,0,0,0,0],
                          [0,0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,1]])  # Custom CNOT 1->2
cnot23 = qt.Qobj(cnot23_matrix, dims=[[2,2,2], [2,2,2]])
entangled_state = cnot23 * cnot12 * superposed_state

# Convert to density matrix
rho = entangled_state * entangled_state.dag()

# Add depolarizing noise (real-world randomness, e.g., weather/accidents)
p_depol = 0.05  # 5% depolarization probability per qubit
depol_op_single = [np.sqrt(1 - 3*p_depol/4) * qt.qeye(2),
                   np.sqrt(p_depol/4) * qt.sigmax(),
                   np.sqrt(p_depol/4) * qt.sigmay(),
                   np.sqrt(p_depol/4) * qt.sigmaz()]
depol_ops = [qt.tensor(k1, k2, k3) for k1 in depol_op_single for k2 in depol_op_single for k3 in depol_op_single]

# Apply noise
noisy_state = sum([op * rho * op.dag() for op in depol_ops])

# Measurements: Probabilities for all 8 states (|000> to |111>)
measurement_ops = [qt.tensor(qt.basis(2, i).proj(), qt.basis(2, j).proj(), qt.basis(2, k).proj()) for i in range(2) for j in range(2) for k in range(2)]
probs = [qt.expect(op, noisy_state) for op in measurement_ops]

print("Probabilities for 3-Ampel States:")
for idx, state in enumerate(['000', '001', '010', '011', '100', '101', '110', '111']):
    print(f"|{state}> :", probs[idx])

# Time evolution: Simulate traffic flow as a Hamiltonian (e.g., interaction between amps)
H = qt.tensor(qt.sigmax(), qt.sigmax(), qt.qeye(2)) + qt.tensor(qt.sigmax(), qt.qeye(2), qt.sigmax())  # X0*X1 + X0*X2
times = np.linspace(0, np.pi/2, 10)  # Evolve over time
result = qt.mesolve(H, noisy_state, times)  # Master equation solver (no collapse for simplicity)

# Prob at final time
final_state = result.states[-1]
final_probs = [qt.expect(op, final_state) for op in measurement_ops]

print("\nProbabilities after time evolution:")
for idx, state in enumerate(['000', '001', '010', '011', '100', '101', '110', '111']):
    print(f"|{state}> :", final_probs[idx])

# Add collapse operators for dissipation: Favor |1> by relaxing towards free flow (e.g., traffic optimization)
gamma = 0.1  # Dissipation rate
c_ops = [np.sqrt(gamma) * qt.tensor(qt.sigmap(), qt.qeye(2), qt.qeye(2)),
         np.sqrt(gamma) * qt.tensor(qt.qeye(2), qt.sigmap(), qt.qeye(2)),
         np.sqrt(gamma) * qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmap())]

# Evolve with dissipation
result_diss = qt.mesolve(H, noisy_state, times, c_ops=c_ops)

# Final probs with dissipation
final_diss_state = result_diss.states[-1]
final_diss_probs = [qt.expect(op, final_diss_state) for op in measurement_ops]

print("\nProbabilities after time evolution with dissipation:")
for idx, state in enumerate(['000', '001', '010', '011', '100', '101', '110', '111']):
    print(f"|{state}> :", final_diss_probs[idx])

# Optimization loop: Find time with minimal |000>
min_stau = float('inf')
best_time = 0
for t_idx, state in enumerate(result_diss.states):
    stau_prob = qt.expect(measurement_ops[0], state)  # |000>
    if stau_prob < min_stau:
        min_stau = stau_prob
        best_time = times[t_idx]
print(f"\nOptimal time for minimal Stau (|000>): t={best_time:.2f}, Prob={min_stau:.4f}")

# Gamification: Score based on minimal Stau
score = int(100 * (1 - min_stau))
print(f"Your Traffic Optimization Score: {score}/100")
if score > 80:
    print("Level Complete: Unlock next qubit!")
else:
    print("Try again: Increase gamma or adjust H.")

# Visualize Bloch spheres for each qubit
rho0 = noisy_state.ptrace(0)
rho1 = noisy_state.ptrace(1)
rho2 = noisy_state.ptrace(2)

fig = plt.figure(figsize=(15, 5))
for i, rho in enumerate([rho0, rho1, rho2]):
    b = qt.Bloch(fig=fig, axes=fig.add_subplot(131 + i, projection='3d'))
    b.add_states(rho)
    b.render()
    b.axes.set_title(f'Ampel {i+1} Bloch Sphere')

plt.savefig('bloch_verkehr_3q.png')  # Updated filename for 3 qubits
plt.show()
