import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Zwei-Qubit-System: Qubit 1 = Ampel A (|0> Stau, |1> Fluss), Qubit 2 = Ampel B
# Start: Verschränkter Bell-Zustand (max. Entanglement: Zustände korreliert)
bell_state = (qt.basis(2, 0) * qt.basis(2, 0) + qt.basis(2, 1) * qt.basis(2, 1)).unit()

# Hamilton-Operator: Interaktion zwischen Ampeln (z. B. J*sigma_x1 * sigma_x2 für Kopplung)
omega = 1.0  # Kopplungsstärke
H = 2 * np.pi * omega * (qt.tensor(qt.sigmax(), qt.sigmax())) / 2

# Noise: Bit-Flip-Channel (z. B. für Verkehrs-Störungen wie Unfälle)
gamma_bitflip = 0.05  # Error-Rate
c_ops = [
    np.sqrt(gamma_bitflip) * qt.tensor(qt.sigmax(), qt.qeye(2)),  # Bit-Flip auf Qubit 1
    np.sqrt(gamma_bitflip) * qt.tensor(qt.qeye(2), qt.sigmax())   # Bit-Flip auf Qubit 2
]

# Zeiten für Evolution
tlist = np.linspace(0, 10, 200)

# Löse Master-Gleichung (mit Noise)
result = qt.mesolve(H, bell_state, tlist, c_ops, [
    qt.tensor(qt.sigmax(), qt.qeye(2)),  # Erwartungswert X für Qubit 1
    qt.tensor(qt.qeye(2), qt.sigmax())   # Erwartungswert X für Qubit 2
])

# Plot Erwartungswerte (Evolution der Zustände)
fig, ax = plt.subplots()
ax.plot(tlist, result.expect[0], label='Ampel A (X)')
ax.plot(tlist, result.expect[1], label='Ampel B (X)')
ax.set_xlabel('Zeit')
ax.set_ylabel('Erwartungswert')
ax.legend()
plt.savefig('verkehr_entanglement.png')  # Für Artifact-Upload
plt.close(fig)

# Messung: Berechne Wahrscheinlichkeit für Stau an beiden Ampeln am Ende
final_state = result.states[-1]
proj_stau_both = qt.tensor(qt.basis(2, 0), qt.basis(2, 0)).proj()
prob_stau_both = qt.expect(proj_stau_both, final_state)
print(f"Wahrscheinlichkeit für Stau an beiden Ampeln: {prob_stau_both:.2f}")

print("Erweiterte Q-Simulation abgeschlossen: Entangled Verkehrsnetz mit Noise.")
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Define basis states for traffic: |0> = Congestion (Stau), |1> = Free Flow (Freie Fahrt)
psi0 = qt.basis(2, 0)  # Initial state for Ampel 1: Stau
psi1 = qt.basis(2, 1)  # Initial state for Ampel 2: Free

# Create a two-qubit entangled state: If Ampel 1 is Stau (|0>), Ampel 2 is also affected (correlated Stau)
# Using a Bell-like state for entanglement: (1/sqrt(2)) * (|00> + |11>)
bell_state = (qt.tensor(psi0, psi0) + qt.tensor(psi1, psi1)).unit()

# Model traffic interaction: Apply a CNOT gate to entangle them further if needed
# Here, Ampel 1 controls Ampel 2
cnot_matrix = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]])
cnot = qt.Qobj(cnot_matrix, dims=[[2, 2], [2, 2]])
entangled_state = cnot * bell_state

# Convert to density matrix for noise application
rho = entangled_state * entangled_state.dag()

# Add noise: Correct Bit-flip error channel (independent on each qubit)
p_flip = 0.1  # Probability of bit-flip (10%)
single_bitflip = [np.sqrt(1 - p_flip) * qt.qeye(2), np.sqrt(p_flip) * qt.sigmax()]
bit_flip_op = [qt.tensor(k1, k2) for k1 in single_bitflip for k2 in single_bitflip]

# Apply the noise channel to the density matrix
noisy_state = sum([op * rho * op.dag() for op in bit_flip_op])

# Measurement: Measure in Z-basis to get probabilities of states
# Probabilities for each outcome: |00>, |01>, |10>, |11>
measurement_ops = [qt.tensor(qt.basis(2, i).proj(), qt.basis(2, j).proj()) for i in range(2) for j in range(2)]
probs = [qt.expect(op, noisy_state) for op in measurement_ops]

print("Probabilities:")
print("|00> (Both Stau):", probs[0])
print("|01> (Ampel1 Stau, Ampel2 Free):", probs[1])
print("|10> (Ampel1 Free, Ampel2 Stau):", probs[2])
print("|11> (Both Free):", probs[3])

# Visualize Bloch spheres for each qubit
# Partial trace to get reduced density matrices
rho0 = noisy_state.ptrace(0)  # Qubit 0 (Ampel 1)
rho1 = noisy_state.ptrace(1)  # Qubit 1 (Ampel 2)

fig = plt.figure(figsize=(10, 5))

b0 = qt.Bloch(fig=fig, axes=fig.add_subplot(121, projection='3d'))
b0.add_states(rho0)
b0.render()
b0.axes.set_title('Ampel 1 Bloch Sphere')

b1 = qt.Bloch(fig=fig, axes=fig.add_subplot(122, projection='3d'))
b1.add_states(rho1)
b1.render()
b1.axes.set_title('Ampel 2 Bloch Sphere')

plt.savefig('bloch_verkehr.png')  # Save for GitHub Artifacts
plt.show()  # Optional for local view
