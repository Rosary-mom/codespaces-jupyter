import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Modell: Qubit für Verkehrs-Zustand (|0> = Stau, |1> = freier Fluss)
psi0 = qt.basis(2, 0)  # Start im Stau

# Hamilton-Operator: Übergang zwischen Zuständen (z. B. Ampel-Switch)
omega = 1.0  # Frequenz des Übergangs
H = 2 * np.pi * omega * qt.sigmax() / 2

# Füge Noise hinzu (Dephasing für reale Quantenfehler, z. B. Umwelteinflüsse)
gamma = 0.1  # Dephasing-Rate
c_ops = [np.sqrt(gamma) * qt.sigmaz()]

# Zeiten
tlist = np.linspace(0, 10, 200)

# Löse Master-Gleichung (mit Noise)
result = qt.mesolve(H, psi0, tlist, c_ops, [qt.sigmax(), qt.sigmay(), qt.sigmaz()])

# Plot Bloch-Sphäre
fig = plt.figure()
b = qt.Bloch(fig=fig)
b.add_points(result.expect, meth='l')
b.make_sphere()
plt.savefig('bloch_verkehr.png')  # Speichere für Artifact-Upload
plt.close(fig)

print("Q-Simulation für Verkehr abgeschlossen. Bloch-Sphäre zeigt Evolution von Stau zu Fluss mit Noise.")
