import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------------
# INPUT DATA
# ------------------------------

geometry_sets = [
    # (L_AB, L_BC, M_b, M_c)
    (1.0, 0.8, 2.0, 1.5),
    (1.2, 0.6, 3.0, 2.0),
    (0.9, 1.0, 1.5, 2.5),
    (1.1, 0.7, 2.2, 1.0),
    (1.3, 0.9, 2.5, 3.0)
]

motion_sets = [
    # (omega1, omega2)
    (2.0, 1.0),
    (3.0, 1.5),
    (4.0, 2.0),
    (2.5, 3.0),
    (5.0, 0.5)
]

n_steps = 1000
theta1_array = np.linspace(0, 2*np.pi, n_steps)

# ------------------------------
# SIMULATION FUNCTION
# ------------------------------

def simulate_case(L1, L2, Mb, Mc, w1, w2):
    F_axial = []

    for theta1 in theta1_array:
        theta2 = w2/w1 * theta1
        thetaBC = theta1 - theta2

        rB = np.array([L1*np.sin(theta1),
                       L1*np.cos(theta1)])

        rC = rB + np.array([L2*np.sin(thetaBC),
                            L2*np.cos(thetaBC)])

        aB = -w1**2 * rB
        aC = aB - (w1-w2)**2 * np.array([L2*np.sin(thetaBC),
                                         L2*np.cos(thetaBC)])

        F_total = Mb*aB + Mc*aC

        eAB = np.array([np.sin(theta1),
                        np.cos(theta1)])

        F_axial.append(np.dot(F_total, eAB))

    return np.array(F_axial)

# ------------------------------
# RUN ALL 25 COMBINATIONS
# ------------------------------

results = []

for g in geometry_sets:
    for m in motion_sets:
        F = simulate_case(*g, *m)
        results.append((g, m, F))

# ------------------------------
# PLOTS
# ------------------------------

plt.figure(figsize=(10,8))

for i, (g, m, F) in enumerate(results):
    plt.plot(np.degrees(theta1_array), F, label=f"Case {i+1}")

plt.xlabel("Rotation angle AB (deg)")
plt.ylabel("Axial Force in AB")
plt.title("Axial Force vs Rotation Angle")
plt.legend(ncol=3, fontsize=7)
plt.grid(True)
plt.show()

# ------------------------------
# EXTREME LOADS
# ------------------------------

max_tension = -1e9
max_compression = 1e9
max_t_case = None
max_c_case = None

for i, (g, m, F) in enumerate(results):
    if F.max() > max_tension:
        max_tension = F.max()
        max_t_case = (i+1, g, m)

    if F.min() < max_compression:
        max_compression = F.min()
        max_c_case = (i+1, g, m)

print("Highest tension:", max_tension, "Case:", max_t_case)
print("Highest compression:", max_compression, "Case:", max_c_case)