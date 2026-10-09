"""
Step 1: 1D Steady-State Thermal Rod Relaxation
----------------------------------------------
Demonstrates the fundamental principle of iterative relaxation in 1D.
Solves Laplace's heat equation along a uniform rod:
    d^2 T / dx^2 = 0

Boundary Conditions:
    T(x = 0) = 100.0 C  (Hot heat reservoir)
    T(x = L) =   0.0 C  (Cold heat sink)

Demonstrates:
1. Iterative Gauss-Seidel / Jacobi relaxation from an arbitrary initial state.
2. Progressive smoothing of local gradients until steady-state flux balance is achieved.
3. Emergence of the strictly linear temperature profile T(x) = T_0 * (1 - x/L).
4. Exponential decay of the residual to machine precision.

Author: Bart Leplae
Project: gravity-as-relaxation
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARAMETERS & GRID DEFINITION
# =============================================================================
L = 1.0                # Normalized length of the rod
N_points = 101         # Number of grid nodes
x = np.linspace(0.0, L, N_points)
dx = L / (N_points - 1)

# Dirichlet boundary conditions
T_hot = 100.0          # Left boundary at x = 0
T_cold = 0.0           # Right boundary at x = L

# =============================================================================
# 2. INITIALIZATION
# =============================================================================
# Initial temperature distribution with a prominent perturbation
# to clearly visualize the smoothing dynamics during relaxation
T = np.zeros(N_points)
T[0] = T_hot
T[-1] = T_cold

# Perturbed initial guess: sine bump on top of linear ramp
T[1:-1] = T_hot * (1.0 - x[1:-1] / L) + 60.0 * np.sin(np.pi * x[1:-1] / L)

# Storage for profile snapshots at specific iterations
snapshots = {0: T.copy()}
snapshot_iterations = [1, 5, 20, 60, 200, 800]

# =============================================================================
# 3. NUMERICAL RELAXATION (GAUSS-SEIDEL / OVER-RELAXATION)
# =============================================================================
omega = 1.85           # Successive Over-Relaxation (SOR) parameter
max_iterations = 3000
tolerance = 1e-9

residuals = []
print(f"--> Starting 1D thermal rod relaxation over {N_points} nodes...")

for iteration in range(1, max_iterations + 1):
    max_diff = 0.0
    
    # Gauss-Seidel update across interior nodes
    for i in range(1, N_points - 1):
        target = 0.5 * (T[i - 1] + T[i + 1])
        diff = target - T[i]
        T[i] += omega * diff
        max_diff = max(max_diff, abs(diff))
    
    # Preserve exact boundary conditions
    T[0] = T_hot
    T[-1] = T_cold
    
    residuals.append(max_diff)
    
    # Capture intermediate snapshots
    if iteration in snapshot_iterations:
        snapshots[iteration] = T.copy()
        
    if max_diff < tolerance:
        print(f"    Converged after {iteration} iterations with max residual {max_diff:.2e} < {tolerance:.1e}")
        snapshots[iteration] = T.copy()
        break

# =============================================================================
# 4. ANALYTICAL VERIFICATION
# =============================================================================
# Exact steady-state solution to d^2T/dx^2 = 0
T_exact = T_hot + (T_cold - T_hot) * (x / L)
max_error = np.max(np.abs(T - T_exact))

print(f"    Maximum absolute error against analytical linear solution: {max_error:.4e} C")

# =============================================================================
# 5. VISUALIZATION (2 VERTICALLY ALIGNED PANELS)
# =============================================================================
# Ensure target output directory exists at repository root
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "simulate_thermal_rod.png")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 10.5))
plt.subplots_adjust(hspace=0.28)

# --- PANEL 1: Profile Evolution Across Relaxation Iterations ---
colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(snapshots)))

for (iter_num, T_snap), col in zip(snapshots.items(), colors):
    if iter_num == 0:
        ax1.plot(x, T_snap, color='crimson', linestyle=':', lw=2.2,
                 label='Initial Guess $T_{\mathrm{init}}(x)$ (Perturbed)')
    elif iter_num == list(snapshots.keys())[-1]:
        ax1.plot(x, T_snap, color='blue', lw=2.5,
                 label=f'Final Relaxed Profile (Iter {iter_num})')
    else:
        ax1.plot(x, T_snap, color=col, linestyle='--', lw=1.4, alpha=0.85,
                 label=f'Iteration {iter_num}')

# Overlay exact analytical solution
ax1.plot(x, T_exact, color='black', linestyle='-', lw=1.2, alpha=0.6,
         label=r'Analytical Solution: $T(x) = T_0 (1 - x/L)$')

# Annotate physical boundaries
ax1.scatter([0.0, L], [T_hot, T_cold], color='red', s=60, zorder=6)
ax1.annotate(r'$T(0) = 100^\circ\mathrm{C}$' + '\n(Hot Boundary)',
             xy=(0.0, T_hot), xytext=(0.06, T_hot - 10),
             fontsize=9.5, fontweight='bold', color='darkred',
             arrowprops=dict(arrowstyle='->', color='crimson', lw=1.4))
ax1.annotate(r'$T(L) = 0^\circ\mathrm{C}$' + '\n(Cold Sink)',
             xy=(L, T_cold), xytext=(L - 0.22, T_cold + 18),
             fontsize=9.5, fontweight='bold', color='navy',
             arrowprops=dict(arrowstyle='->', color='navy', lw=1.4))

ax1.set_xlim(-0.02, L + 0.02)
ax1.set_ylim(-5.0, 165.0)
ax1.set_xlabel('Position along the Rod $x / L$', fontsize=11, fontweight='bold')
ax1.set_ylabel(r'Temperature $T(x)$ [$^\circ\mathrm{C}$]', fontsize=11, fontweight='bold')
ax1.set_title(r'Step 1: Relaxation of 1D Heat Equation  $\left[ \frac{d^2 T}{dx^2} = 0 \right]$',
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=9.0)

# --- PANEL 2: Convergence History (Residual Decay) ---
ax2.semilogy(range(1, len(residuals) + 1), residuals, color='purple', lw=2.2,
             label=r'Max Node Residual: $\max |\Delta T_i|$')
ax2.axhline(tolerance, color='gray', linestyle=':', lw=1.6,
            label=f'Convergence Threshold ($\epsilon = {tolerance:.0e}$)')

# Highlight captured snapshot points on convergence curve
for iter_num in snapshot_iterations:
    if iter_num <= len(residuals):
        res_val = residuals[iter_num - 1]
        ax2.scatter(iter_num, res_val, color='darkorange', s=40, zorder=5)
        ax2.annotate(f'Iter {iter_num}',
                     xy=(iter_num, res_val), xytext=(iter_num * 1.15, res_val * 2.5),
                     fontsize=8.5, color='darkorange', fontweight='semibold')

ax2.set_xlim(1, len(residuals) * 1.05)
ax2.set_xlabel('Relaxation Iteration Count', fontsize=11, fontweight='bold')
ax2.set_ylabel(r'Maximum Residual $\max |\Delta T|$ [log scale]', fontsize=11, fontweight='bold')
ax2.set_title('Numerical Convergence to Steady-State Equilibrium', fontsize=12, fontweight='bold')
ax2.grid(True, which='both', alpha=0.3)
ax2.legend(loc='upper right', fontsize=9.5)

# Text box highlighting key relaxation properties
summary_box = (
    "1D Thermal Relaxation Properties:\n"
    r"• Governed by: $\nabla^2 T = 0$ (Harmonic averaging)" + "\n"
    r"• Boundary: $T(0) = 100^\circ\mathrm{C} \rightarrow T(L) = 0^\circ\mathrm{C}$" + "\n"
    r"• Equilibrium: Strictly linear gradient" + "\n"
    f"• Max Error vs Exact: {max_error:.2e}" + r" $^\circ\mathrm{C}$"
)
ax2.text(0.04, 0.15, summary_box, transform=ax2.transAxes, fontsize=9.0,
         verticalalignment='bottom',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='purple', alpha=0.9))

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()