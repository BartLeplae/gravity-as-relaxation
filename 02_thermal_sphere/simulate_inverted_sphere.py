"""
Step 2B: Spherical Heat Relaxation via Inverted Coordinates (u = 1/r)
--------------------------------------------------------------------
Demonstrates the mathematical transformation that bridges 3D spherical
heat conduction and 1D linear rod relaxation:
    nabla^2 T = (1 / r^2) * d/dr [ r^2 * (dT / dr) ] = 0

By substituting the compact inverted coordinate u = 1/r:
    du / dr = - 1 / r^2 = - u^2
    r^2 * (dT / dr) = - dT / du
    d/dr [ r^2 * (dT / dr) ] = - u^2 * d/du [ - dT / du ] = u^2 * (d^2 T / du^2) = 0
    --> d^2 T / du^2 = 0

Boundary Conditions:
    T(u = 0)   =   0.0 C  (True infinity: r -> infinity)
    T(u = u_0) = 100.0 C  (Inner spherical cavity: r = r_0, u_0 = 1 / r_0)

Key Insights:
1. Exact Mathematical Equivalence: Under u = 1/r, the 3D spherical Laplace equation
   transforms IDENTICALLY into the 1D linear rod equation from Step 1.
2. Compactifying Infinity: The entire infinite outer universe r in [r_0, infinity)
   is mapped onto a finite, closed 1D computational interval u in [0, 1/r_0].
3. Natural Adaptive Mesh Refinement: Uniform spacing Delta u in inverted space
   maps to Delta r ~ r^2 * Delta u in physical space. Nodes naturally cluster
   densely near the cavity where gradients are steepest, while stretching outward
   to infinity with zero wasted resolution.

Author: Bart Leplae
Project: gravity-as-relaxation
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARAMETERS & COMPACT INVERTED GRID (u-SPACE)
# =============================================================================
r0 = 1.0                  # Inner cavity radius
u_max = 1.0 / r0          # Inner boundary in u-space (u = 1.0)
u_min = 0.0               # Outer boundary in u-space (r -> infinity => u = 0.0)

N_points = 101            # Number of grid nodes
u = np.linspace(u_min, u_max, N_points)
du = (u_max - u_min) / (N_points - 1)

# Dirichlet boundary conditions
T_inf = 0.0               # Temperature at true infinity (u = 0)
T_cavity = 100.0          # Temperature at the inner cavity (u = u_max)

# =============================================================================
# 2. INITIALIZATION & RELAXATION IN u-SPACE
# =============================================================================
# Initialize with an exaggerated perturbation to illustrate smoothing dynamics
T_u = np.linspace(T_inf, T_cavity, N_points)
T_u[1:-1] += 40.0 * np.sin(np.pi * u[1:-1] / u_max)

snapshots = {0: T_u.copy()}
snapshot_iters = [5, 25, 100, 400]

omega = 1.85              # Successive Over-Relaxation (SOR) factor
max_iterations = 2500
tolerance = 1e-10

print(f"--> Starting 1D relaxation in inverted coordinate u in [0, {u_max:.1f}]...")

for iteration in range(1, max_iterations + 1):
    # Odd nodes update
    i_odd = np.arange(1, N_points - 1, 2)
    target_odd = 0.5 * (T_u[i_odd - 1] + T_u[i_odd + 1])
    diff_odd = target_odd - T_u[i_odd]
    T_u[i_odd] += omega * diff_odd

    # Even nodes update
    i_even = np.arange(2, N_points - 1, 2)
    target_even = 0.5 * (T_u[i_even - 1] + T_u[i_even + 1])
    diff_even = target_even - T_u[i_even]
    T_u[i_even] += omega * diff_even

    # Enforce exact boundary values
    T_u[0] = T_inf
    T_u[-1] = T_cavity

    if iteration in snapshot_iters:
        snapshots[iteration] = T_u.copy()

    max_residual = max(np.max(np.abs(diff_odd)), np.max(np.abs(diff_even)))
    if max_residual < tolerance:
        print(f"    Converged after {iteration} iterations (max residual: {max_residual:.2e})")
        snapshots[iteration] = T_u.copy()
        break

# =============================================================================
# 3. ANALYTICAL VERIFICATION IN u-SPACE & r-SPACE
# =============================================================================
# In u-space, the steady-state profile is strictly linear:
T_exact_u = T_cavity * (u / u_max)
max_err_u = np.max(np.abs(T_u - T_exact_u))

# In physical space (r = 1/u), the profile is purely hyperbolic (1/r):
r_nodes = 1.0 / u[1:]     # Exclude u=0 (r = infinity)
T_nodes = T_u[1:]
T_exact_r_nodes = T_cavity * (r0 / r_nodes)
max_err_r = np.max(np.abs(T_nodes - T_exact_r_nodes))

print(f"    Maximum error in inverted u-space: {max_err_u:.4e} C")
print(f"    Maximum error in physical r-space: {max_err_r:.4e} C")

# =============================================================================
# 4. VISUALIZATION (2 VERTICALLY ALIGNED PANELS)
# =============================================================================
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "simulate_inverted_sphere.png")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 11))
plt.subplots_adjust(hspace=0.32)

# --- TOP PANEL: Relaxation in Computational u-Space ---
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snapshots)))

for (iter_num, T_snap), col in zip(snapshots.items(), colors):
    if iter_num == 0:
        ax1.plot(u / u_max, T_snap, color='crimson', linestyle=':', lw=2.0,
                 label='Initial Perturbed Guess')
    elif iter_num == list(snapshots.keys())[-1]:
        ax1.plot(u / u_max, T_snap, color='blue', lw=2.5,
                 label=f'Final Relaxed State (Iter {iter_num})')
    else:
        ax1.plot(u / u_max, T_snap, color=col, linestyle='--', lw=1.3, alpha=0.8,
                 label=f'Iteration {iter_num}')

ax1.plot(u / u_max, T_exact_u, 'k-', lw=1.2, alpha=0.5,
         label=r'Exact Linear Solution: $T(u) = T_{\mathrm{cavity}} \cdot u / u_0$')
ax1.scatter([0.0, 1.0], [T_inf, T_cavity], color='red', s=60, zorder=6)

ax1.annotate(r'$u = 0 \rightarrow r = \infty$' + '\n' + r'$T = 0^\circ\mathrm{C}$ (True Infinity)',
             xy=(0.0, T_inf), xytext=(0.04, 25),
             fontsize=9.5, fontweight='bold', color='navy',
             arrowprops=dict(arrowstyle='->', color='navy', lw=1.4))
ax1.annotate(r'$u = u_0 \rightarrow r = r_0$' + '\n' + r'$T = 100^\circ\mathrm{C}$ (Inner Cavity)',
             xy=(1.0, T_cavity), xytext=(0.72, T_cavity - 22),
             fontsize=9.5, fontweight='bold', color='darkred',
             arrowprops=dict(arrowstyle='->', color='crimson', lw=1.4))

ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-5, 145)
ax1.set_xlabel(r'Inverted Compact Coordinate $u / u_0 = r_0 / r$  [Infinity $\rightarrow$ Cavity]',
              fontsize=11, fontweight='bold')
ax1.set_ylabel(r'Temperature $T(u)$ [$^\circ\mathrm{C}$]', fontsize=11, fontweight='bold')
ax1.set_title(r'Step 2B (Top): Linear 1D Relaxation in Inverted Space  $\left[ \frac{d^2 T}{du^2} = 0 \right]$',
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=9.0)

box_top = (
    r"Inverted Coordinate Transformation ($u = 1/r$):" + "\n"
    r"• Spherical Laplace: $\frac{1}{r^2}\frac{d}{dr}\left(r^2 \frac{dT}{dr}\right) = 0 \rightarrow \frac{d^2 T}{du^2} = 0$" + "\n"
    r"• Boundary at Infinity: $r \rightarrow \infty \rightarrow u = 0$ (Finite grid!)" + "\n"
    r"• Exact Equivalence: 3D sphere maps identically to a 1D rod!"
)
ax1.text(0.04, 0.58, box_top, transform=ax1.transAxes, fontsize=9.0,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='blue', alpha=0.9))

# --- BOTTOM PANEL: Physical Space Mapping & Adaptive Mesh ---
r_dense = np.linspace(r0, 15.0, 500)
T_exact_r_dense = T_cavity * (r0 / r_dense)

# Step 2A Reference Curve (finite outer boundary R = 10)
R_step2a = 10.0
A_2a = T_cavity / (1.0 / r0 - 1.0 / R_step2a)
B_2a = - A_2a / R_step2a
T_step2a = np.maximum(0.0, A_2a / r_dense + B_2a)

ax2.plot(r_dense, T_exact_r_dense, color='crimson', lw=2.5,
         label=r'Exact Inverted Profile ($R \rightarrow \infty$): $T(r) = 100^\circ\mathrm{C} \cdot (r_0 / r)$')
ax2.plot(r_dense, T_step2a, color='gray', linestyle=':', lw=1.8,
         label=r'Step 2A Reference (Truncated at $R = 10\,r_0$, artificial zero)')

mask_view = (r_nodes <= 15.0)
ax2.plot(r_nodes[mask_view], T_nodes[mask_view], 'bo', ms=5.0, alpha=0.7,
         label=r'Relaxed Nodes Mapped to Physical Space ($r_i = 1/u_i$)')

ax2.scatter([r0], [T_cavity], color='red', s=60, zorder=6)
ax2.annotate(r'$r = r_0$, $T = 100^\circ\mathrm{C}$',
             xy=(r0, T_cavity), xytext=(r0 + 0.8, T_cavity - 8),
             fontsize=9.5, fontweight='bold', color='darkred',
             arrowprops=dict(arrowstyle='->', color='crimson', lw=1.4))

ax2.set_xlim(r0 - 0.3, 15.3)
ax2.set_ylim(-5, 110)
ax2.set_xlabel(r'Physical Radial Distance $r / r_0$', fontsize=11, fontweight='bold')
ax2.set_ylabel(r'Temperature $T(r)$ [$^\circ\mathrm{C}$]', fontsize=11, fontweight='bold')
ax2.set_title('Step 2B (Bottom): Physical Space Mapping & Natural Adaptive Resolution',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9.0)

box_bottom = (
    r"Adaptive Resolution Benefits of $u = 1/r$:" + "\n"
    r"• Node Spacing: $\Delta r \approx r^2 \Delta u$" + "\n"
    r"  - Near cavity ($r \approx 1$): Dense nodes where gradient is steep." + "\n"
    r"  - Far field ($r \gg 1$): Naturally sparse where field is nearly flat." + "\n"
    r"• True Asymptotic Decay: No artificial truncation boundary needed."
)
ax2.text(0.38, 0.42, box_bottom, transform=ax2.transAxes, fontsize=9.0,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='linen', edgecolor='crimson', alpha=0.95))

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()