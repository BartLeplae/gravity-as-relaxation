"""
Step 2A: Spherical Shell Heat Relaxation on a Uniform Radial Grid
-----------------------------------------------------------------
Models steady-state heat conduction through a 3D spherical shell medium
between an inner spherical cavity and an outer spherical boundary:
    nabla^2 T = (1 / r^2) * d/dr [ r^2 * (dT / dr) ] = 0

Boundary Conditions:
    T(r = r_0) = 100.0 C  (Hot inner spherical cavity)
    T(r = R)   =   0.0 C  (Cold outer spherical boundary)

Demonstrates:
1. Conservative finite-difference relaxation with spherical area weighting A(r) ~ r^2.
2. Emergence of the hyperbolic 1/r profile: T(r) = A / r + B.
3. Physical conservation of total heat current Q = 4*pi*r^2 * (-dT/dr) = Constant.
4. Comparison against naive 1D linear averaging (which ignores 3D area expansion).
5. The core limitation of uniform r-grids: steep gradients cluster near the cavity r_0,
   wasting resolution at large r and preventing the outer boundary from reaching infinity.
   (Motivates Step 2B: the compact inverted coordinate u = 1/r).

Author: Bart Leplae
Project: gravity-as-relaxation
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARAMETERS & UNIFORM RADIAL GRID (r-SPACE)
# =============================================================================
r0 = 1.0               # Inner cavity radius
R = 10.0               # Outer boundary radius
N_points = 101         # Number of radial nodes

r = np.linspace(r0, R, N_points)
dr = (R - r0) / (N_points - 1)

# Dirichlet boundary conditions
T_inner = 100.0        # Cavity temperature at r = r0
T_outer = 0.0          # Ambient temperature at r = R

# =============================================================================
# 2. ANALYTICAL SPHERICAL SOLUTION
# =============================================================================
# General solution to (1/r^2) d/dr (r^2 dT/dr) = 0 is T(r) = A / r + B
# Applying T(r0) = T_inner and T(R) = T_outer:
A_const = T_inner / (1.0 / r0 - 1.0 / R)
B_const = - A_const / R
T_exact = A_const / r + B_const

# =============================================================================
# 3. NUMERICAL RELAXATION (SPHERICAL AREA WEIGHTING VS NAIVE 1D)
# =============================================================================
# --- Model 1: Naive 1D Rod Averaging (Ignoring r^2 Area Growth) ---
T_naive = np.linspace(T_inner, T_outer, N_points)

for _ in range(3000):
    T_naive[1:-1] = 0.5 * (T_naive[:-2] + T_naive[2:])
    T_naive[0] = T_inner
    T_naive[-1] = T_outer

# --- Model 2: Conservative Spherical Flux Relaxation ---
# Flux balance across spherical shells:
# A_{i+1/2} * (T_{i+1} - T_i) / dr - A_{i-1/2} * (T_i - T_{i-1}) / dr = 0
# Thermal conductance of spherical shell between nodes: A_eff ~ r_i * r_{i+1}
T_spherical = np.linspace(T_inner, T_outer, N_points)
omega = 1.85           # Over-relaxation parameter

for iteration in range(6000):
    # Odd nodes
    i_odd = np.arange(1, N_points - 1, 2)
    wp_odd = r[i_odd] * r[i_odd + 1]
    wm_odd = r[i_odd] * r[i_odd - 1]
    target_odd = (wp_odd * T_spherical[i_odd + 1] + wm_odd * T_spherical[i_odd - 1]) / (wp_odd + wm_odd)
    T_spherical[i_odd] += omega * (target_odd - T_spherical[i_odd])

    # Even nodes
    i_even = np.arange(2, N_points - 1, 2)
    wp_even = r[i_even] * r[i_even + 1]
    wm_even = r[i_even] * r[i_even - 1]
    target_even = (wp_even * T_spherical[i_even + 1] + wm_even * T_spherical[i_even - 1]) / (wp_even + wm_even)
    T_spherical[i_even] += omega * (target_even - T_spherical[i_even])

    # Fix boundary conditions
    T_spherical[0] = T_inner
    T_spherical[-1] = T_outer

max_error = np.max(np.abs(T_spherical - T_exact))
print(f"--> Spherical relaxation complete.")
print(f"    Maximum error against analytical 1/r profile: {max_error:.4e} C")

# =============================================================================
# 4. FLUX CONSERVATION & RADIAL DILUTION ANALYSIS
# =============================================================================
# Local heat flux density: q(r) = - dT/dr = A_const / r^2
q_density = A_const / (r**2)

# Total discrete heat current across concentric spherical shell interfaces:
r_mid = 0.5 * (r[:-1] + r[1:])
Q_total = 4.0 * np.pi * r[:-1] * r[1:] * (T_spherical[:-1] - T_spherical[1:]) / dr

print(f"    Total conserved heat current Q = {np.mean(Q_total):.2f} W (std: {np.std(Q_total):.2e} W)")

# =============================================================================
# 5. VISUALIZATION (2 VERTICALLY ALIGNED PANELS)
# =============================================================================
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "simulate_uniform_sphere.png")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 11))
plt.subplots_adjust(hspace=0.32)

# --- PANEL 1: Temperature Profile T(r) vs Radial Distance ---
ax1.plot(r, T_naive, color='gray', linestyle=':', lw=2.0,
         label=r'Naive 1D Linear Averaging (Ignoring $r^2$ Area Expansion)')
ax1.plot(r, T_spherical, color='blue', lw=2.5,
         label=r'Relaxed Spherical Profile $T_{\mathrm{num}}(r)$ (Area-Weighted)')
ax1.plot(r, T_exact, color='crimson', linestyle='--', lw=1.8,
         label=r'Analytical Spherical Solution: $T(r) = \frac{A}{r} + B$')

ax1.scatter([r0, R], [T_inner, T_outer], color='red', s=60, zorder=6)
ax1.annotate(r'$T(r_0) = 100^\circ\mathrm{C}$' + '\n(Inner Cavity)',
             xy=(r0, T_inner), xytext=(r0 + 0.6, T_inner - 10),
             fontsize=9.5, fontweight='bold', color='darkred',
             arrowprops=dict(arrowstyle='->', color='crimson', lw=1.4))
ax1.annotate(r'$T(R) = 0^\circ\mathrm{C}$' + '\n(Outer Boundary)',
             xy=(R, T_outer), xytext=(R - 2.5, T_outer + 18),
             fontsize=9.5, fontweight='bold', color='navy',
             arrowprops=dict(arrowstyle='->', color='navy', lw=1.4))

ax1.set_xlim(r0 - 0.2, R + 0.2)
ax1.set_ylim(-5, 110)
ax1.set_xlabel(r'Radial Distance $r / r_0$', fontsize=11, fontweight='bold')
ax1.set_ylabel(r'Temperature $T(r)$ [$^\circ\mathrm{C}$]', fontsize=11, fontweight='bold')
ax1.set_title(r'Step 2A: Heat Conduction in a Spherical Shell  $\left[ \nabla^2 T = \frac{1}{r^2}\frac{d}{dr}\left(r^2 \frac{dT}{dr}\right) = 0 \right]$',
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=9.5)

# Text box in Panel 1
box_1 = (
    "Uniform Radial Grid Setup:\n"
    r"• Inner cavity: $r_0 = 1.0$, $T(r_0) = 100^\circ\mathrm{C}$" + "\n"
    r"• Outer boundary: $R = 10.0$, $T(R) = 0^\circ\mathrm{C}$" + "\n"
    r"• Spherical Solution: $T(r) \propto 1/r$" + "\n"
    f"• Max Error vs Exact: {max_error:.2e}" + r" $^\circ\mathrm{C}$"
)
ax1.text(0.48, 0.45, box_1, transform=ax1.transAxes, fontsize=9.0,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='blue', alpha=0.9))

# --- PANEL 2: Heat Flux Density vs Conserved Total Current ---
ax2_twin = ax2.twinx()
p1 = ax2.plot(r, q_density, color='darkorange', lw=2.2,
              label=r'Flux Density $q(r) = -dT/dr \propto 1/r^2$ (Steep at Cavity)')
p2 = ax2_twin.plot(r_mid, Q_total, color='teal', linestyle='--', lw=2.2,
                   label=r'Total Heat Current $Q = 4\pi r^2 q(r) = \mathrm{Constant}$')

ax2.set_xlabel(r'Radial Distance $r / r_0$', fontsize=11, fontweight='bold')
ax2.set_ylabel(r'Local Flux Density $q(r)$ [W/m$^2$]', color='darkorange', fontsize=11, fontweight='bold')
ax2_twin.set_ylabel(r'Conserved Heat Current $Q$ [W]', color='teal', fontsize=11, fontweight='bold')
ax2_twin.set_ylim(0, 1600)
ax2.set_ylim(-5, 125)

ax2.set_title(r'Physical Mechanism: Spherical Area Dilution Forces $\frac{dT}{dr} \propto \frac{1}{r^2} \rightarrow T(r) \propto \frac{1}{r}$',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Limitations callout in Panel 2
box_2 = (
    "Limitations of Uniform $r$-Grid:\n"
    "1. Gradient Concentration: 80% of temperature drop occurs in $r \in [1, 3]$.\n"
    "2. Wasted Resolution: Nodes at large $r$ have negligible gradient.\n"
    "3. Horizon to Infinity: Cannot model $R \to \infty$ without infinite nodes!\n"
    r"$\rightarrow$ Motivates Step 2B: Inverted coordinate $u = 1/r$."
)
ax2.text(0.12, 0.45, box_2, transform=ax2.transAxes, fontsize=9.0,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='linen', edgecolor='darkorange', alpha=0.95))

lines = p1 + p2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='upper right', fontsize=9.0)

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()