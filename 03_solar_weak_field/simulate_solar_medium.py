"""
Step 3: From Thermal Diffusion to Gravitational Medium (The Sun)
----------------------------------------------------------------
Extends the spherical relaxation framework of Step 2B to model the Sun
as a weak-field gravitational refractive medium.

In weak-field gravitation, the Newtonian potential Phi satisfies Laplace's equation:
    nabla^2 Phi = 0

Under the compact inverted coordinate u = 1/r:
    d^2 Phi / du^2 = 0

According to Einstein's optical-mechanical analogy (and isotropic weak-field GR):
    n(r) = 1 + 2|Phi| / c^2 = 1 + r_s / r
    c_eff(r) = c_0 / n(r) \approx c_0 * (1 - r_s / r)

The speed-of-light deficit Delta c(r) = c_0 - c_eff(r) \approx c_0 * (r_s / r) = c_0 * r_s * u
satisfies the exact same 1D linear relaxation equation as the thermal rod:
    d^2 (Delta c) / du^2 = 0

Boundary Conditions:
    Delta c(u = 0)         =    0.0 m/s  (Deep vacuum at infinity: c_eff = c_0)
    Delta c(u = u_sun)     = 1271.5 m/s  (Solar surface limb: c_eff = c_0 - 1.27 km/s)

Demonstrates:
1. Direct transition from heat diffusion to gravitational space-vacuum relaxation.
2. Emergence of the solar speed-of-light deficit Delta c(r) \approx 1.27 km/s * (R_sun / r).
3. The solar refractive index perturbation Delta n(r) = r_s / r (4.24 ppm at solar limb).
4. Physical foundation for the 1.75'' Eddington light bending: wavefronts travel slower
   near the Sun, tilting toward the center of mass.

Author: Bart Leplae
Project: gravity-as-relaxation
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PHYSICAL CONSTANTS & SOLAR SYSTEM PARAMETERS
# =============================================================================
G = 6.67430e-11           # Gravitational constant [m^3 / (kg s^2)]
M_sun = 1.98847e30        # Solar mass [kg]
c0 = 299792458.0          # Speed of light in deep vacuum [m/s]
R_sun = 6.96342e8         # Solar radius [m]
rs_sun = 2.0 * G * M_sun / (c0**2)  # Schwarzschild radius of the Sun ~ 2953.34 m

# Surface parameters at r = R_sun
epsilon_surface = rs_sun / R_sun               # Dimensionless potential ~ 4.24122e-6
delta_c_surface = c0 * epsilon_surface         # Speed deficit ~ 1271.49 m/s = 1.27 km/s
delta_n_surface_ppm = epsilon_surface * 1e6    # Refractive perturbation ~ 4.24 ppm

print(f"--> Physical parameters of the Sun:")
print(f"    Schwarzschild radius r_s : {rs_sun:.2f} m")
print(f"    Solar radius R_sun       : {R_sun:.3e} m")
print(f"    Surface speed deficit    : {delta_c_surface:.2f} m/s ({delta_c_surface/1e3:.3f} km/s)")
print(f"    Surface refractive shift : {delta_n_surface_ppm:.3f} ppm")

# =============================================================================
# 2. COMPACT INVERTED GRID (u-SPACE: u = 1 / r)
# =============================================================================
u_max = 1.0 / R_sun       # Inner boundary at solar surface (r = R_sun)
u_min = 0.0               # Outer boundary at true infinity (r -> infinity)

N_points = 101
u = np.linspace(u_min, u_max, N_points)
du = (u_max - u_min) / (N_points - 1)

# Dirichlet boundary conditions
delta_c_inf = 0.0         # At u = 0 (r -> infinity): no deficit
delta_c_limb = delta_c_surface  # At u = u_max (r = R_sun)

# =============================================================================
# 3. NUMERICAL RELAXATION IN u-SPACE
# =============================================================================
# Initialize with an exaggerated perturbation to illustrate smoothing dynamics
delta_c_u = np.linspace(delta_c_inf, delta_c_limb, N_points)
delta_c_u[1:-1] += 500.0 * np.sin(np.pi * u[1:-1] / u_max)

snapshots = {0: delta_c_u.copy()}
snapshot_iters = [5, 25, 100, 400]

omega = 1.85              # Successive Over-Relaxation (SOR) factor
max_iterations = 2500
tolerance = 1e-9

print(f"--> Starting relaxation of the solar refractive deficit in u-space...")

for iteration in range(1, max_iterations + 1):
    # Odd nodes
    i_odd = np.arange(1, N_points - 1, 2)
    diff_odd = 0.5 * (delta_c_u[i_odd - 1] + delta_c_u[i_odd + 1]) - delta_c_u[i_odd]
    delta_c_u[i_odd] += omega * diff_odd

    # Even nodes
    i_even = np.arange(2, N_points - 1, 2)
    diff_even = 0.5 * (delta_c_u[i_even - 1] + delta_c_u[i_even + 1]) - delta_c_u[i_even]
    delta_c_u[i_even] += omega * diff_even

    # Enforce boundary conditions
    delta_c_u[0] = delta_c_inf
    delta_c_u[-1] = delta_c_limb

    if iteration in snapshot_iters:
        snapshots[iteration] = delta_c_u.copy()

    max_residual = max(np.max(np.abs(diff_odd)), np.max(np.abs(diff_even)))
    if max_residual < tolerance:
        print(f"    Converged after {iteration} iterations (max residual: {max_residual:.2e} m/s)")
        snapshots[iteration] = delta_c_u.copy()
        break

# Analytical linear verification in u-space
delta_c_exact_u = delta_c_limb * (u / u_max)
max_err_u = np.max(np.abs(delta_c_u - delta_c_exact_u))
print(f"    Maximum error in inverted u-space: {max_err_u:.4e} m/s")

# =============================================================================
# 4. MAPPING TO PHYSICAL SPACE & OPTICAL PROPERTIES
# =============================================================================
# Map nodes back to physical space: r = 1 / u
r_nodes = 1.0 / u[1:]     # In meters (excluding u=0)
r_norm = r_nodes / R_sun   # In units of solar radius
delta_c_nodes = delta_c_u[1:]

# Continuous physical profile
r_dense = np.linspace(1.0, 10.0, 500)  # In units of R_sun
r_dense_m = r_dense * R_sun
delta_c_dense = c0 * (rs_sun / r_dense_m)             # Speed deficit [m/s]
delta_n_dense_ppm = (rs_sun / r_dense_m) * 1e6        # Refractive perturbation [ppm]

# Distant solar system planetary contexts
r_mercury = 5.791e10 / R_sun     # ~ 83.2 R_sun
r_earth = 1.496e11 / R_sun       # ~ 214.8 R_sun (1 AU)
delta_c_mercury = c0 * (rs_sun / (r_mercury * R_sun))
delta_c_earth = c0 * (rs_sun / (r_earth * R_sun))

print(f"    Distant contexts:")
print(f"      Mercury ({r_mercury:.1f} R_sun): Delta c = {delta_c_mercury:.2f} m/s")
print(f"      Earth   ({r_earth:.1f} R_sun): Delta c = {delta_c_earth:.2f} m/s")

# =============================================================================
# 5. VISUALIZATION (2 VERTICALLY ALIGNED PANELS)
# =============================================================================
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "simulate_solar_medium.png")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 11))
plt.subplots_adjust(hspace=0.32)

# --- TOP PANEL: Relaxation in Inverted u-Space ---
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snapshots)))

for (iter_num, dc_snap), col in zip(snapshots.items(), colors):
    if iter_num == 0:
        ax1.plot(u / u_max, dc_snap, color='crimson', linestyle=':', lw=2.0,
                 label=r'Initial Perturbed State $\Delta c_{\mathrm{init}}(u)$')
    elif iter_num == list(snapshots.keys())[-1]:
        ax1.plot(u / u_max, dc_snap, color='blue', lw=2.5,
                 label=f'Final Relaxed Deficit (Iter {iter_num})')
    else:
        ax1.plot(u / u_max, dc_snap, color=col, linestyle='--', lw=1.3, alpha=0.8,
                 label=f'Iteration {iter_num}')

ax1.plot(u / u_max, delta_c_exact_u, 'k-', lw=1.2, alpha=0.5,
         label=r'Exact Linear Solution: $\Delta c(u) = \Delta c_\odot \cdot (u / u_\odot)$')
ax1.scatter([0.0, 1.0], [delta_c_inf, delta_c_limb], color='red', s=60, zorder=6)

ax1.annotate(r'$u = 0 \rightarrow r = \infty$' + '\n' + r'$c_{\mathrm{eff}} = c_0$ (Unperturbed Vacuum)',
             xy=(0.0, delta_c_inf), xytext=(0.04, 320),
             fontsize=9.5, fontweight='bold', color='navy',
             arrowprops=dict(arrowstyle='->', color='navy', lw=1.4))
ax1.annotate(r'$u = u_\odot \rightarrow r = R_\odot$' + '\n' + r'$\Delta c_\odot = 1271.5\ \mathrm{m/s}$ (Solar Limb)',
             xy=(1.0, delta_c_limb), xytext=(0.65, delta_c_limb + 180),
             fontsize=9.5, fontweight='bold', color='darkred',
             arrowprops=dict(arrowstyle='->', color='crimson', lw=1.4))

ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-60, 1750)
ax1.set_xlabel(r'Inverted Coordinate $u / u_\odot = R_\odot / r$  [Infinity $\rightarrow$ Solar Surface]',
              fontsize=11, fontweight='bold')
ax1.set_ylabel(r'Speed Deficit $\Delta c(u) = c_0 - c_{\mathrm{eff}}$ [m/s]', fontsize=11, fontweight='bold')
ax1.set_title(r'Step 3 (Top): Relaxation of the Solar Refractive Field in Inverted Space  $\left[ \frac{d^2 (\Delta c)}{du^2} = 0 \right]$',
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=9.0)

box_top = (
    r"From Thermal Diffusion to Gravitational Medium:" + "\n"
    r"• Gravitational Potential: $\nabla^2 \Phi = 0 \rightarrow \frac{d^2 \Phi}{du^2} = 0$" + "\n"
    r"• Optical Analogy: $\Delta c(u) = c_0 \cdot \frac{r_s}{r} = c_0 r_s u$" + "\n"
    r"• Boundary Conditions:" + "\n"
    r"  - Deep Vacuum ($u = 0$): $\Delta c = 0 \rightarrow c_{\mathrm{eff}} = c_0$" + "\n"
    r"  - Solar Limb ($u = u_\odot$): $\Delta c_\odot = c_0 \frac{r_s}{R_\odot} \approx 1271.5\ \mathrm{m/s}$"
)
ax1.text(0.46, 0.10, box_top, transform=ax1.transAxes, fontsize=9.0,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='blue', alpha=0.9))

# --- BOTTOM PANEL: Physical Space Mapping & Optical Index ---
ax2_twin = ax2.twinx()

p1 = ax2.plot(r_dense, delta_c_dense, color='crimson', lw=2.5,
              label=r'Speed Deficit $\Delta c(r) = c_0 \frac{r_s}{r} = 1.27\ \mathrm{km/s} \cdot (R_\odot / r)$')
p2 = ax2_twin.plot(r_dense, delta_n_dense_ppm, color='teal', linestyle='--', lw=2.2,
                   label=r'Refractive Index Perturbation $\Delta n(r) \approx \frac{r_s}{r}$ [ppm]')

mask_view = (r_norm <= 10.0)
p3 = ax2.plot(r_norm[mask_view], delta_c_nodes[mask_view], 'bo', ms=4.5, alpha=0.7,
              label=r'Relaxed Nodes Mapped to Physical Space ($r_i = 1/u_i$)')

ax2.scatter([1.0], [delta_c_surface], color='red', s=60, zorder=6)
ax2.annotate(r'$r = 1.0\,R_\odot$ : $\Delta c = 1.27\ \mathrm{km/s}$' + '\n' + r'$\Delta n = 4.24\ \mathrm{ppm}$',
             xy=(1.0, delta_c_surface), xytext=(1.8, delta_c_surface - 150),
             fontsize=9.5, fontweight='bold', color='darkred',
             arrowprops=dict(arrowstyle='->', color='crimson', lw=1.4))

ax2.set_xlim(0.8, 10.2)
ax2.set_ylim(-40, 1400)
ax2_twin.set_ylim(-0.15, 4.7)

ax2.set_xlabel(r'Radial Distance from Sun Center $r / R_\odot$', fontsize=11, fontweight='bold')
ax2.set_ylabel(r'Speed Deficit $\Delta c(r)$ [m/s]', color='crimson', fontsize=11, fontweight='bold')
ax2_twin.set_ylabel(r'Refractive Index Increase $\Delta n$ [ppm]', color='teal', fontsize=11, fontweight='bold')
ax2.set_title('Step 3 (Bottom): The Sun as a Gravitational Refractive Lens in Physical Space',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

lines = p1 + p2 + p3
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='upper right', fontsize=9.0)

box_bottom = (
    "Physical Implications for Light Bending:\n"
    r"• Gradient: $\nabla n(r) = - \frac{r_s}{r^2} \hat{r}$ points toward Sun center." + "\n"
    r"• Wavefront Tilt: Wavefronts travel slower near the Sun ($c_{\mathrm{eff}} < c_0$)." + "\n"
    r"• Result: Light rays bend inward by $\Delta\theta = \frac{4GM}{c^2 b} = 1.75^{\prime\prime}$ (Eddington 1919)." + "\n"
    r"• Context: Mercury ($83\,R_\odot$): $\Delta c \approx 15\ \mathrm{m/s}$ | Earth ($215\,R_\odot$): $\Delta c \approx 6\ \mathrm{m/s}$."
)
ax2.text(0.18, 0.42, box_bottom, transform=ax2.transAxes, fontsize=9.0,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='linen', edgecolor='crimson', alpha=0.95))

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()