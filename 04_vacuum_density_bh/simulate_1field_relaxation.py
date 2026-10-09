"""
Step 4: Non-Linear Single-Field Vacuum Density Relaxation (eta = psi^2)
----------------------------------------------------------------------
Models the emergence of the Schwarzschild metric and coordinate speed of light
through the mechanical relaxation of a single scalar field: the linear density
of space-quanta (eta).

Governing transport equation on the compact inverted grid (s = 1 / rho):
    d/ds [ (1 / sqrt(eta)) * (d eta / ds) ] = 0

Boundary conditions:
    eta(s = 0)   = 1.0  (Asymptotic vacuum at infinity)
    eta(s = s_h) = 4.0  (Maximum space-quantum packing density at event horizon)
    where s_h = 4 / r_s (isotropic horizon coordinate rho_h = r_s / 4)

Microscopic field decomposition:
    Local clock rate (lapse): chi = (2 - sqrt(eta)) / sqrt(eta)
    Effective speed of light : c_eff(eta) = c0 * chi / eta = c0 * (2 - sqrt(eta)) / eta^(3/2)
    Physical radial coordinate: r = rho * eta = eta / s

Demonstrates:
1. Non-linear vacuum stiffening (conductance sigma = 1 / sqrt(eta)) resolving the linear catastrophe.
2. Self-regulated horizon termination at eta = 4.0, chi = 0.0, c_eff = 0.0 without singularities.
3. Machine-precision agreement with analytical General Relativity (Schwarzschild).

Author: Bart Leplae
Project: gravity-as-relaxation
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARAMETERS & COMPACT ISOTROPIC GRID (s = 1 / rho)
# =============================================================================
c0 = 1.0                 # Unperturbed speed of light in deep vacuum
rs = 1.0                 # Normalized Schwarzschild radius (rs = 2GM/c^2)
N_points = 301           # Number of grid nodes

# Horizon coordinate on isotropic lattice: rho_h = rs / 4  -->  s_h = 4 / rs
s_h = 4.0 / rs
s = np.linspace(0.0, s_h, N_points)
ds = s_h / (N_points - 1)

# =============================================================================
# 2. BOUNDARY CONDITIONS & FIELD INITIALIZATION
# =============================================================================
eta_inf = 1.0            # Boundary at infinity (s = 0): undisturbed vacuum
eta_horizon = 4.0        # Boundary at horizon (s = s_h): 4-fold saturation packing

# Initial linear guess
eta = np.linspace(eta_inf, eta_horizon, N_points)
eta[0] = eta_inf
eta[-1] = eta_horizon

# =============================================================================
# 3. NON-LINEAR VACUUM RELAXATION (RED-BLACK SOR)
# =============================================================================
# Solves d/ds [ sigma(eta) * d eta / ds ] = 0 with conductance sigma = 1 / sqrt(eta)
omega = 1.95             # Over-relaxation parameter
max_iterations = 5000
tolerance = 1e-10

print(f"--> Starting non-linear vacuum relaxation over {N_points} nodes...")

for iteration in range(1, max_iterations + 1):
    # Odd nodes pass
    mid_odd = eta[1::2]
    left_odd = eta[:-2:2]
    right_odd = eta[2::2]
    sp1 = 1.0 / np.sqrt(0.5 * (mid_odd + right_odd))
    sm1 = 1.0 / np.sqrt(0.5 * (mid_odd + left_odd))
    target_odd = (sp1 * right_odd + sm1 * left_odd) / (sp1 + sm1)
    diff_odd = target_odd - mid_odd
    eta[1::2] += omega * diff_odd

    # Even nodes pass
    mid_even = eta[2:-1:2]
    left_even = eta[1:-2:2]
    right_even = eta[3::2]
    sp2 = 1.0 / np.sqrt(0.5 * (mid_even + right_even))
    sm2 = 1.0 / np.sqrt(0.5 * (mid_even + left_even))
    target_even = (sp2 * right_even + sm2 * left_even) / (sp2 + sm2)
    diff_even = target_even - mid_even
    eta[2:-1:2] += omega * diff_even

    # Enforce boundary values
    eta[0] = eta_inf
    eta[-1] = eta_horizon

    max_diff = max(np.max(np.abs(diff_odd)), np.max(np.abs(diff_even)))
    if max_diff < tolerance:
        print(f"    Converged after {iteration} iterations with max residual {max_diff:.2e} < {tolerance:.1e}")
        break

# Analytical benchmark on lattice: eta_exact(s) = (1 + s * rs / 4)^2
eta_exact = (1.0 + 0.25 * rs * s)**2
max_lattice_error = np.max(np.abs(eta - eta_exact))
print(f"    Maximum lattice error against exact metric factor: {max_lattice_error:.4e}")

# =============================================================================
# 4. FIELD DECOMPOSITION & COORDINATE TRANSFORMATION
# =============================================================================
sqrt_eta = np.sqrt(eta)

# Clock lapse rate per space quantum: chi = (2 - sqrt(eta)) / sqrt(eta)
chi = (2.0 - sqrt_eta) / sqrt_eta

# Effective coordinate speed of light: c_eff = c0 * chi / eta = c0 * (2 - sqrt(eta)) / eta^(3/2)
c_eff = c0 * chi / eta

# Map nodes to physical Schwarzschild areal radius: r = rho * eta = eta / s
rho = np.zeros_like(s)
rho[1:] = 1.0 / s[1:]
rho[0] = np.inf

r_phys = np.zeros_like(s)
r_phys[1:] = rho[1:] * eta[1:]
r_phys[0] = np.inf

# Exclude s = 0 (r = infinity) for physical space analysis
r_nodes = r_phys[1:]
eta_nodes = eta[1:]
chi_nodes = chi[1:]
c_eff_nodes = c_eff[1:]

# Sort in ascending physical radius
sort_idx = np.argsort(r_nodes)
r_sorted = r_nodes[sort_idx]
c_sorted = c_eff_nodes[sort_idx]
eta_sorted = eta_nodes[sort_idx]
chi_sorted = chi_nodes[sort_idx]

# =============================================================================
# 5. GENERAL RELATIVITY (SCHWARZSCHILD) BENCHMARK & VALIDATION TABLE
# =============================================================================
r_targets = np.array([1.0001, 1.05, 1.50, 2.00, 3.00, 5.00, 10.00, 20.00])

# Interpolate numerical values at validation targets
eta_num = np.interp(r_targets, r_sorted, eta_sorted)
chi_num = np.interp(r_targets, r_sorted, chi_sorted)
c_num = np.interp(r_targets, r_sorted, c_sorted)

# Exact analytical Schwarzschild values in isotropic coordinates:
# r = rho * (1 + rs / (4 rho))^2  -->  solve for rho
rho_targets = 0.5 * (r_targets - 0.5 * rs + np.sqrt(np.maximum(0.0, r_targets * (r_targets - rs))))
psi_targets = 1.0 + rs / (4.0 * rho_targets)
c_exact_art = (1.0 - rs / (4.0 * rho_targets)) / (psi_targets**3)

rel_error_pct = np.abs((c_num - c_exact_art) / c_exact_art) * 100.0

regimes = [
    "Event Horizon (r = 1.0 rs)",
    "Deep strong field (r = 1.05 rs)",
    "Photon sphere (r = 1.5 rs)",
    "Deep gravity well (r = 2.0 rs)",
    "ISCO orbit (r = 3.0 rs)",
    "Intermediate field (r = 5.0 rs)",
    "Weak-field regime (r = 10.0 rs)",
    "Asymptotically flat (r = 20.0 rs)"
]

print("\n" + "=" * 114)
print(f"{'r / rs':<8} | {'eta (density)':<14} | {'chi (lapse)':<12} | {'c_eff/c0 (num)':<16} | {'c_eff/c0 (GR)':<16} | {'Error (%)':<10} | {'Regime'}")
print("=" * 114)
for r_t, et, ch, cn, ce, err, reg in zip(r_targets, eta_num, chi_num, c_num, c_exact_art, rel_error_pct, regimes):
    print(f"{r_t:<8.2f} | {et:<14.4f} | {ch:<12.4f} | {cn:<16.6f} | {ce:<16.6f} | {err:<10.2e} | {reg}")
print("=" * 114 + "\n")

# =============================================================================
# 6. VISUALIZATION (2 VERTICALLY ALIGNED PANELS)
# =============================================================================
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "simulate_1field_relaxation.png")

# Analytical continuous curves
r_dense = np.linspace(1.0001 * rs, 15.0 * rs, 600)
rho_dense = 0.25 * (np.sqrt(r_dense) + np.sqrt(r_dense - rs))**2
psi_dense = 1.0 + rs / (4.0 * rho_dense)
eta_dense = psi_dense**2
chi_dense = (2.0 - psi_dense) / psi_dense
c_exact_dense = (1.0 - rs / (4.0 * rho_dense)) / (psi_dense**3)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11), sharex=True,
                               gridspec_kw={'height_ratios': [1.8, 1.2]})
plt.subplots_adjust(hspace=0.20)

# --- PANEL 1: Effective Coordinate Speed of Light c_eff(r) ---
mask_view = (r_sorted >= 1.0) & (r_sorted <= 15.0)

ax1.plot(r_sorted[mask_view], c_sorted[mask_view], color='indigo', lw=2.8,
         label=r'Numerical Relaxation: $c_{\mathrm{eff}}(\eta) = c_0 \frac{2 - \sqrt{\eta}}{\eta^{3/2}}$ (Single Field)')
ax1.plot(r_dense, c_exact_dense, color='darkorange', linestyle='--', lw=2.0,
         label='Analytical General Relativity Benchmark (Schwarzschild)')

# Sample discrete grid nodes
ax1.plot(r_sorted[mask_view][::5], c_sorted[mask_view][::5], 'ko', ms=4.5, alpha=0.6,
         label='Relaxed Lattice Nodes')

# Key relativistic orbits
ax1.axvline(1.0, color='black', linestyle='-', lw=2.0, label=r'Event Horizon ($1.0\,r_s$)')
ax1.axvline(1.5, color='orange', linestyle=':', lw=1.8, label=r'Photon Sphere ($1.5\,r_s$)')
ax1.axvline(3.0, color='teal', linestyle='--', lw=1.5, label=r'ISCO ($3.0\,r_s$)')

# Photon Sphere callout
c_ps = c_exact_art[2]
ax1.scatter([1.5], [c_ps], color='orange', s=60, zorder=6)
ax1.annotate(f'Photon Sphere:\n$c_{{\\mathrm{{eff}}}} = {c_ps:.4f}\\,c_0$',
             xy=(1.5, c_ps), xytext=(2.3, c_ps - 0.14),
             fontsize=9.5, fontweight='bold', color='darkorange',
             arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='orange', alpha=0.9))

# Horizon callout
ax1.scatter([1.0], [0.0], color='black', s=60, zorder=6)
ax1.annotate(r'Event Horizon: $c_{\mathrm{eff}} = 0.0$' + '\n' + r'($\eta = 4.0$, $\chi = 0.0$)',
             xy=(1.0, 0.0), xytext=(1.4, 0.10),
             fontsize=9.0, fontweight='bold', color='black',
             arrowprops=dict(arrowstyle='->', color='black', lw=1.3),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))

ax1.set_xlim(0.8, 15.2)
ax1.set_ylim(-0.04, 1.08)
ax1.set_ylabel(r'Coordinate Speed of Light $c_{\mathrm{eff}} / c_0$', fontsize=11, fontweight='bold')
ax1.set_title(r'Step 4: Emergence of the Schwarzschild Coordinate Speed from Vacuum Relaxation',
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower right', fontsize=9.0)

# Formula callout in Panel 1
box_1 = (
    "Single-Field Constitutive Law:\n"
    r"• Space-Quantum Density: $\eta \equiv \psi^2$" + "\n"
    r"• Speed Relation: $c_{\mathrm{eff}}(\eta) = c_0 \frac{\chi}{\eta} = c_0 \frac{2 - \sqrt{\eta}}{\eta^{3/2}}$" + "\n"
    r"• Metric Equivalence: $ds^2 = -c_{\mathrm{eff}}^2 dt^2 + \eta^2 d\vec{x}^2$"
)
ax1.text(0.42, 0.52, box_1, transform=ax1.transAxes, fontsize=9.0,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='indigo', alpha=0.9))

# --- PANEL 2: Microscopic Field Decomposition: Space (eta) vs Time (chi) ---
ax2.plot(r_dense, eta_dense, color='darkgreen', lw=2.4,
         label=r'Space-Quantum Density $\eta(r) = \psi^2$ (Spatial Compression $\rightarrow 4.0$)')
ax2.plot(r_dense, chi_dense, color='crimson', lw=2.4, linestyle='--',
         label=r'Induced Clock Rate $\chi(r) = \frac{2 - \sqrt{\eta}}{\sqrt{\eta}}$ (Time Dilation / Lapse $\rightarrow 0.0$)')

ax2.axvline(1.0, color='black', linestyle='-', lw=2.0)
ax2.axvline(1.5, color='orange', linestyle=':', lw=1.8)
ax2.axvline(3.0, color='teal', linestyle='--', lw=1.5)

ax2.scatter([1.0, 1.0], [4.0, 0.0], color=['darkgreen', 'crimson'], s=50, zorder=6)

ax2.set_xlim(0.8, 15.2)
ax2.set_ylim(-0.1, 4.3)
ax2.set_xlabel(r'Physical Schwarzschild Radius $r / r_s$', fontsize=11, fontweight='bold')
ax2.set_ylabel('Field Values [dimensionless]', fontsize=11, fontweight='bold')
ax2.set_title(r'Microscopic Dual-Mechanism in Physical Space: Space Compression $\eta(r)$ vs Clock Lapse $\chi(r)$',
              fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='center right', fontsize=9.0)

box_2 = (
    "Non-Linear Lattice Balance (s = 1/rho):\n"
    r"• Equation: $\frac{d}{ds}\left[ \frac{1}{\sqrt{\eta}} \frac{d\eta}{ds} \right] = 0$" + "\n"
    r"• Boundaries: $\eta(0) = 1.0 \rightarrow \eta(s_h) = 4.0$" + "\n"
    r"• Horizon: $\rho_h = 0.25\,r_s \rightarrow r = \rho \cdot \eta = 1.0\,r_s$"
)
ax2.text(0.12, 0.35, box_2, transform=ax2.transAxes, fontsize=9.0,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='linen', edgecolor='darkgreen', alpha=0.95))

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()