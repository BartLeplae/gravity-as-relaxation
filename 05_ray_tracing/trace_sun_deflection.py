"""
Step 5A: Solar Light Deflection via Numerically Relaxed Vacuum Field
-------------------------------------------------------------------
Integrates photon trajectories passing the Sun across impact parameters
b in [1.0, 10.0] R_sun through a numerically relaxed refractive index field.

Dependencies:
    numpy, matplotlib (Zero SciPy dependency required!)

Two vertically aligned subplots:
1. Top plot: 2D geometry of the Sun and the numerically relaxed speed-of-light
   deficit Delta c_eff(x, y), displaying horizontal dashed blue light rays that
   correspond directly to the lower plot. The 1.75'' Eddington deflection at the
   solar limb (b = 1.0 R_sun) is highlighted via a direct callout arrow.
2. Bottom plot: Deflection angle Delta theta vs impact parameter b, with the
   exact same blue rays projected as vertical dashed guidelines intersecting the
   hyperbolic deflection curve at 1.75'', 1.17'', 0.80'', and 0.55''.

Author: Bart Leplae
Project: gravity-as-relaxation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# =============================================================================
# 1. PHYSICAL CONSTANTS & NORMALIZED UNITS
# =============================================================================
G = 6.67430e-11          # Gravitational constant [m^3 / (kg s^2)]
M_sun = 1.98847e30       # Solar mass [kg]
c0 = 299792458.0         # Unperturbed speed of light in deep vacuum [m/s]
R_sun = 6.96342e8        # Solar radius [m]
rs_sun = 2.0 * G * M_sun / (c0**2)  # Schwarzschild radius of the Sun ~ 2953.34 m

# Normalized units: Length = R_sun, Velocity = c0
rs_norm = rs_sun / R_sun  # ~ 4.24122e-6

# =============================================================================
# 2. STEP A: NUMERICAL VACUUM RELAXATION ON DISCRETE GRID (s = 1 / rho)
# =============================================================================
# Domain: s = 0 (infinity) to s_max = 1.0 (solar surface R_sun)
s_min = 0.0
s_max = 1.0
N_grid = 501

s_grid = np.linspace(s_min, s_max, N_grid)
ds = s_max / (N_grid - 1)

# Boundary conditions for density field eta:
eta_inf = 1.0
eta_surface = (1.0 + 0.25 * rs_norm * s_max)**2

print(f"--> [Step A] Starting numerical relaxation over s in [0.0, {s_max:.1f}]...")

eta_num = np.linspace(eta_inf, eta_surface, N_grid)
omega = 1.95  # Red-Black SOR over-relaxation factor

for iteration in range(12000):
    # Pass 1: Odd grid nodes
    mid_odd = eta_num[1::2]
    left_odd = eta_num[:-2:2]
    right_odd = eta_num[2::2]
    sigma_p1 = 1.0 / np.sqrt(0.5 * (mid_odd + right_odd))
    sigma_m1 = 1.0 / np.sqrt(0.5 * (mid_odd + left_odd))
    target_odd = (sigma_p1 * right_odd + sigma_m1 * left_odd) / (sigma_p1 + sigma_m1)
    eta_num[1::2] += omega * (target_odd - mid_odd)

    # Pass 2: Even grid nodes
    mid_even = eta_num[2:-1:2]
    left_even = eta_num[1:-2:2]
    right_even = eta_num[3::2]
    sigma_p2 = 1.0 / np.sqrt(0.5 * (mid_even + right_even))
    sigma_m2 = 1.0 / np.sqrt(0.5 * (mid_even + left_even))
    target_even = (sigma_p2 * right_even + sigma_m2 * left_even) / (sigma_p2 + sigma_m2)
    eta_num[2:-1:2] += omega * (target_even - mid_even)

    # Enforce exact Dirichlet boundary conditions
    eta_num[0] = eta_inf
    eta_num[-1] = eta_surface

eta_exact = (1.0 + 0.25 * rs_norm * s_grid)**2
max_grid_err = np.max(np.abs(eta_num - eta_exact))
print(f"    Vacuum field relaxation complete. Maximum lattice error: {max_grid_err:.4e}")

# =============================================================================
# 3. STEP B: COUPLING & INTERPOLATION (PURE NUMPY)
# =============================================================================
c_eff_nodes = (2.0 - np.sqrt(eta_num)) / (eta_num**1.5)
ln_c_nodes = np.log(c_eff_nodes)
dlnc_ds_nodes = np.gradient(ln_c_nodes, ds)

def get_c_and_grad_from_relaxed_grid(x, y):
    """Queries c_eff and spatial gradient directly from the relaxed lattice using pure NumPy."""
    rho = np.hypot(x, y)
    s = min(1.0 / rho, s_max)

    ln_c = np.interp(s, s_grid, ln_c_nodes)
    c_eff = np.exp(ln_c)

    # Chain rule: d(ln c)/drho = - s^2 * d(ln c)/ds
    dlnc_ds = np.interp(s, s_grid, dlnc_ds_nodes)
    dlnc_drho = - (s**2) * dlnc_ds
    grad_lnc = dlnc_drho * np.array([x, y]) / rho
    return c_eff, grad_lnc

# =============================================================================
# 4. STEP C: RAY-TRACING SCAN VIA BUILT-IN RUNGE-KUTTA (RK4)
# =============================================================================
def ray_derivatives(state):
    """Evaluates the Huygens-Fermat ray acceleration equation."""
    x, y, vx, vy = state
    c_eff, grad_lnc = get_c_and_grad_from_relaxed_grid(x, y)
    v_dot_g = vx * grad_lnc[0] + vy * grad_lnc[1]
    ax = 2.0 * v_dot_g * vx - (c_eff**2) * grad_lnc[0]
    ay = 2.0 * v_dot_g * vy - (c_eff**2) * grad_lnc[1]
    return np.array([vx, vy, ax, ay])

def trace_single_photon(b, x_start=-250.0, x_end=250.0):
    """Integrates photon path from x_start to x_end using adaptive RK4 integration."""
    c_init, _ = get_c_and_grad_from_relaxed_grid(x_start, b)
    state = np.array([x_start, b, c_init, 0.0])

    while state[0] < x_end:
        rho = np.hypot(state[0], state[1])
        # Adaptive step size: ultra-fine near the solar limb, larger far away
        h = 0.02 if rho < 2.5 else (0.1 if rho < 10.0 else 0.5)

        k1 = ray_derivatives(state)
        k2 = ray_derivatives(state + 0.5 * h * k1)
        k3 = ray_derivatives(state + 0.5 * h * k2)
        k4 = ray_derivatives(state + h * k3)
        state += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    vx_f = state[2]
    vy_f = state[3]
    defl_rad = - vy_f / vx_f
    factor = 0.5 * (state[0] / np.hypot(state[0], b) + abs(x_start) / np.hypot(x_start, b))
    defl_arcsec = (defl_rad / factor) * (180.0 / np.pi) * 3600.0
    return defl_arcsec

print("--> [Step C] Starting ray-tracing through the numerically relaxed vacuum field...")

b_values = np.linspace(1.0, 10.0, 25)
deflections_num = np.array([trace_single_photon(b) for b in b_values])

# Analytical curves for comparison
b_dense = np.linspace(1.0, 10.0, 300)
defl_einstein = (2.0 * rs_norm / b_dense) * (180.0 / np.pi) * 3600.0
defl_newton = (rs_norm / b_dense) * (180.0 / np.pi) * 3600.0

# Representative rays for visual projection between subplots
sample_rays = [1.0, 1.5, 2.2, 3.2]
sample_defls = [(2.0 * rs_norm / b_r) * (180.0 / np.pi) * 3600.0 for b_r in sample_rays]

print(f"    Numerical deflection at solar limb (b = 1.0 R_sun): {deflections_num[0]:.5f}''")
print(f"    Analytical Einstein prediction (4GM / c^2 R)        : {defl_einstein[0]:.5f}''")
print(f"    Agreement                                          : {100.0 * (1.0 - abs(deflections_num[0] - defl_einstein[0])/defl_einstein[0]):.4f}%")

# =============================================================================
# 5. VISUALIZATION (2 VERTICALLY ALIGNED PANELS)
# =============================================================================
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "trace_sun_deflection.png")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 11))

# --- TOP PANEL: Geometry & Light Ray Paths in the Relaxed Vacuum Field ---
X, Y = np.meshgrid(np.linspace(-6.5, 7.5, 280), np.linspace(-4.0, 4.3, 260))
R_grid = np.hypot(X, Y)
S_grid = 1.0 / np.maximum(R_grid, 1.0)
c_field = np.exp(np.interp(S_grid, s_grid, ln_c_nodes))
delta_c_ppm = (1.0 - c_field) * 1e6  # Speed deficit in parts per million (ppm)

contour = ax1.contourf(X, Y, delta_c_ppm, levels=32, cmap="YlOrRd", alpha=0.35)
cbar = plt.colorbar(contour, ax=ax1, fraction=0.035, pad=0.03)
cbar.set_label(r"Relaxed speed-of-light deficit $\Delta c_{\mathrm{eff}} / c_0$ [ppm]", fontsize=10)

sun_circle = Circle((0, 0), 1.0, color="gold", ec="darkorange", lw=2.5, zorder=5, label=r"Sun ($1.0\,R_\odot$)")
ax1.add_patch(sun_circle)

# Draw the 4 dashed blue representative light rays
for i, b_r in enumerate(sample_rays):
    lbl = "Corresponding light rays ($b$)" if i == 0 else ""
    ax1.plot([-6.5, 7.5], [b_r, b_r], color="royalblue", linestyle="--", lw=1.8, alpha=0.85, label=lbl)

# Clean, uniform labels for all rays on the right-hand side
for b_r in sample_rays:
    ax1.text(4.8, b_r + 0.12, f"$b = {b_r:.1f}\\,R_\\odot$", fontsize=9.5, color="darkblue", fontweight="semibold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"))

# Callout arrow pointing to the grazing solar limb with Eddington measurement
ax1.annotate(r"$\mathbf{1.7496^{\prime\prime}}$ (Eddington 1919)",
             xy=(0.0, 1.0), xytext=(-3.8, 1.85),
             fontsize=9.5, color="darkblue", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="royalblue", lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="royalblue"))

ax1.set_xlim(-6.5, 7.5)
ax1.set_ylim(-3.5, 4.3)
ax1.set_aspect("equal")
ax1.set_xlabel(r"Orbital plane coordinate $x / R_\odot$", fontsize=11, fontweight="bold")
ax1.set_ylabel(r"Orbital plane coordinate $y / R_\odot$", fontsize=11, fontweight="bold")
ax1.set_title("Top: Numerically Relaxed Vacuum Field and Light Paths around the Sun", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="lower left", fontsize=9.5)

# --- BOTTOM PANEL: Deflection Angle vs Impact Parameter ---
ax2.plot(b_values, deflections_num, "ko", ms=6, label="Ray-tracing through relaxed field")
ax2.plot(b_dense, defl_einstein, "r-", lw=2.2, label=r"Relativistic (Einstein): $\Delta\theta = 1.75^{\prime\prime} \cdot (R_\odot / b)$")
ax2.plot(b_dense, defl_newton, color="darkorange", linestyle="--", lw=1.8, label=r"Classical (Soldner/Newton): $\Delta\theta = 0.87^{\prime\prime} \cdot (R_\odot / b)$")

# Vertical dashed blue guidelines matching the top plot
for i, (b_r, d_r) in enumerate(zip(sample_rays, sample_defls)):
    lbl = "Corresponding light rays from top panel" if i == 0 else ""
    ax2.axvline(b_r, color="royalblue", linestyle="--", lw=1.8, alpha=0.85, label=lbl)
    ax2.plot(b_r, d_r, "o", color="royalblue", ms=7, zorder=6)
    if b_r == 1.0:
        label_text = f"$b = 1.0\\,R_\\odot$\n{d_r:.2f}'' (Eddington)"
        ax2.annotate(label_text, xy=(b_r, d_r), xytext=(b_r + 0.35, d_r - 0.04),
                     fontsize=9, color="darkblue", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="royalblue", lw=1.2))
    else:
        label_text = f"{d_r:.2f}''"
        ax2.annotate(label_text, xy=(b_r, d_r), xytext=(b_r + 0.18, d_r + 0.08),
                     fontsize=9, color="darkblue", fontweight="bold")

ax2.set_xlabel(r"Impact parameter at infinity $b / R_\odot$", fontsize=11, fontweight="bold")
ax2.set_ylabel(r"Deflection angle $\Delta\theta$ [arcseconds ($^{\prime\prime}$)]", fontsize=11, fontweight="bold")
ax2.set_title("Bottom: Light Deflection with Projections to Corresponding Rays", fontsize=12, fontweight="bold")
ax2.set_xlim(0.8, 10.2)
ax2.set_ylim(0.0, 2.05)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right", fontsize=9.5)

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()