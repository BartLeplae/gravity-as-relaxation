"""
Step 5B: Strong-Field Black Hole Ray-Tracing & Photon Rings
-----------------------------------------------------------
Simulates light propagation in the strong-field regime around a Schwarzschild
black hole using the numerically relaxed vacuum density field.

Dependencies:
    numpy, matplotlib (Zero SciPy dependency required!)

Physical Landmarks Derived:
1. Critical Capture Radius: b_c = (3*sqrt(3)/2) * r_s = 2.598076 r_s
2. Unstable Photon Sphere : r_ph = 1.5 r_s (rho_ph = 0.9330 r_s)
3. Event Horizon          : r_h = 1.0 r_s  (rho_h = 0.25 r_s)

Two vertically aligned panels:
- Top panel: 2D photon trajectories in physical Schwarzschild coordinates (x, y)
  showing the capture zone (crimson), critical photon rings (gold/orange), and
  scattered rays (royal blue).
- Bottom panel: Total deflection angle Delta theta vs impact parameter b,
  illustrating the logarithmic relativistic divergence at b -> b_c.

Author: Bart Leplae
Project: gravity-as-relaxation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# =============================================================================
# 1. PARAMETERS & RELAXED VACUUM FIELD (COMPACT ISOTROPIC LATTICE)
# =============================================================================
c0 = 1.0                 # Asymptotic speed of light
rs = 1.0                 # Normalized Schwarzschild radius (rs = 2GM/c^2)
s_h = 4.0 / rs           # Isotropic horizon coordinate rho_h = rs / 4 -> s_h = 4 / rs
N_grid = 601             # Lattice nodes

s_grid = np.linspace(0.0, s_h, N_grid)
ds = s_h / (N_grid - 1)

# Non-linear relaxation of the single field eta: d/ds [ (1/sqrt(eta)) * d eta / ds ] = 0
eta_num = np.linspace(1.0, 4.0, N_grid)
omega = 1.95             # Red-Black SOR over-relaxation parameter

for _ in range(8000):
    # Odd nodes
    mid_odd = eta_num[1::2]
    left_odd = eta_num[:-2:2]
    right_odd = eta_num[2::2]
    sp1 = 1.0 / np.sqrt(0.5 * (mid_odd + right_odd))
    sm1 = 1.0 / np.sqrt(0.5 * (mid_odd + left_odd))
    target_odd = (sp1 * right_odd + sm1 * left_odd) / (sp1 + sm1)
    eta_num[1::2] += omega * (target_odd - mid_odd)

    # Even nodes
    mid_even = eta_num[2:-1:2]
    left_even = eta_num[1:-2:2]
    right_even = eta_num[3::2]
    sp2 = 1.0 / np.sqrt(0.5 * (mid_even + right_even))
    sm2 = 1.0 / np.sqrt(0.5 * (mid_even + left_even))
    target_even = (sp2 * right_even + sm2 * left_even) / (sp2 + sm2)
    eta_num[2:-1:2] += omega * (target_even - mid_even)

    # Boundary conditions: Asymptotic flat vacuum (1.0) and Horizon saturation (4.0)
    eta_num[0] = 1.0
    eta_num[-1] = 4.0

# Calculate coordinate light speed c_eff and logarithmic lattice gradients
c_eff_nodes = c0 * (2.0 - np.sqrt(eta_num)) / (eta_num**1.5)
c_eff_nodes[-1] = 0.0    # Exact zero at the horizon
ln_c_nodes = np.log(np.maximum(c_eff_nodes, 1e-12))
dlnc_ds_nodes = np.gradient(ln_c_nodes, ds)

# Critical theoretical benchmark
bc_exact = 1.5 * np.sqrt(3.0) * rs  # ~ 2.598076 rs

print(f"--> Vacuum relaxation complete.")
print(f"    Critical capture impact parameter b_c: {bc_exact:.6f} r_s")

# =============================================================================
# 2. PURE NUMPY FIELD QUERY & RK4 RAY-TRACING ENGINE
# =============================================================================
def get_c_and_grad(x, y):
    """Interpolates c_eff and spatial gradient from the relaxed isotropic lattice."""
    rho = np.hypot(x, y)
    s = min(1.0 / max(rho, 1e-6), s_h)
    ln_c = np.interp(s, s_grid, ln_c_nodes)
    c_eff = np.exp(ln_c)
    dlnc_ds = np.interp(s, s_grid, dlnc_ds_nodes)
    dlnc_drho = - (s**2) * dlnc_ds
    grad_lnc = dlnc_drho * np.array([x, y]) / max(rho, 1e-6)
    return c_eff, grad_lnc

def trace_photon(b_target, x_start=-25.0, max_steps=40000):
    """
    Integrates photon path with asymptotic impact parameter b_target.
    Returns physical trajectory (x_phys, y_phys), status, and final deflection angle.
    """
    # Initialize state at large distance: J = - y * vx / c^2 = b_target
    c_est, _ = get_c_and_grad(x_start, b_target)
    y_start = b_target * c_est
    c_init, _ = get_c_and_grad(x_start, y_start)
    state = np.array([x_start, y_start, c_init, 0.0])

    traj_x = [state[0]]
    traj_y = [state[1]]
    rho_h = 0.25 * rs

    passed_closest = False
    prev_rho = np.hypot(state[0], state[1])

    for _ in range(max_steps):
        x, y, vx, vy = state
        rho = np.hypot(x, y)

        if rho > prev_rho and prev_rho < 5.0:
            passed_closest = True
        prev_rho = rho

        # Condition 1: Horizon capture
        if rho <= rho_h * 1.002:
            traj_x.append(x)
            traj_y.append(y)
            status = "captured"
            break

        # Condition 2: Escape to outer space
        if passed_closest and rho > 16.0:
            traj_x.append(x)
            traj_y.append(y)
            status = "escaped"
            break

        c_curr, _ = get_c_and_grad(x, y)
        # Adaptive step size: ultra-fine near photon sphere, larger far away
        ds_step = 0.003 if rho < 1.2 else (0.015 if rho < 3.5 else 0.06)
        dt = ds_step / max(c_curr, 0.005)

        # Huygens-Fermat ray acceleration
        def deriv(st):
            xx, yy, vxx, vyy = st
            c_val, g = get_c_and_grad(xx, yy)
            v_dot_g = vxx * g[0] + vyy * g[1]
            ax = 2.0 * v_dot_g * vxx - (c_val**2) * g[0]
            ay = 2.0 * v_dot_g * vyy - (c_val**2) * g[1]
            return np.array([vxx, vyy, ax, ay])

        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        traj_x.append(state[0])
        traj_y.append(state[1])
    else:
        status = "orbiting"

    traj_x = np.array(traj_x)
    traj_y = np.array(traj_y)

    # Compute deflection angle for escaped rays
    if status == "escaped":
        vx_f, vy_f = state[2], state[3]
        angle_in = 0.0
        angle_out = np.arctan2(vy_f, vx_f)
        defl_deg = np.degrees(np.abs(angle_out - angle_in))
    else:
        defl_deg = np.nan

    # Map trajectory points from isotropic (rho) to physical Schwarzschild (r)
    # r_phys = rho * eta(rho)
    rho_traj = np.hypot(traj_x, traj_y)
    eta_traj = (1.0 + 0.25 * rs / np.maximum(rho_traj, rho_h))**2
    x_phys = traj_x * eta_traj
    y_phys = traj_y * eta_traj

    return x_phys, y_phys, status, defl_deg

# =============================================================================
# 3. COMPUTE TRAJECTORIES ACROSS SYSTEMATIC IMPACT PARAMETERS
# =============================================================================
print("--> Tracing characteristic light paths around the black hole...")

# Curated set of impact parameters showing capture, orbiting, and scattering
b_sample = [
    1.20, 1.80, 2.30, 2.52, 2.585,          # Captured (b < b_c)
    2.602, 2.615, 2.64, 2.75, 3.10, 4.00, 5.50  # Escaped / Photon rings (b > b_c)
]

trajectories = []
for b in b_sample:
    xp, yp, stat, defl = trace_photon(b)
    trajectories.append((b, xp, yp, stat, defl))
    print(f"    b = {b:5.3f} r_s (b/b_c = {b/bc_exact:5.3f}) -> {stat:<8}")

# Scan for the deflection angle divergence curve
b_scan = np.linspace(2.603, 6.0, 35)
defl_scan = []
for b in b_scan:
    _, _, stat, defl = trace_photon(b)
    defl_scan.append(defl)
defl_scan = np.array(defl_scan)

# =============================================================================
# 4. VISUALIZATION (2 VERTICALLY ALIGNED PANELS)
# =============================================================================
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "trace_black_hole_rings.png")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
plt.subplots_adjust(hspace=0.28)

# --- TOP PANEL: 2D Light Trajectories in Physical Schwarzschild Space ---
# Physical landmarks
horizon_circle = Circle((0, 0), 1.0, color="black", zorder=10, label=r"Event Horizon ($r = 1.0\,r_s$)")
photon_sphere = Circle((0, 0), 1.5, color="none", ec="darkorange", linestyle="--", lw=2.0,
                       zorder=9, label=r"Photon Sphere ($r = 1.5\,r_s$)")
isco_circle = Circle((0, 0), 3.0, color="none", ec="teal", linestyle=":", lw=1.5,
                     zorder=8, label=r"ISCO Orbit ($r = 3.0\,r_s$)")

ax1.add_patch(horizon_circle)
ax1.add_patch(photon_sphere)
ax1.add_patch(isco_circle)

# Plot rays
plotted_labels = set()
for b, xp, yp, stat, _ in trajectories:
    if stat == "captured":
        col = "crimson"
        lbl = r"Captured Rays ($b < b_c$)" if "cap" not in plotted_labels else ""
        plotted_labels.add("cap")
        lw = 1.6
    elif b < 2.65:
        col = "darkorange"
        lbl = r"Photon Rings / Loops ($b \approx b_c$)" if "loop" not in plotted_labels else ""
        plotted_labels.add("loop")
        lw = 2.0
    else:
        col = "royalblue"
        lbl = r"Deflected Rays ($b > b_c$)" if "defl" not in plotted_labels else ""
        plotted_labels.add("defl")
        lw = 1.5

    ax1.plot(xp, yp, color=col, lw=lw, alpha=0.85, label=lbl)

# Highlight critical capture boundary b_c
ax1.axhline(bc_exact, color="black", linestyle="-.", lw=1.2, alpha=0.6)
ax1.text(-7.8, bc_exact + 0.15, rf"Critical Impact Parameter $b_c = \frac{{3\sqrt{{3}}}}{{2}} r_s \approx {bc_exact:.3f}\,r_s$",
         fontsize=9.5, fontweight="bold", color="black",
         bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="gray", alpha=0.85))

ax1.set_xlim(-8.0, 8.0)
ax1.set_ylim(-6.0, 6.0)
ax1.set_aspect("equal")
ax1.set_xlabel(r"Physical Coordinate $x / r_s$", fontsize=11, fontweight="bold")
ax1.set_ylabel(r"Physical Coordinate $y / r_s$", fontsize=11, fontweight="bold")
ax1.set_title(r"Top: Photon Orbits & Relativistic Rings in Schwarzschild Space $(r = \rho \cdot \eta)$",
              fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="lower left", fontsize=9.0)

# --- BOTTOM PANEL: Relativistic Deflection & Logarithmic Divergence ---
ax2.plot(b_scan, defl_scan, "o-", color="indigo", lw=2.2, ms=5,
         label=r"Ray-tracing through relaxed field $c_{\mathrm{eff}}(r)$")

# Asymptotic critical threshold
ax2.axvline(bc_exact, color="crimson", linestyle="--", lw=2.0,
            label=rf"Capture Horizon $b_c = {bc_exact:.3f}\,r_s$")

# Shaded Capture Zone
ax2.axvspan(1.5, bc_exact, color="crimson", alpha=0.12, label="Black Hole Capture Zone (No Escape)")

# Annotate logarithmic divergence at b -> b_c
ax2.annotate(r"Logarithmic Divergence:" + "\n" + r"$\Delta\theta \rightarrow \infty$ (Infinite loops on photon sphere)",
             xy=(bc_exact + 0.03, 300), xytext=(bc_exact + 0.55, 270),
             fontsize=9.5, fontweight="bold", color="darkred",
             arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="crimson", alpha=0.9))

# Weak-field reference (Einstein 4GM/c^2 b)
b_weak = np.linspace(3.5, 6.0, 100)
defl_weak_deg = np.degrees((2.0 * rs / b_weak))
ax2.plot(b_weak, defl_weak_deg, "k:", lw=1.8, label=r"Weak-field limit $\Delta\theta \approx 2 r_s / b$")

ax2.set_xlim(2.4, 6.0)
ax2.set_ylim(0.0, 360.0)
ax2.set_xlabel(r"Impact Parameter $b / r_s$", fontsize=11, fontweight="bold")
ax2.set_ylabel(r"Deflection Angle $\Delta\theta$ [degrees]", fontsize=11, fontweight="bold")
ax2.set_title(r"Bottom: Relativistic Deflection Angle vs Impact Parameter (Logarithmic Divergence at $b_c$)",
              fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right", fontsize=9.0)

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()
print(f"--> Successfully saved figure to {output_path}")