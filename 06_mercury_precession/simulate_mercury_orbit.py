"""
Step 6: Relativistic Perihelion Precession of Mercury (43'' / Century)
----------------------------------------------------------------------
Simulates the orbital dynamics and perihelion advance of Mercury using the
relativistic Binet equation derived from the non-linear vacuum relaxation framework.

Dependencies:
    numpy, matplotlib (Zero SciPy dependency required!)

Physical Foundation:
1. In Step 2B, the spherical Laplace equation transformed into a 1D rod under u = 1/r:
       d^2 T / du^2 = 0
2. In Step 4, strong-field vacuum stiffening added non-linear tegendruk (sigma = 1 / sqrt(eta)).
3. For massive orbital motion in this non-linear medium (via the Maupertuis-Jacobi analogy),
   the classical Newtonian orbit equation:
       d^2 u / dphi^2 + u = 1 / p
   gains an emergent non-linear self-interaction term:
       d^2 u / dphi^2 + u = (1 / p) + (3/2) * r_s * u^2

Theoretical Relativistic Shift:
    Delta phi = (3 * pi * r_s) / (a * (1 - e^2)) = 5.019e-7 rad / orbit = 0.10352'' / orbit
    Over 415.2 orbits per Earth century:
    Delta phi_century = 0.10352'' * 415.20 = 42.98'' / century (Einstein's famous 43'')

Two Vertically Aligned Panels:
- Top Panel   : 2D orbital trajectory (rosette pattern) over 8 revolutions with an
                enhancement factor alpha = 60000 so the relativistic precession is clearly
                visible to the human eye, contrasted against the static Newtonian ellipse.
- Bottom Panel: Systematic scaling validation across enhancement factors (alpha = 10000 to 60000)
                proving linear convergence to the exact analytical GR prediction, accompanied
                by the unscaled astronomical breakdown.

Author: Bart Leplae
Project: gravity-as-relaxation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# =============================================================================
# 1. PHYSICAL CONSTANTS & ASTRONOMICAL PARAMETERS (MERCURY - SUN)
# =============================================================================
G = 6.67430e-11           # Gravitational constant [m^3 / (kg s^2)]
M_sun = 1.98847e30        # Solar mass [kg]
c0 = 299792458.0          # Vacuum speed of light [m/s]
rs_sun = 2.0 * G * M_sun / (c0**2)  # Schwarzschild radius of the Sun ~ 2953.34 m

# Mercury orbital parameters
a_merc = 5.79091e10       # Semi-major axis [m] (~ 0.3871 AU)
e_merc = 0.2056306        # Orbital eccentricity
p_merc = a_merc * (1.0 - e_merc**2)  # Semi-latus rectum [m]
T_days = 87.96926         # Orbital period [Earth days]
orbits_century = 100.0 * 365.25 / T_days  # ~ 415.204 orbits per century

# Analytical relativistic shift per orbit and per century
dphi_rad_phys = 3.0 * np.pi * rs_sun / p_merc
dphi_arcsec_orbit = dphi_rad_phys * (180.0 / np.pi) * 3600.0
dphi_arcsec_century = dphi_arcsec_orbit * orbits_century

print("--> Physical Parameters of Mercury's Orbit:")
print(f"    Semi-major axis (a)      : {a_merc:.3e} m ({a_merc/1.496e11:.4f} AU)")
print(f"    Eccentricity (e)         : {e_merc:.5f}")
print(f"    Solar Schwarzschild r_s  : {rs_sun:.2f} m")
print(f"    Orbital period (T)       : {T_days:.2f} days ({orbits_century:.2f} orbits/century)")
print(f"    Relativistic shift/orbit : {dphi_arcsec_orbit:.5f} arcsec ({dphi_rad_phys:.4e} rad)")
print(f"    Precession per century   : {dphi_arcsec_century:.2f} arcsec/century (Einstein 1915: 43'')")

# =============================================================================
# 2. NUMERICAL SIMULATION IN NORMALIZED BINET COORDINATES (u = 1 / r)
# =============================================================================
# In normalized units: length scale = a_merc, so p_dim = 1 - e^2
p_dim = 1.0 - e_merc**2
u_peri = 1.0 / (1.0 - e_merc)  # Initial value at perihelion (phi = 0)

# Visualization enhancement factor to clearly show the rosette advance
alpha_vis = 60000.0
rs_dim_vis = (rs_sun / a_merc) * alpha_vis
num_orbits = 8

dphi = 0.0005
phi_max = num_orbits * 2.0 * np.pi * 1.05
steps = int(phi_max / dphi)

phi_arr = np.linspace(0.0, phi_max, steps)
u_arr = np.zeros(steps)
u_arr[0] = u_peri
up = 0.0

peri_angles = [0.0]
du_prev = 0.0
phi_prev = 0.0

print(f"--> Simulating {num_orbits} orbits with visualization factor alpha = {alpha_vis:.0f}...")

# Built-in Runge-Kutta 4 (RK4) integration of the relativistic Binet ODE:
#   u'' + u = (1 / p) + 1.5 * rs_dim * u^2
for i in range(steps - 1):
    u_c = u_arr[i]
    phi_c = phi_arr[i]

    # Detect perihelion passage: u'(phi) crosses zero from positive to negative
    if du_prev > 0.0 and up <= 0.0 and phi_c > 1.0:
        frac = du_prev / (du_prev - up)
        phi_root = phi_prev + frac * dphi
        peri_angles.append(phi_root)
    du_prev = up
    phi_prev = phi_c

    def binet_deriv(st):
        u_val, up_val = st
        upp_val = (1.0 / p_dim) + 1.5 * rs_dim_vis * (u_val**2) - u_val
        return np.array([up_val, upp_val])

    st = np.array([u_c, up])
    k1 = binet_deriv(st)
    k2 = binet_deriv(st + 0.5 * dphi * k1)
    k3 = binet_deriv(st + 0.5 * dphi * k2)
    k4 = binet_deriv(st + dphi * k3)
    st += (dphi / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    u_arr[i + 1] = st[0]
    up = st[1]

# Convert inverted trajectory back to 2D physical coordinates: r = 1 / u
r_arr = 1.0 / u_arr
x_arr = r_arr * np.cos(phi_arr)
y_arr = r_arr * np.sin(phi_arr)

# Unprecessing Newtonian reference orbit (alpha = 0)
phi_newt = np.linspace(0, 2 * np.pi, 600)
r_newt = p_dim / (1.0 + e_merc * np.cos(phi_newt))
x_newt = r_newt * np.cos(phi_newt)
y_newt = r_newt * np.sin(phi_newt)

shifts_deg = [np.degrees(peri_angles[k] - 2 * np.pi * k) for k in range(len(peri_angles))]

# =============================================================================
# 3. SYSTEMATIC SCALING SCAN ACROSS ENHANCEMENT FACTORS (alpha)
# =============================================================================
print("--> Running systematic scaling scan across enhancement factors alpha...")

alpha_scan = np.array([10000, 20000, 30000, 40000, 50000, 60000])
shifts_scan_deg = []

for alf in alpha_scan:
    rs_temp = (rs_sun / a_merc) * alf
    st = np.array([u_peri, 0.0])
    up_t = 0.0
    du_p = 0.0
    ph_p = 0.0
    peri_t = 0.0

    for step in range(int(3.0 * np.pi / dphi)):
        ph = step * dphi
        if du_p > 0.0 and up_t <= 0.0 and ph > 1.0:
            frac = du_p / (du_p - up_t)
            peri_t = ph_p + frac * dphi
            break
        du_p = up_t
        ph_p = ph

        def b_der(s):
            return np.array([s[1], (1.0 / p_dim) + 1.5 * rs_temp * (s[0]**2) - s[0]])

        k1 = b_der(st)
        k2 = b_der(st + 0.5 * dphi * k1)
        k3 = b_der(st + 0.5 * dphi * k2)
        k4 = b_der(st + dphi * k3)
        st += (dphi / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        up_t = st[1]

    shifts_scan_deg.append(np.degrees(peri_t - 2 * np.pi))

shifts_scan_deg = np.array(shifts_scan_deg)
theory_scan_deg = np.degrees(alpha_scan * 3.0 * np.pi * (rs_sun / a_merc) / p_dim)

# Extrapolated physical shift per orbit and per century from numerical slope
numerical_slope = np.polyfit(alpha_scan, shifts_scan_deg, 1)[0]
extrapolated_shift_orbit = (numerical_slope / 1.0) * 3600.0  # In arcseconds
extrapolated_precession_century = extrapolated_shift_orbit * orbits_century

print(f"    Numerical extrapolated precession: {extrapolated_precession_century:.2f}''/century")
print(f"    Analytical theoretical value     : {dphi_arcsec_century:.2f}''/century")
print(f"    Agreement                        : {100.0 * (1.0 - abs(extrapolated_precession_century - dphi_arcsec_century)/dphi_arcsec_century):.4f}%")

# =============================================================================
# 4. VISUALIZATION (2 VERTICALLY ALIGNED PANELS)
# =============================================================================
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "simulate_mercury_orbit.png")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
plt.subplots_adjust(hspace=0.32)

# --- TOP PANEL: 2D Precessing Orbit (Rosette Pattern) ---
# Plot static Newtonian reference ellipse
ax1.plot(x_newt, y_newt, 'k--', lw=1.5, alpha=0.5, label='Newtonian Static Ellipse (Zero Precession)')

# Plot relativistic precessing orbits with sequential colormap
orbit_colors = plt.cm.plasma(np.linspace(0.15, 0.85, num_orbits))
pts_per_orbit = int(steps / num_orbits)

for k in range(num_orbits):
    i_start = k * pts_per_orbit
    i_end = min((k + 1) * pts_per_orbit, steps)
    lbl = 'Orbit 1' if k == 0 else ('Orbit 8' if k == num_orbits - 1 else "")
    ax1.plot(x_arr[i_start:i_end], y_arr[i_start:i_end], color=orbit_colors[k], lw=1.8, label=lbl)

# Mark the Sun at the focus
sun = Circle((0, 0), 0.05, color='gold', ec='darkorange', lw=2.0, zorder=10, label='Sun (Focus)')
ax1.add_patch(sun)

# Mark successive perihelia along the trajectory
r_peri_dist = 1.0 - e_merc
for k, phi_p in enumerate(peri_angles):
    xp = r_peri_dist * np.cos(phi_p)
    yp = r_peri_dist * np.sin(phi_p)
    ax1.plot([0, 1.25 * xp], [0, 1.25 * yp], color='crimson', linestyle=':', lw=1.0, alpha=0.6)
    ax1.scatter(xp, yp, color='red', s=45, zorder=11)
    if k == 0:
        ax1.annotate(r'Initial Perihelion ($\phi = 0^\circ$)',
                     xy=(xp, yp), xytext=(xp + 0.12, yp - 0.28),
                     fontsize=9.5, fontweight='bold', color='darkred',
                     arrowprops=dict(arrowstyle='->', color='crimson', lw=1.3))
    elif k == len(peri_angles) - 1:
        ax1.annotate(f'Orbit {k} Perihelion\n' + r'$\Delta\phi = +' + f'{shifts_deg[k]:.1f}^\\circ$',
                     xy=(xp, yp), xytext=(xp + 0.12, yp + 0.18),
                     fontsize=9.5, fontweight='bold', color='darkred',
                     arrowprops=dict(arrowstyle='->', color='crimson', lw=1.3))

ax1.set_aspect('equal')
ax1.set_xlim(-1.45, 1.55)
ax1.set_ylim(-1.25, 1.25)
ax1.set_xlabel(r'Orbital Plane $x / a$', fontsize=11, fontweight='bold')
ax1.set_ylabel(r'Orbital Plane $y / a$', fontsize=11, fontweight='bold')
ax1.set_title(rf'Step 6 (Top): Numerical Simulation of Mercury Precession (Rosette Orbit, $\alpha = {int(alpha_vis)}$)',
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower left', fontsize=9.0)

box_top = (
    r"Relativistic Binet Equation in $u = 1/r$:" + "\n"
    r"$\frac{d^2 u}{d\phi^2} + u = \frac{1}{p} + \frac{3}{2} r_s u^2$" + "\n"
    r"• Extra non-linear term $\frac{3}{2} r_s u^2$ originates from" + "\n"
    r"  vacuum self-stiffening (Step 4: $\sigma = 1/\sqrt{\eta}$)." + "\n"
    r"• Forces forward perihelion advance per revolution."
)
ax1.text(0.03, 0.72, box_top, transform=ax1.transAxes, fontsize=9.0,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='purple', alpha=0.9))

# --- BOTTOM PANEL: Linear Scaling Scan & Physical Extrapolation ---
ax2.plot(alpha_scan, shifts_scan_deg, 'ko', ms=6, label='Numerical RK4 Integration')
ax2.plot(alpha_scan, theory_scan_deg, 'r--', lw=2.0,
         label=r'Analytical First-Order GR: $\Delta\phi = \alpha \cdot \frac{3\pi r_s}{a(1-e^2)}$')

ax2.set_xlim(5000, 65000)
ax2.set_ylim(0.0, 2.0)
ax2.set_xlabel(r'Simulation Enhancement Factor $\alpha$', fontsize=11, fontweight='bold')
ax2.set_ylabel(r'Perihelion Shift per Orbit $\Delta\phi$ [degrees]', fontsize=11, fontweight='bold')
ax2.set_title('Step 6 (Bottom): Linear Scaling of Numerical Shift & Physical Extrapolation',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=9.5)

summary_box = (
    "Physical Relativistic Precession of Mercury:\n"
    rf"• Semi-Major Axis ($a$)  : ${a_merc/1e9:.3f} \times 10^6\ \mathrm{{km}}$ ($0.387\ \mathrm{{AU}}$)" + "\n"
    rf"• Orbital Eccentricity ($e$): ${e_merc:.5f}$" + "\n"
    rf"• Orbital Period ($T$)    : ${T_days:.2f}\ \mathrm{{days}}$ (${orbits_century:.2f}\ \mathrm{{orbits/century}}$)" + "\n"
    r"• Relativistic Shift per Orbit:" + "\n"
    rf"  $\Delta\phi = \frac{{3\pi r_s}}{{a(1-e^2)}} = {dphi_rad_phys:.3e}\ \mathrm{{rad}} = \mathbf{{{dphi_arcsec_orbit:.5f}^{{\prime\prime}}}}$" + "\n"
    "• Precession per Century:" + "\n"
    rf"  $\mathbf{{{dphi_arcsec_orbit:.5f}^{{\prime\prime}}}} \times {orbits_century:.2f} = \mathbf{{{dphi_arcsec_century:.2f}^{{\prime\prime}} / \mathrm{{century}}}}$" + "\n"
    r"  - Einstein GR Prediction (1915): $42.98^{\prime\prime} / \mathrm{century}$" + "\n"
    r"  - Observed Relativistic Excess : $43.1 \pm 0.5^{\prime\prime} / \mathrm{century}$"
)
ax2.text(0.40, 0.08, summary_box, transform=ax2.transAxes, fontsize=9.0,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='linen', edgecolor='darkred', alpha=0.95))

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()
print(f"--> Successfully saved figure to {output_path}")