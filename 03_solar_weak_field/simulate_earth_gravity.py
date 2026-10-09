"""
Step 3B: Earth Gravity Profile & Acceleration from Vacuum Gradient
------------------------------------------------------------------
Demonstrates how everyday gravitational acceleration emerges directly from
the spatial gradient of the coordinate speed of light c_eff and vacuum density eta:
    a_vec = - c_eff * nabla(c_eff) = - 0.5 * nabla(c_eff^2) ~ + c_0^2 * nabla(eta)

Models both the interior (uniform density sphere) and exterior of Earth:
1. Interior (r < R_earth):
       Phi(r) = - (G * M) / (2 * R) * [ 3 - (r/R)^2 ]
       a(r)   = - (G * M * r) / R^3
2. Exterior (r >= R_earth):
       Phi(r) = - (G * M) / r
       a(r)   = - (G * M) / r^2

Optical & Density Analogy:
    c_eff(r) = c_0 * sqrt(1 + 2*Phi/c_0^2)
    Delta c_eff(r) = c_eff(r) - c_0  (Coordinate speed deficit)
    Delta eta(r) = - Delta c_eff(r) / c_0  (Vacuum packing excess in ppb)

Demonstrates:
- At Earth Core (r = 0): Density peaks (+1.04 ppb) & speed dips (-0.313 m/s),
  but slope is zero -> a = 0 m/s^2 (weightless).
- At Earth Surface (r = R_earth): Gradient nabla(c_eff) is steepest -> peak acceleration g = 9.82 m/s^2.
- Far Field (r >> R_earth): Follows exact inverse-square law a(r) = -GM/r^2.

Dependencies:
    numpy, matplotlib (Zero external dependencies!)

Author: Bart Leplae
Project: gravity-as-relaxation
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PHYSICAL CONSTANTS (EARTH)
# =============================================================================
G = 6.67430e-11         # Gravitational constant [m^3 / (kg s^2)]
M_earth = 5.9722e24     # Mass of the Earth [kg]
R_earth = 6371000.0     # Radius of the Earth [m]
c0 = 299792458.0        # Speed of light in deep vacuum [m/s]

# 1D slice along X-axis through the center of the Earth (-4 R_earth to +4 R_earth)
N_points = 1000
x_radii = np.linspace(-4.0, 4.0, N_points)
x_meters = x_radii * R_earth
r = np.abs(x_meters)

# =============================================================================
# 2. POTENTIAL, ACCELERATION & VACUUM DENSITY FIELDS
# =============================================================================
phi = np.zeros_like(x_meters)
a_analytic = np.zeros_like(x_meters)

outside = r >= R_earth
inside = r < R_earth

# 1. Outside the Earth
phi[outside] = - (G * M_earth) / r[outside]
a_analytic[outside] = - (G * M_earth) / (r[outside]**2) * np.sign(x_meters[outside])

# 2. Inside the Earth (Uniform density sphere)
phi[inside] = - (G * M_earth) / (2.0 * R_earth) * (3.0 - (r[inside]**2 / R_earth**2))
a_analytic[inside] = - (G * M_earth * r[inside]) / (R_earth**3) * np.sign(x_meters[inside])

# Coordinate speed of light: c_eff = c0 * sqrt(1 + 2*Phi/c0^2)
c_eff = c0 * np.sqrt(1.0 + (2.0 * phi) / (c0**2))
delta_c = c_eff - c0  # Coordinate speed deficit [m/s]

# Vacuum packing density excess Delta eta = eta - 1 [parts per billion, ppb]
delta_eta_ppb = (- delta_c / c0) * 1e9

surface_g = (G * M_earth) / (R_earth**2)
core_change = (c0 * np.sqrt(1.0 + (2.0 * (- (G * M_earth) / (2.0 * R_earth) * 3.0)) / c0**2)) - c0
core_eta_ppb = (- core_change / c0) * 1e9

surface_eta_ppb = delta_eta_ppb[np.argmin(np.abs(x_radii - 1.0))]
surface_c_change = delta_c[np.argmin(np.abs(x_radii - 1.0))]

print("--> Physical Parameters of Earth's Coordinate Speed & Density Landscape:")
print(f"    Surface Gravity               : {surface_g:.2f} m/s^2")
print(f"    Surface Coordinate Deficit    : {surface_c_change:.3f} m/s")
print(f"    Surface Density Excess        : {surface_eta_ppb:.2f} ppb")
print(f"    Core Max Coordinate Deficit   : {core_change:.3f} m/s")
print(f"    Core Max Density Excess       : {core_eta_ppb:.2f} ppb")

# =============================================================================
# 3. VISUALIZATION
# =============================================================================
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "simulate_earth_gravity.png")

fig, ax1 = plt.subplots(figsize=(12, 7))
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 65))

# --- Axis 1: Acceleration Vector (Green) ---
color1 = 'tab:green'
ax1.set_xlabel(r'Distance from Earth Center (Earth Radii, $R_{\oplus}$)', fontsize=12, fontweight='bold')
ax1.set_ylabel(r'Acceleration: $\vec{a} = -c_{\mathrm{eff}} \nabla c_{\mathrm{eff}} = -\frac{1}{2}\nabla(c_{\mathrm{eff}}^2)$ [$\mathrm{m/s}^2$]',
               color=color1, fontsize=11, fontweight='bold')
l1 = ax1.plot(x_radii, a_analytic, color=color1, linewidth=2.8, label=r'Acceleration Vector $\vec{a}$')
ax1.tick_params(axis='y', labelcolor=color1)

# --- Axis 2: Coordinate Speed Deficit (Blue Dashed) ---
color2 = 'tab:blue'
ax2.set_ylabel(r'Coordinate Speed Deficit: $\Delta c_{\mathrm{eff}} = c_{\mathrm{eff}} - c_0$ [m/s]',
               color=color2, fontsize=11, fontweight='bold')
l2 = ax2.plot(x_radii, delta_c, color=color2, linewidth=2.5, linestyle='--', label=r'Coordinate Speed Deficit $\Delta c_{\mathrm{eff}}$')
ax2.tick_params(axis='y', labelcolor=color2)

# --- Axis 3: Vacuum Packing Density Excess (Purple Dotted) ---
color3 = 'purple'
ax3.set_ylabel(r'Vacuum Packing Excess: $\Delta\eta \equiv \eta - 1$ [parts per billion, ppb]',
               color=color3, fontsize=11, fontweight='bold')
l3 = ax3.plot(x_radii, delta_eta_ppb, color=color3, linewidth=2.4, linestyle=':', label=r'Vacuum Density Excess $\Delta\eta$')
ax3.tick_params(axis='y', labelcolor=color3)

# Highlight Core Values
ax2.scatter([0.0], [core_change], color='blue', zorder=5, s=55)
ax3.scatter([0.0], [core_eta_ppb], color='purple', zorder=5, s=55)

# Highlight Surface Values
ax1.scatter([1.0, -1.0], [-surface_g, surface_g], color='red', zorder=5, s=55)

# Earth Interior Shading
ax1.axvspan(-1.0, 1.0, color='saddlebrown', alpha=0.12, label="Earth Interior")
ax1.axvline(-1.0, color='gray', linestyle=':', lw=1.5)
ax1.axvline(1.0, color='gray', linestyle=':', lw=1.5)

# Callout 1: Earth Core Box
core_box_text = (
    f"Earth Core (Max Density):\n"
    f"• Density $\\Delta\\eta = +{core_eta_ppb:.2f}$ ppb\n"
    f"• Min Speed $\\Delta c_{{\\mathrm{{eff}}}} = {core_change:.3f}$ m/s\n"
    f"• Slope $\\nabla c = 0 \\rightarrow a = 0$ m/s$^2$"
)
ax2.annotate(core_box_text, 
             xy=(0.0, core_change), xytext=(-3.75, core_change * 0.42),
             fontsize=9.2, fontweight='bold', color='navy',
             arrowprops=dict(facecolor='blue', edgecolor='navy', shrink=0.05, width=1.2, headwidth=6),
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='blue', alpha=0.9))

# Callout 2: Earth Surface Box (Mirrored structure)
surface_box_text = (
    f"Earth Surface ($|x| = 1\\,R_{{\\oplus}}$):\n"
    f"• Density $\\Delta\\eta = +{surface_eta_ppb:.2f}$ ppb\n"
    f"• Speed $\\Delta c_{{\\mathrm{{eff}}}} = {surface_c_change:.3f}$ m/s\n"
    f"• Max Slope $\\nabla c \\rightarrow a = \\pm{surface_g:.2f}$ m/s$^2$"
)
ax1.annotate(surface_box_text, 
             xy=(1.0, -surface_g), xytext=(1.65, -surface_g + 2.0),
             fontsize=9.2, fontweight='bold', color='darkred',
             arrowprops=dict(facecolor='red', edgecolor='darkred', shrink=0.05, width=1.2, headwidth=6),
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='red', alpha=0.9))

ax1.set_title(r'Earth Gravity Profile: Vacuum Packing $\eta$, Coordinate Speed $c_{\mathrm{eff}}$, and Acceleration $\vec{a}$',
              fontsize=13, fontweight='bold', pad=15)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_xlim(-4.0, 4.0)
ax1.set_ylim(-12.0, 12.0)

# Unified Legend
lines = l1 + l2 + l3
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='lower left', framealpha=0.9, fontsize=9.0)

# Physical Foundation Box
box_text = (
    r"Microscopic-to-Macroscopic Bridge:" + "\n"
    r"• Vacuum Density: $\eta \equiv \psi^2 = 1 + \Delta\eta$" + "\n"
    r"• Speed Relation: $\Delta c_{\mathrm{eff}} \approx -c_0 \Delta\eta$ (mirrored inverse)" + "\n"
    r"• Inward Pull: $\vec{a} = -c_{\mathrm{eff}} \nabla c_{\mathrm{eff}} \approx +c_0^2 \nabla\eta$" + "\n"
    r"• Physical Law: Matter accelerates toward higher vacuum density $\eta$!"
)
ax1.text(0.48, 0.70, box_text, transform=ax1.transAxes, fontsize=8.8,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='linen', edgecolor='darkgreen', alpha=0.95))

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()
print(f"--> Successfully saved figure to {output_path}")