import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

target = "61 Cygni (STF 2758 AB) | Arc: 2000-2026"

# "61 Cygni (STF 2758 AB) | Arc: 2000-2026"
# "Kruger 60 AB (DO Cephei) | Arc: 2020-2026"
unit = 'arcsec'
last_index = -1
data = np.genfromtxt('61cyg.csv', delimiter=',', skip_header=1)

theta = np.deg2rad(data[:, 0])
r = data[:, 1]
x = r * np.cos(theta)
y = r * np.sin(theta)

dates = data[:, 2]



"""# Get x and y columns
x = data[:, 0]
y = data[:, 1]

# Convert to polar coordinates (origin = primary star = focus)
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)
"""

# --- Orbit model: ellipse in polar form with focus at origin ---
# r = a(1 - e^2) / (1 + e*cos(theta - omega))
def orbit_model(theta, a, e, omega):
    return a * (1 - e**2) / (1 + e * np.cos(theta - omega))


def calc_residuals():
    r_obs_fit = orbit_model(theta, *popt)
    residuals = r - r_obs_fit
    return np.sum(residuals**2)

omega_list = []
residual_list = []

bounds = ([0, 0, 0], [np.inf, 0.9999, 2*np.pi])
count = 0
for omega_guess in np.linspace(0, 2*np.pi, 50):
    p0 = [np.mean(r), (r.max()-r.min())/(r.max()+r.min()), omega_guess]
    popt, pcov = curve_fit(orbit_model, theta, r, p0=p0, bounds=bounds)
    a_fit, e_fit, omega_fit = popt
    omega_list.append(omega_fit)
    residual_list.append(calc_residuals())
    count+=1
    print(f"    {count}/50: Argument of Periapsis Guess: {np.degrees(omega_guess)} deg: Done")
omega_bestguess = omega_list[residual_list.index(min(residual_list))]
p0 = [np.mean(r), (r.max()-r.min())/(r.max()+r.min()), omega_bestguess]
popt, pcov = curve_fit(orbit_model, theta, r, p0=p0, bounds=bounds)
a_fit, e_fit, omega_fit = popt 

# Get total residual for fitted orbit R^2
ss_res = calc_residuals()
ss_tot = np.sum((r - np.mean(r))**2)
r_squared = 1 - (ss_res / ss_tot)


"""
# -------------------------------------
# estimate orbital period based on observational data

t_first = data[0][2] # year of first observation
t_last = data[last_index][2] # year of second observation

rho_first = data[0][1] 
rho_last = data[last_index][1] 

first_theta = data[0][0] - np.degrees(omega_fit) # degrees
last_theta = data[last_index][0] - np.degrees(omega_fit) # degrees


E_first = 2 * np.arctan(np.sqrt((1-e_fit)/(1+e_fit)) * np.tan(np.deg2rad(first_theta)/2))
E_last = 2 * np.arctan(np.sqrt((1-e_fit)/(1+e_fit)) * np.tan(np.deg2rad(last_theta)/2))


def kepler_eq(E, e):
    return E - e * np.sin(np.deg2rad(E))

M_first = kepler_eq(E_first, e_fit)
M_last = kepler_eq(E_last, e_fit)

P = abs((2 * np.pi) * (t_last - t_first) / (M_last - M_first))

# print(f"Estimated orbital period: {P:.3f} years")
t_i = 0 # time of first observation (arbitrary)


E_i = 2 * np.arctan(np.sqrt((1-e_fit)/(1+e_fit)) * np.tan(np.radians(last_theta)/2))
t0 = abs(t_i - (E_i - e_fit * np.sin(E_i)) * (P * 365.25 * 24 * 60**2) / (2 * np.pi))

if rho_last - rho_first >= 0:
    t_yrnextperiapsis = t_last-t0/(365.25 * 24 * 60**2) + P
    t_yrnextapoapsis = t_last-t0/(365.25 * 24 * 60**2) + P/2
    t_yrlastperiapsis = t_last-t0/(365.25 * 24 * 60**2)
    t_yrlastapoapsis = t_last-t0/(365.25 * 24 * 60**2) - P/2

else:
    t_yrnextperiapsis = t_last+t0/(365.25 * 24 * 60**2)
    t_yrnextapoapsis = t_last+t0/(365.25 * 24 * 60**2) + P/2
    t_yrlastperiapsis = t_last+t0/(365.25 * 24 * 60**2) - P
    t_yrlastapoapsis = t_last+t0/(365.25 * 24 * 60**2) - P/2

# -------------------------------------
"""


# --- Plot ---
theta_dense = np.linspace(0, 2 * np.pi, 1000)
r_fit = orbit_model(theta_dense, *popt)
x_fit = r_fit * np.cos(theta_dense)
y_fit = r_fit * np.sin(theta_dense)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot fitted orbit
ax.plot(x_fit, y_fit, color='black', label='Sky-Projected Orbit Fit', zorder=1)

# Plot fitted periapsis and apoapsis points
peri_x = a_fit * (1 - e_fit) * np.cos(omega_fit)
peri_y = a_fit * (1 - e_fit) * np.sin(omega_fit)
apo_x = a_fit * (1 + e_fit) * np.cos(np.pi + omega_fit)
apo_y = a_fit * (1 + e_fit) * np.sin(np.pi + omega_fit)
ax.scatter(peri_x, peri_y, color='green', marker='+', s=100, label='Periapsis', zorder=2)
ax.scatter(apo_x, apo_y, color='purple', marker='+', s=100, label='Apoapsis', zorder=3)

# Plot observations
ax.scatter(x, y, color='blue', marker = "x",zorder=4, label='Observations', s=20) 

# Plot primary star at origin
ax.scatter(0, 0, color='red', s=200, marker='o', zorder=5, label='_Hidden Item')

# Plot secondary star at last observed position
ax.scatter(x[-1], y[-1], color='orange', s=100, marker = 'o', zorder=6, label='_Hidden Item')


ax.set_xlabel(f'→ N  ({unit})',)
ax.set_ylabel(f'→ E ({unit})')

# ax.set_xlim(-1.5*a_fit, 1.5*a_fit)
# ax.set_ylim(-1.5*a_fit, 1.5*a_fit)

"""
Removed orbital period data, fix later
Period = {P:.3f} yrs,\nLast Periapsis Yr: {t_yrlastperiapsis:.3f}, Last Apoapsis Yr: {t_yrlastapoapsis:.3f}\nNext Periapsis Yr: {t_yrnextperiapsis:.3f}, Next Apoapsis Yr: {t_yrnextapoapsis:.3f}
"""

title = f'{target}\nn = {len(x)}, R² = {r_squared:.6f}\na = {a_fit:.3f} {unit}, e = {e_fit:.3f}, ω = {np.degrees(omega_fit):.3f} deg,\nPeriapsis = {a_fit*(1-e_fit):.3f} {unit}, Apoapsis = {a_fit*(1+e_fit):.3f} {unit}'
ax.set_title(title, fontsize=8, pad=4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(-0.1, -0.1), loc='upper left', fontsize='small', markerscale = 1, ncols = 4)
#plt.tight_layout()
plt.savefig('orbit_fit.png', dpi=750, bbox_inches='tight')

# print(f"Fitted orbital parameters:")
# print(f"  Semi-major axis  a = {a_fit:.4f} {unit}")
# print(f"  Eccentricity     e = {e_fit:.4f}")
# print(f"  Arg. of peri.    ω = {np.degrees(omega_fit):.4f} deg")
# print(f"  Periapsis dist.    = {a_fit*(1-e_fit):.4f} {unit}, ({peri_x:.4f}, {peri_y:.4f}) {unit}")
# print(f"  Apoapsis dist.     = {a_fit*(1+e_fit):.4f} {unit}, ({apo_x:.4f}, {apo_y:.4f}) {unit}")

print("Plot saved to orbit_fit.png")

