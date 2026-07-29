"""
Visual binary orbital-element solver iterating over Thiele-Innes Method
7/28/2026
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import time


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# 0=P 1=T 2=e 3=a" 4=i 5=Omega 6=omega 7=M_total 8=cost 9=R2

start_time = time.perf_counter()

# Target Name
target = "Sirius (HD 48915)"
unit = '"' # arcsec
csv_file = "csv data/test.csv"

# Fitting mode:
#   "grid"  -> scan P over a grid; at each fixed P fit the other 6 elements.
#              This is the correct way to MAP the degenerate range of orbits
#              for a short observation arc.
fit_mode = "grid"

bool_plot_orbits = False

P_lower = 1e-1 # years
P_upper = 200  # years

# --- grid-mode settings ---
n_P_grid = 100      # number of periods to scan across [P_lower, P_upper]
P_grid_log = False        # log-spaced grid (better when P spans decades)
n_restarts_per_P = 10    # random restarts of the 6 free elements at each fixed P

# Cost threshold for "acceptable" orbits, expressed as a multiple of the
# best cost found. Orbits with cost <= accept_factor * best_cost are counted
# as members of the acceptable family when reporting the range.
accept_factor = 1.5 # Best
m_total_frac_accept = 0.1 # Mass

# Inputs for period/semi-major axis constrainments based on spectroscopic data (optional)
m1_guess = 2.17
m2_guess = 1.00

# ----------------------------------------------------------------------
#          P (yr)   T (yr)   e    a (")  i      Omega  omega
lower = [P_lower, 0.0, 0.0, 1e-3, 0.0, 0.0, 0.0]
upper = [P_upper, 3000.0, 0.999, 100.0, np.pi, 2*np.pi, 2*np.pi]

# ----------------------------------------------------------------------

# Constants (DO NOT CHANGE)
M_Sun = 1.98847e30  # kg
AU = 1.495978707e11  # m
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
data = np.genfromtxt(csv_file, delimiter=",", skip_header=1, usecols = (0, 1, 2))
theta_obs = np.deg2rad(data[:, 0])   # position angle, N through E
rho_obs = data[:, 1]                 # separation (arcsec)
t_obs = data[:, 2]                   # epoch (decimal year)

x_obs = rho_obs * np.cos(theta_obs)  
y_obs = rho_obs * np.sin(theta_obs)  


# Parallax lives only on the first data row, column 4
first_row = np.genfromtxt(csv_file, delimiter=",", skip_header=1, max_rows=1)

parallax_mas = first_row[3]
parallax_arcsec = parallax_mas / 1000.0


mass_constrain = False
m_total_guess = None
if (m1_guess is not None and m2_guess is not None):
    mass_constrain = True
    m_total_guess = m1_guess + m2_guess

folder_name = f'{target}_{t_obs[0]}_{t_obs[-1]}_{n_P_grid}_{n_restarts_per_P}_{mass_constrain}'
os.makedirs(folder_name, exist_ok=True)

logname = f'{folder_name}/logfile_{target}_{n_P_grid}_{n_restarts_per_P}_{mass_constrain}.txt'

with open(logname, "w") as f:
    f.write(f'Grid mode: {n_P_grid} periods in [{P_lower}, {P_upper}] yr, {n_restarts_per_P} Iterations each, Mass Constrain = {mass_constrain}\nTotal Mass Fractional Acceptance = {m_total_frac_accept}, Best Cost Accept Factor = {accept_factor}\nPrimary Star (m1) Mass Guess: {m1_guess} MSol, Secondary Star (m2) Mass Guess: {m2_guess} MSol\n')





# ----------------------------------------------------------------------
# Kepler Equation
# ----------------------------------------------------------------------
def solve_kepler(M, e, tol=1e-12, itmax=100):
    """Solve Kepler's equation M = E - e sin E (vectorized Newton)."""
    M = np.mod(M, 2 * np.pi)
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(itmax):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def thiele_innes(a, i, Omega, omega):
    """Campbell elements -> Thiele-Innes constants A, B, F, G."""
    cO, sO = np.cos(Omega), np.sin(Omega)
    co, so = np.cos(omega), np.sin(omega)
    ci = np.cos(i)
    A = a * (co * cO - so * sO * ci)
    B = a * (co * sO + so * cO * ci)
    F = a * (-so * cO - co * sO * ci)
    G = a * (-so * sO + co * cO * ci)
    return A, B, F, G


def model_xy(params, t):
    """Predicted sky position (x=N, y=E) at epochs t for the 7 elements."""
    P, T, e, a, i, Omega, omega = params
    M = 2 * np.pi * (t - T) / P
    E = solve_kepler(M, e)
    # Elliptical rectangular coordinates in the true orbital plane
    X = np.cos(E) - e
    Y = np.sqrt(1 - e**2) * np.sin(E)
    A, B, F, G = thiele_innes(a, i, Omega, omega)
    x = A * X + F * Y   # North
    y = B * X + G * Y   # East
    return x, y


def residuals(params, t, x, y):
    xm, ym = model_xy(params, t)
    return np.concatenate([xm - x, ym - y])


def plot_orbits(P, T, e, a, i, Omega, omega, M_total, cost, r_squared, index):
    # ----------------------------------------------------------------------
    # Plotting: Sky-Projected Orbit Fit and True Orbit Fit
    # ----------------------------------------------------------------------
    # Angles arrive in DEGREES (as stored by record()); convert to radians
    # for all trig / Thiele-Innes math. Keep degree copies for the titles.
    i_deg, Omega_deg, omega_deg = i, Omega, omega
    i, Omega, omega = np.radians(i), np.radians(Omega), np.radians(omega)

    E_dense = np.linspace(0, 2 * np.pi, 2000)
    X_d = np.cos(E_dense) - e
    Y_d = np.sqrt(1 - e**2) * np.sin(E_dense)
    A_, B_, F_, G_ = thiele_innes(a, i, Omega, omega)
    x_fit = A_ * X_d + F_ * Y_d
    y_fit = B_ * X_d + G_ * Y_d

    # Periastron: E = 0
    x_peri, y_peri = A_ * (1 - e), B_ * (1 - e)
    # Apastron: E = pi
    x_apo, y_apo = A_ * (-1 - e), B_ * (-1 - e)

    peri_sky = np.hypot(x_peri, y_peri)  
    apo_sky  = np.hypot(x_apo,  y_apo)  

    # Set up Sky-Projected Orbit Fit and True Orbit Fit subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 7),
                                layout="constrained")
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    # Plot Primary Star
    ax1.scatter(0, 0, color="red", s=180, marker="o", label="Primary",
                zorder=7)


    ax1.scatter(x_obs[-1], y_obs[-1], color="orange", s=120, marker="o",
                label="Secondary", zorder=6)

    ax1.scatter(x_obs, y_obs, c="tab:blue", marker="x", s=25,
                label="Observations", zorder=5)

    # Plot Periastron and Apastron on ax1
    ax1.scatter(x_peri, y_peri, color="green", marker="x", s=120,
            label="Periastron", zorder=4)
    ax1.scatter(x_apo, y_apo, color="purple", marker="x", s=120,
            label="Apastron", zorder=3)

    # Line of nodes (intersection of orbital and sky planes)
    normal = np.array([-np.sin(Omega), np.cos(Omega)])
    node_vals = normal[0] * x_fit + normal[1] * y_fit
    crossings = np.where(np.sign(node_vals[:-1]) != np.sign(node_vals[1:]))[0]
    node_points = []
    node_types = []

    # Compute z coordinate along the true orbit to distinguish ascending vs descending nodes
    z_fit = np.sin(i) * (np.sin(omega) * X_d + np.cos(omega) * Y_d)
    for j in crossings:
        t = node_vals[j] / (node_vals[j] - node_vals[j + 1])
        x_int = x_fit[j] + t * (x_fit[j + 1] - x_fit[j])
        y_int = y_fit[j] + t * (y_fit[j + 1] - y_fit[j])
        node_points.append((x_int, y_int))
        if z_fit[j] < 0 and z_fit[j + 1] > 0:
            node_types.append("ascending")
        elif z_fit[j] > 0 and z_fit[j + 1] < 0:
            node_types.append("descending")
        else:
            node_types.append("unknown")

    if len(node_points) >= 2:
        node_x = np.array([node_points[0][0], node_points[1][0]])
        node_y = np.array([node_points[0][1], node_points[1][1]])
    else:
        node_x = np.array([-1 * a, 1 * a]) * np.cos(Omega) * 1.2
        node_y = np.array([-1 * a, 1 * a]) * np.sin(Omega) * 1.2
    ax1.plot(node_x, node_y, "--", color="gray", lw=0.8, label="Line of nodes", zorder = 2)

    # Plot Sky-Projected Orbit Fit  (x-axis = East, y-axis = North)
    ax1.plot(x_fit, y_fit, "k-", lw=1.2, zorder=1)


    ax1.set_xlabel(f"$\\rightarrow$N ({unit})")
    ax1.set_ylabel(f"$\\rightarrow$E ({unit})")
    ax1.grid(True, alpha=0.3)

    fig.legend(fontsize=8, ncol=6, loc="outside lower center", frameon=False, borderaxespad=0.2)
    # Plot true orbit ---------------------------------------
    def polar_ellipse(a_AU, e, theta):
        return a_AU * (1 - e**2) / (1 - e * np.cos(theta-np.pi))

    ax2.grid(True, alpha=0.3)

    # CONVERT SMA TO PHYSICAL UNITS
    a_AU = a * d_pc
    M_total = a_AU**3 / P**2

    theta = np.linspace(0, 2 * np.pi, 1000)
    r = polar_ellipse(a_AU, e, theta)
    x_true = r * np.cos(theta)
    y_true = r * np.sin(theta)

    # Plot True Orbit
    ax2.plot(x_true, y_true, "k-", lw=1.2, label="True Orbit", zorder=1)

    # Plot Primary Star on ax2
    ax2.scatter(0, 0, color="red", s=180, marker="o", label="Primary",
                zorder=5)

    # Plot Periastron and Apastron on ax2
    ax2.scatter(a_AU * (1 - e), 0, color="green", marker="x", s=120,
            label="Periastron", zorder=3)
    ax2.scatter(a_AU * (-1 - e), 0, color="purple", marker="x", s=120,
            label="Apastron", zorder=3)

    # Plot secondary star and observations on the True orbital plane
    det = A_ * G_ - B_ * F_
    X_obs = ( G_ * x_obs - F_ * y_obs) / det
    Y_obs = (-B_ * x_obs + A_ * y_obs) / det



    # Convert to AU
    X_obs_AU = a_AU * X_obs
    Y_obs_AU = a_AU * Y_obs


    # Get true anomaly at last observation
    true_anom = np.arctan2(Y_obs_AU[-1], X_obs_AU[-1])

    ax2.scatter(X_obs_AU, Y_obs_AU, c="tab:blue", marker="x", s=25,
                label="Observations", zorder=4)
    ax2.scatter(X_obs_AU[-1], Y_obs_AU[-1], color="orange", s=120, marker="o",
                label="Secondary", zorder=5)


    ax2.set_xlabel("X (AU)")
    ax2.set_ylabel("Y (AU)")

    T = t_obs[0] - np.mod(t_obs[0] - T, P)

    fig.suptitle(
        f'{target}\n'
        f'Obs Arc: {t_obs[0]:.0f} - {t_obs[-1]:.0f} | Orbit Fits = {n_P_grid}, Iterations = {n_restarts_per_P}\n'
        f'R² = {r_squared}, Cost = {cost}, Mass Constrain = {mass_constrain}\n\n'
        f'Parallax = {parallax_mas:.4f} mas, Distance = {d_pc:.2f} pc, $M_{{total}}$ = {M_total:.3f} M$_\\odot$\n'
        f'P = {P:.3f} yr, T = {T:.3f} yr'
        , fontsize=10)

    ax1.set_title(
        f'Sky-Projected Orbit Fit\n'
        f'a = {a:.3f}{unit}, e = {e:.3f}, '
        f'i = {i_deg:.3f}$^\\circ$, $\\Omega$ = {Omega_deg:.3f}$^\\circ$, $\\omega$ = {omega_deg:.3f}$^\\circ$'
        f'\nApastron = {apo_sky:.3f}{unit}, Periastron = {peri_sky:.3f}{unit}'
        , fontsize = 9
    )
    ax2.set_title(
        f'True Orbit Fit\n'
        f'a = {a_AU:.3f} AU, e = {e:.3f}, $\\nu$ = {np.degrees(true_anom):.3f}$^\\circ$\n'
        f'Apastron = {a_AU*(1+e):.3f} AU, Periastron = {a_AU*(1-e):.3f} AU'
        , fontsize = 9
    )

    fig.savefig(f'{folder_name}/orbit_fit{index}.png', dpi=200, bbox_inches='tight')
    
    with open(logname, "a") as f:
        f.write(f'Plot saved to orbit_fit{index}.png\n')
    plt.close(fig)
# ----------------------------------------------------------------------
# Fit: multi-start nonlinear least squares
# ----------------------------------------------------------------------
d_pc = 1.0 / parallax_arcsec

ss_tot = np.sum((x_obs - x_obs.mean())**2) + np.sum((y_obs - y_obs.mean())**2)


def fit_six_at_fixed_P(P_fixed, n_restarts, rng):
    """
    Fit the six non-period elements (T, e, a, i, Omega, omega) with P held
    fixed, using several random restarts. Returns the lowest-cost
    OptimizeResult for this P, or None if every restart failed.

    P is pinned by giving least_squares a razor-thin bound around P_fixed,
    so we reuse the existing 7-parameter model/residual functions unchanged.
    """
    eps = max(P_fixed * 1e-9, 1e-9)
    lo = [P_fixed - eps, lower[1], lower[2], lower[3], lower[4], lower[5], lower[6]]
    hi = [P_fixed + eps, upper[1], upper[2], upper[3], upper[4], upper[5], upper[6]]

    local_best = None
    for _ in range(n_restarts):
        p0 = [
            P_fixed,
            rng.uniform(0.0, 3000.0),      # T
            rng.uniform(0.0, 0.95),        # e
            rng.uniform(0.1, 60.0),        # a (arcsec)
            rng.uniform(0.0, np.pi),       # i
            rng.uniform(0.0, 2*np.pi),     # Omega
            rng.uniform(0.0, 2*np.pi),     # omega
        ]
        try:
            sol = least_squares(residuals, p0,
                                args=(t_obs, x_obs, y_obs),
                                bounds=(lo, hi), method="trf",
                                x_scale="jac", max_nfev=2000)
        except Exception as ex:
            continue
        if local_best is None or sol.cost < local_best.cost:
            local_best = sol
    return local_best


def record(sol):
    P, T, e, a, i, Omega, omega = sol.x
    r = residuals(sol.x, t_obs, x_obs, y_obs)
    ss_res = np.sum(r**2)
    r_squared = 1 - ss_res / ss_tot
    a_AU = a * d_pc
    M_total = a_AU**3 / P**2   # solar masses (Kepler III, relative orbit)
    return [P, T, e, a, np.degrees(i), np.degrees(Omega),
            np.degrees(omega), M_total, sol.cost, r_squared]


# ----------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------
rng = np.random.default_rng()
fitted_values = []

if P_grid_log:
    P_values = np.geomspace(P_lower, P_upper, n_P_grid)
else:
    P_values = np.linspace(P_lower, P_upper, n_P_grid)

"""print(f'Grid mode: {n_P_grid} periods '
      f'in [{P_lower}, {P_upper}] yr, '
      f'{n_restarts_per_P} restarts each.')"""

for gi, P_fixed in enumerate(P_values):
    sol = fit_six_at_fixed_P(P_fixed, n_restarts_per_P, rng)
    if sol is None:
        # print(f'  [{gi+1}/{n_P_grid}] P={P_fixed:8.2f} yr  -> no fit')
        continue
    row = record(sol)
    fitted_values.append(row)
    """print(f'    [{gi+1}/{n_P_grid}] P = {P_fixed:.2f} yr '
          f'a = {row[3]:.3f}"  e = {row[2]:.3f}  '
          f'M = {row[7]:.3f} Msun  cost = {row[8]}')"""
    with open(logname, "a") as f:
        f.write(f'  [{gi+1}/{n_P_grid}] P = {P_fixed:8.2f} yr, a = {row[3]:.3f}"  e = {row[2]:.3f}, M = {row[7]:.3f} Msun  cost = {row[8]}\n')

    if bool_plot_orbits:
        # row cols: 0=P 1=T 2=e 3=a" 4=i(deg) 5=Omega(deg) 6=omega(deg)
        #           7=M_total 8=cost 9=R2   -- plot_orbits converts angles.
        P, T, e, a, i, Omega, omega, M_Total, cost, r2 = row
        plot_orbits(P, T, e, a, i, Omega, omega, M_Total, cost, r2, f'_{P_fixed:.2f}')

if not fitted_values:
    raise RuntimeError("No successful fits — check data and bounds.")


fitted_values = np.array(fitted_values, dtype=float)
#  col: 0=P 1=T 2=e 3=a" 4=i 5=Omega 6=omega 7=M_total 8=cost 9=R2

# --- best fit ---
best_idx = int(np.argmin(fitted_values[:, 8]))
best_fit = fitted_values[best_idx]
best_cost = best_fit[8]

# The range of orbits consistent with the arc ---
accept = fitted_values[fitted_values[:, 8] <= accept_factor * best_cost]

# The range of orbits consistent with total mass guess
if (mass_constrain == True):
    accept = accept[accept[:, 7] <= (1+m_total_frac_accept) * m_total_guess]
    accept = accept[accept[:, 7] >= (1-m_total_frac_accept) * m_total_guess]
# accept is the list of orbits that fit cost and mass constraints

if len(accept) == 0:
    with open(logname, "a") as f:
        f.write(f'No orbits pass the cost + mass constraints. Widen accept_factor or m_total_frac_accept.\n')
    best_accept_idx = best_idx
    best_accept_fit = fitted_values[best_accept_idx]
    # Then plot lowest cost orbit
    P, T, e, a, i, Omega, omega, M_total, cost, r_squared = best_accept_fit
    plot_orbits(P, T, e, a, i, Omega, omega, M_total, cost, r_squared, "_lowest_cost")

    # raise RuntimeError("No orbits pass the cost + mass constraints; "
    #                    "widen accept_factor or m_total_frac_accept. Plotting lowest cost found.")
else:
    best_accept_idx = int(np.argmin(accept[:, 8]))
    best_accept_fit = accept[best_accept_idx]

# print('-' * 55)
if (mass_constrain == True):
    """print(f'ACCEPTABLE ORBITS  (cost <= {accept_factor * best_cost}), ({(1-m_total_frac_accept) *  m_total_guess} <= m_total <= {(1+m_total_frac_accept) *  m_total_guess}): '
      f'{len(accept)} of {len(fitted_values)} orbits')"""
    with open(logname, "a") as f:
        f.write(f'ACCEPTABLE ORBITS  (cost <= {accept_factor * best_cost}), ({(1-m_total_frac_accept) *  m_total_guess} <= m_total <= {(1+m_total_frac_accept) *  m_total_guess}): '
      f'{len(accept)} of {len(fitted_values)} orbits\n')
else:
    """print(f'ACCEPTABLE ORBITS  (cost <= {accept_factor * best_cost}): '
      f'{len(accept)} of {len(fitted_values)} orbits')"""
    with open(logname, "a") as f:
        f.write(f'ACCEPTABLE ORBITS  (cost <= {accept_factor * best_cost}): '
      f'{len(accept)} of {len(fitted_values)} orbits\n')

labels = ['P (yr)', 'T (yr)', 'e', f'a ({unit})', 'i (deg)',
          'Omega (deg)', 'omega (deg)', 'M_tot (Msun)']
if len(accept) != 0:
    for c, lab in enumerate(labels):
        col = accept[:, c]
        """print(f'  {lab:14s} range [{col.min():10.3f}, {col.max():10.3f}]  '
            f'median {np.median(col):10.3f}')"""
        with open(logname, "a") as f:
            f.write(f'  {lab:14s} range [{col.min():10.3f}, {col.max():10.3f}]  '
            f'median {np.median(col):10.3f}\n')


# -----------------------------------------------------------------
orbital_periods = fitted_values[:, 0].tolist()
orbital_periods_accepted = accept[:,0].tolist()

costs = fitted_values[:, 8].tolist()
accept_costs = accept[:, 8].tolist()

# Plot cost vs Period: the key degeneracy diagnostic.
plt.figure(figsize=(10, 6))
plt.xlabel("Orbital Period (yr)")
plt.ylabel("Cost")

if mass_constrain == False:
    plt.title(f'{target} | {t_obs[0]} - {t_obs[-1]}\n'
        f'mode = {fit_mode} | {len(fitted_values)} Orbit Fits, {n_restarts_per_P} Iterations, Mass Constrain = {mass_constrain}\n'
        f'{len(accept)}/{len(fitted_values)} Accepted, Cost $\\leq$ {accept_factor * best_cost}\n'
        f'{P_lower} $\\leq$ P $\\leq$ {P_upper}')
else:
    plt.title(f'{target} | {t_obs[0]} - {t_obs[-1]}\n'
                  f'mode = {fit_mode} | {len(fitted_values)} Orbit Fits, {n_restarts_per_P} Iterations, Mass Constrain = {mass_constrain}\n'
                  f'{len(accept)}/{len(fitted_values)} Accepted, Cost $\\leq$ {accept_factor * best_cost}\n'
                  f'{P_lower} $\\leq$ P $\\leq$ {P_upper}, {(1-m_total_frac_accept) *  m_total_guess:.3f} M$_\\odot$ $\\leq$ $M_{{total}}$ $\\leq$ {(1+m_total_frac_accept) *  m_total_guess:.3f} M$_\\odot$')
plt.scatter(orbital_periods, costs, c="tab:blue", s=30, alpha=0.7, marker="x",
            label="Modeled Orbits")
plt.scatter(orbital_periods_accepted, accept_costs, c="tab:red", s=30, alpha=0.7, marker="x",
            label="Accepted Orbits")

plt.axhline(accept_factor * best_cost, color="tab:red", ls="--", lw=1,
            label=f"Accept = {accept_factor}x Best Cost")
plt.axvline(best_accept_fit[0], color="tab:green", ls=":", lw=1,
            label=f"Best Period = {best_accept_fit[0]} yr")
ax = plt.gca()
ax.set_yscale('log')
if P_grid_log:
    ax.set_xscale('log')
plt.legend(fontsize=8)
plt.savefig(f"{folder_name}/fitted_periods_cost_{target}_{n_P_grid}_{n_restarts_per_P}_{mass_constrain}.png", dpi=200, bbox_inches='tight')
#print(f"Cost vs Period graph saved to Fitted Orbits/fitted_periods_cost_{target}_{n_P_grid}_{n_restarts_per_P}_{mass_constrain}.png")
with open(logname, "a") as f:
    f.write(f"Cost vs Period graph saved to fitted_periods_cost_{target}_{n_P_grid}_{n_restarts_per_P}_{mass_constrain}.png\n")

plt.close()
# -----------------------------------------------------------------



orbital_eccents = fitted_values[:, 2].tolist()
orbital_eccents_accepted = accept[:,2].tolist()

# Plot cost vs eccentricity
# Plot cost vs Period: the key degeneracy diagnostic.
plt.figure(figsize=(10, 6))
plt.xlabel("Eccentricity")
plt.ylabel("Cost")
if mass_constrain == False:
        plt.title(f'{target} | {t_obs[0]} - {t_obs[-1]}\n'
              f'mode = {fit_mode} | {len(fitted_values)} Orbit Fits, {n_restarts_per_P} Iterations, Mass Constrain = {mass_constrain}\n'
              f'{len(accept)}/{len(fitted_values)} Accepted, Cost $\\leq$ {accept_factor * best_cost}\n'
              f'{P_lower} $\\leq$ P $\\leq$ {P_upper}')
else:
        plt.title(f'{target} | {t_obs[0]} - {t_obs[-1]}\n'
                      f'mode = {fit_mode} | {len(fitted_values)} Orbit Fits, {n_restarts_per_P} Iterations, Mass Constrain = {mass_constrain}\n'
                      f'{len(accept)}/{len(fitted_values)} Accepted, Cost $\\leq$ {accept_factor * best_cost}\n'
                      f'{P_lower} $\\leq$ P $\\leq$ {P_upper}, {(1-m_total_frac_accept) *  m_total_guess:.3f} M$_\\odot$ $\\leq$ $M_{{total}}$ $\\leq$ {(1+m_total_frac_accept) *  m_total_guess:.3f} M$_\\odot$')
plt.scatter(orbital_eccents, costs, c="tab:blue", s=30, alpha=0.7, marker="x",
            label="Modeled Orbits")
plt.scatter(orbital_eccents_accepted, accept_costs, c="tab:red", s=30, alpha=0.7, marker="x",
            label="Accepted Orbits")

plt.axhline(accept_factor * best_cost, color="tab:red", ls="--", lw=1,
            label=f"Accept = {accept_factor}x Best Cost")
plt.axvline(best_accept_fit[2], color="tab:green", ls=":", lw=1,
            label=f"Best Eccentricity = {best_accept_fit[2]}")

ax = plt.gca()
ax.set_yscale('log')
if P_grid_log:
    ax.set_xscale('log')
plt.legend(fontsize=8)
plt.savefig(f"{folder_name}/fitted_eccents_cost_{target}_{n_P_grid}_{n_restarts_per_P}_{mass_constrain}.png", dpi=200, bbox_inches='tight')
# print(f"Cost vs Eccentricity graph saved to Fitted Orbits/fitted_eccents_cost_{target}_{n_P_grid}_{n_restarts_per_P}_{mass_constrain}.png")
with open(logname, "a") as f:
    f.write(f"Cost vs Eccentricity graph saved to fitted_eccents_cost_{target}_{n_P_grid}_{n_restarts_per_P}_{mass_constrain}.png\n")

plt.close()


if len(accept) != 0:
    # Plot best orbit from accept list
    P, T, e, a, i, Omega, omega, M_total, cost, r_squared = best_accept_fit
    plot_orbits(P, T, e, a, i, Omega, omega, M_total, cost, r_squared, f'_best_{target}_{n_P_grid}_{n_restarts_per_P}_{mass_constrain}')


with open(logname, "a") as f:
    f.write(f"All files saved to {folder_name}\n")


end_time = time.perf_counter()
total_time = end_time - start_time
minutes, seconds = divmod(total_time, 60)

with open(logname, "a") as f:
    f.write(f'Runtime: {minutes} min, {seconds:.2f} sec')

print(f'Modelling Complete')