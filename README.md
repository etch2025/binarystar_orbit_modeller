Orbit Fitter for binary stars by iterating over the Thiele-Innes Method 
- Libraries
    - Matplotlib, NumPy, SciPy

Process
- Program generates a uniform vector of orbital periods from P_lower to P_upper with n_P_grid values
- Each period in the vector is fitted to the remaining 6 orbital elements through least squares with n_restarts_per_P random starting points. The lowest cost per iteration for that period is kept.
- Thiele-Innes method is used to obtain values to plot sky-projected orbit
- Each orbit fit at each fixed period is compiled to a list
- Orbits within accept_factor × best_cost and within a fraction of the estimated total mass (m_total_frac_accept) are retained.
- Orbital elements and total mass ranges are derived through the accepted list, inputted parallax measurement and Kepler's laws
- If no orbit fits can be accepted based on accept_factor and mass guesses
    - The lowest cost of all the fits is saved to orbit_fit_lowest_cost.png
- The lowest cost orbit fit in the list of accepted orbits is saved to
    - orbit_fit_best_{target}_{n_P_grid}_{n_restarts_per_P}_{mass_constrain}

Inputs
- CSV file with
    - Position angle (Theta): Degrees 
    - Angular Distance (RHO): Arcseconds (")
    - Decimal Observation Year
    - Parallax Angle (first data row only): Milliarcseconds (mas)
- Primary Star (m1) mass guess based on spectroscopic data (optional)
- Secondary Star (m1) mass guess based on spectroscopic data (optional)
    - Example mass guesses in mass_guess.txt

Assumptions:
- Both stars are approximately the same distance from Earth
- Binding energy < 0 (orbit is elliptical)


Non-linear least squares method using SciPy
Example binary star data from [Stellie Doppie](https://www.stelledoppie.it/) included
- Kruger 60 (DO Cep)
- 70 Ophiuchi
- 61 Cygni
- Sirius (Alp CMa)

Outputs
- Total System Mass
- Periastron Passage Year
- Distance based off parallax (parsecs)
- Orbital Elements
    - Semi-Major Axis (Angular and Physical): $a$
    - Eccentricity: $e$
    - Inclination: $i$
    - Longitude of Ascending Node: $\Omega$
    - Argument of Periastron: $\omega$
    - True Anomaly: $\nu$
    - Periastron Passage Year $T$
- Plots 
    - Best Fit Sky-Projected and True Orbit with component positions and elements
    - Orbital Eccentricity and Period vs Cost (Residuals)
    - Log File with range of accepted orbital elements and parameters

<img width="1300" height="700" alt="image" src="Example Files/orbit_fit_best_Sirius (HD 48915)_250_1000_True.png"/>

<img width="500" height="350" alt="image" src="Example Files/fitted_periods_cost_Sirius (HD 48915)_1000_10_True.png" />

<img width="500" height="350" alt="image" src="Example Files/fitted_eccents_cost_Sirius (HD 48915)_1000_10_True.png" />

- Future features
    - Predicted Position
    - Parabolic/Hyperbolic orbit fitting
    - Past and predicted time of apoapsis and periapsis
    - Roche Lobe location (for interferometic data)
