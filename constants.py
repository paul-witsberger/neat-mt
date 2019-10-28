import numpy as np

# Misc Earth parameters
solar_constant = np.array(1.361)
m_earth_kg = np.array(5.9723e24)
g0_ms2 = np.array(9.80665)
per_earth_day = np.array(365.256)
j2_earth = np.array(1.0826269e-3)
j3_earth = np.array(-2.5323e-6)
j4_earth = np.array(-1.6204e-6)

# Unit conversions
sec_to_min = np.array(1 / 60)
sec_to_hr = np.array(1 / 3600)
sec_to_day = np.array(1 / 86400)
sec_to_year = np.array(1 / 86400 / per_earth_day)
min_to_sec = np.array(60.)
hr_to_sec = np.array(3600.)
day_to_sec = np.array(86400.)
year_to_sec = np.array(86400 * per_earth_day)
m_to_km = np.array(1 / 1000)
km_to_m = np.array(1000.)
deg_to_rad = np.pi / 180
rad_to_deg = 180 / np.pi
au_to_km = np.array(149597870.7)

# Planetary mean radius
r_sun_km = np.array(695700.)
r_mercury_km = np.array(2439.7)
r_venus_km = np.array(6051.8)
r_earth_km = np.array(6371.0)
r_mars_km = np.array(3389.5)
r_jupiter_km = np.array(69911.)
r_saturn_km = np.array(58232.)
r_uranus_km = np.array(25362.)
r_neptune_km = np.array(24622.)
r_pluto_km = np.array(1188.)
r_ceres_km = np.array(965.)
r_eros_km = np.array(33.)
r_vesta_km = np.array(569.)
r_juno_km = np.array(234.)

# Gravitational parameter
mu_km3s2kg = np.array(6.674083131e-2)
u_sun_km3s2 = np.array(132712440018.9)
u_mercury_km3s2 = np.array(22032.9)
u_venus_km3s2 = np.array(324859.9)
u_earth_km3s2 = np.array(398600.4418)
u_moon_km3s2 = np.array(4904.86959)
u_mars_km3s2 = np.array(42328.372)
u_jupiter_km3s2 = np.array(126686534.9)
u_saturn_km3s2 = np.array(37931187.9)
u_uranus_km3s2 = np.array(5793939.9)
u_neptune_km3s2 = np.array(6836529.9)
u_pluto_km3s2 = np.array(871.9)
u_ceres_km3s2 = np.array(939300e15) * mu_km3s2kg
u_eros_km3s2 = np.array(6.69e15) * mu_km3s2kg
u_vesta_km3s2 = np.array(259000e15) * mu_km3s2kg
u_juno_km3s2 = np.array(20000e15) * mu_km3s2kg

# Semi-major axis
a_mercury_km = 0.38709893 * au_to_km
a_venus_km = 0.72333199 * au_to_km
a_earth_km = 1.00000011 * au_to_km
a_mars_km = 1.52366231 * au_to_km
a_jupiter_km = 5.20336301 * au_to_km
a_saturn_km = 9.53707032 * au_to_km
a_uranus_km = 19.19126393 * au_to_km
a_neptune_km = 30.06896348 * au_to_km
a_pluto_km = 39.48168677 * au_to_km
a_ceres_km = 2.768 * au_to_km
a_eros_km = 1.458 * au_to_km
a_vesta_km = 2.362 * au_to_km
a_juno_km = 2.670 * au_to_km

# Eccentricity
e_mercury = np.array(0.20563069)
e_venus = np.array(0.00677323)
e_earth = np.array(0.01671022)
e_mars = np.array(0.09341233)
e_jupiter = np.array(0.04839266)
e_saturn = np.array(0.05415060)
e_uranus = np.array(0.04716771)
e_neptune = np.array(0.00858587)
e_pluto = np.array(0.24880766)
e_ceres = np.array(0.0758)
e_eros = np.array(0.2227)
e_vesta = np.array(0.0889)
e_juno = np.array(0.2563)

# Inclination
i_mercury_rad = np.array(7.00487) * deg_to_rad
i_venus_rad = np.array(3.39471) * deg_to_rad
i_earth_rad = np.array(0.00005) * deg_to_rad
i_mars_rad = np.array(1.85061) * deg_to_rad
i_jupiter_rad = np.array(1.30530) * deg_to_rad
i_saturn_rad = np.array(2.48446) * deg_to_rad
i_uranus_rad = np.array(0.76986) * deg_to_rad
i_neptune_rad = np.array(1.76917) * deg_to_rad
i_pluto_rad = np.array(17.14175) * deg_to_rad
i_ceres_rad = np.array(10.59) * deg_to_rad
i_eros_rad = np.array(10.83) * deg_to_rad
i_vesta_rad = np.array(7.14) * deg_to_rad
i_juno_rad = np.array(12.99) * deg_to_rad

# Longitude of ascending node (Omega)
lan_mercury_rad = np.array(48.33167) * deg_to_rad
lan_venus_rad = np.array(76.68069) * deg_to_rad
lan_earth_rad = np.array(-11.26064) * deg_to_rad
lan_mars_rad = np.array(49.57854) * deg_to_rad
lan_jupiter_rad = np.array(100.55615) * deg_to_rad
lan_saturn_rad = np.array(113.71504) * deg_to_rad
lan_uranus_rad = np.array(74.22988) * deg_to_rad
lan_neptune_rad = np.array(131.72169) * deg_to_rad
lan_pluto_rad = np.array(110.30347) * deg_to_rad

# Longitude of perihelion (omega)
w_mercury_rad = np.array(77.45645) * deg_to_rad
w_venus_rad = np.array(131.53298) * deg_to_rad
w_earth_rad = np.array(102.97419) * deg_to_rad
w_mars_rad = np.array(336.04084) * deg_to_rad
w_jupiter_rad = np.array(14.75385) * deg_to_rad
w_saturn_rad = np.array(92.43194) * deg_to_rad
w_uranus_rad = np.array(170.96424) * deg_to_rad
w_neptune_rad = np.array(44.97135) * deg_to_rad
w_pluto_rad = np.array(224.06676) * deg_to_rad

# Periapsis and apoapsis radius and velocity
rp_earth_km = np.array(147.09e6)
ra_earth_km = np.array(152.10e6)
vc_earth_kms = np.array(29.78)
vp_earth_kms = np.array(30.29)
va_earth_kms = np.array(29.29)

# Other
reference_date_jd = np.array(2451545.0)
