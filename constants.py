import numpy as np
# from operator import itemgetter
# from numba import njit

# Misc Earth parameters
solar_constant = np.array(1.361)
m_earth_kg = np.array(5.9723e24)
g0_ms2 = np.array(9.80665)
per_earth_day = np.array(365.242198781)  # tropical year
j2_earth = np.array(1.0826269e-3)
j3_earth = np.array(-2.5323e-6)
j4_earth = np.array(-1.6204e-6)
f_jd1950 = np.array(358.203475)

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
r_moon_km = np.array(1737.4)
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
a_moon_km = np.array(384399)
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
lp_mercury_rad = np.array(77.45645) * deg_to_rad
lp_venus_rad = np.array(131.53298) * deg_to_rad
lp_earth_rad = np.array(102.97419) * deg_to_rad
lp_mars_rad = np.array(336.04084) * deg_to_rad
lp_jupiter_rad = np.array(14.75385) * deg_to_rad
lp_saturn_rad = np.array(92.43194) * deg_to_rad
lp_uranus_rad = np.array(170.96424) * deg_to_rad
lp_neptune_rad = np.array(44.97135) * deg_to_rad
lp_pluto_rad = np.array(224.06676) * deg_to_rad

# Argument of periapsis (omega)
w_mercury_rad = lp_mercury_rad - lan_mercury_rad
w_venus_rad = lp_venus_rad - lan_venus_rad
w_earth_rad = lp_earth_rad - lan_earth_rad
w_mars_rad = lp_mars_rad - lan_mars_rad
w_jupiter_rad = lp_jupiter_rad - lan_jupiter_rad
w_saturn_rad = lp_saturn_rad - lan_saturn_rad
w_uranus_rad = lp_uranus_rad - lan_uranus_rad
w_neptune_rad = lp_neptune_rad - lan_neptune_rad
w_pluto_rad = lp_pluto_rad - lan_pluto_rad

# Periapsis and apoapsis radius and velocity
rp_earth_km = np.array(147.09e6)
ra_earth_km = np.array(152.10e6)
vc_earth_kms = np.array(29.78)
vp_earth_kms = np.array(30.29)
va_earth_kms = np.array(29.29)

# Other
reference_date_jd1950 = np.array(2433282.423357)
reference_date_jd2000 = np.array(2451545.0)
day_to_jc = 1. / 36525  # day to Julian century

# Mean anomaly coefficients
_semimajor_axis_km = {'mercury': lambda c: 57.9091e6 * c / c,
                      'venus': lambda c: 108.2089e6 * c / c,
                      'earth': lambda c: 149.597927e6 * c / c,
                      'mars': lambda c: 227.9410e6 * c / c,
                      'jupiter': lambda c: 778.3284e6 * c / c,
                      'saturn': lambda c: 1426.9908e6 * c / c,
                      'uranus': lambda c: (2869.5862 - 0.0853 * c) * 1e6,
                      'neptune': lambda c: (4496.5623 + 0.1810 * c) * 1e6,
                      'pluto': lambda c: 5890.2138e6 * c / c
                      }

_eccentricity = {'mercury': lambda c: 0.2056244325 + 0.00002043 * c - 0.000000030 * c * c,
                 'venus': lambda c: 0.00679684275 - 0.000047649 * c + 0.000000091 * c * c,
                 'earth': lambda c: 0.0167301085 - 0.000041926 * c - 0.000000126 * c * c,
                 'mars': lambda c: 0.09335891275 + 0.000091987 * c - 0.000000077 * c * c,
                 'jupiter': lambda c: 0.04841911 + 0.00016302 * c,
                 'saturn': lambda c: 0.055716475 - 0.00034705 * c,
                 'uranus': lambda c: 288.465359 + 0.0117258558 * c,
                 'neptune': lambda c: 150.769275 + 0.0059952644 * c,
                 'pluto': lambda c: 301.687570 + 0.0039892964 * c
                 }

_inclination_rad = {'mercury': lambda c: (7.00381 - 0.00597 * c + 0.000001 * c * c) * deg_to_rad,
                    'venus': lambda c: (3.39413 - 0.00086 * c - 0.00003 * c * c) * deg_to_rad,
                    'earth': lambda c: (0.013076 * c - 0.000009 * c * c) * deg_to_rad,
                    'mars': lambda c: (1.85000 - 0.00821 * c - 0.00002 * c * c) * deg_to_rad,
                    'jupiter': lambda c: (1.30592 - 0.00205 * c + 0.00003 * c * c) * deg_to_rad,
                    'saturn': lambda c: (2.49036 + 0.00186 * c - 0.00003 * c * c) * deg_to_rad,
                    'uranus': lambda c: (0.77300 - 0.00186 * c - 0.00004 * c * c) * deg_to_rad,
                    'neptune': lambda c: (1.77467 + 0.00037 * c + 0.00001 * c * c) * deg_to_rad,
                    'pluto': lambda c: 17.16987 * deg_to_rad * c / c
                    }

_long_asc_node_rad = {'mercury': lambda c: (47.73859 - 0.12559 * c - 0.00009 * c * c) * deg_to_rad,
                      'venus': lambda c: (76.22967 - 0.27785 * c - 0.00014 * c * c) * deg_to_rad,
                      'earth': lambda c: (174.40956 - 0.24166 * c + 0.00006 * c * c) * deg_to_rad,
                      'mars': lambda c: (49.17193 - 0.29470 * c - 0.00065 * c * c) * deg_to_rad,
                      'jupiter': lambda c: (99.94335 - 0.16728 * c + 0.00055 * c * c) * deg_to_rad,
                      'saturn': lambda c: (113.22015 - 0.25973 * c + 0.00002 * c * c) * deg_to_rad,
                      'uranus': lambda c: (73.74521 + 0.06671 * c - 0.00068 * c * c) * deg_to_rad,
                      'neptune': lambda c: (131.22959 - 0.00574 * c - 0.00029 * c * c) * deg_to_rad,
                      'pluto': lambda c: 109.68346 * deg_to_rad * c / c
                      }

_arg_of_peri_rad = {'mercury': lambda c: (28.93892 + 0.28439 * c + 0.00007 * c * c) * deg_to_rad,
                    'venus': lambda c: (54.63793 + 0.28818 * c - 0.00115 * c * c) * deg_to_rad,
                    'earth': lambda c: (287.67097 + 0.56494 * c + 0.00009 * c * c) * deg_to_rad,
                    'mars': lambda c: (285.96668 + 0.73907 * c + 0.00047 * c * c) * deg_to_rad,
                    'jupiter': lambda c: (273.57374 + 0.04756 * c - 0.00086 * c * c) * deg_to_rad,
                    'saturn': lambda c: (338.84837 + 0.82257 * c - 0.00033 * c * c) * deg_to_rad,
                    'uranus': lambda c: (96.10329 + 0.16097 * c + 0.00037 * c * c) * deg_to_rad,
                    'neptune': lambda c: (272.95650 - 0.51258 * c - 0.00002 * c * c) * deg_to_rad,
                    'pluto': lambda c: 114.33841 * deg_to_rad * c / c
                    }

_mean_anomaly_rad = {'mercury': lambda c: (318.537027 + 4.0923344366 * c * 36525 + 2 / 3e6 * c * c) % 360 * deg_to_rad,
                     'venus': lambda c: (311.505478 + 1.602130189 * c * 36525 + 0.001286056 * c * c) % 360 * deg_to_rad,
                     'earth': lambda c: (358.000682 + 0.9856002628 * c * 36525 - 0.0001550000 * c * c
                                         - 0.0000033333 * c * c * c) % 360 * deg_to_rad,
                     'mars': lambda c: (169.458720 + 0.5240207716 * c * 36525 + 0.0001825972 * c * c
                                        + 0.0000011944 * c * c * c) % 360 * deg_to_rad,
                     'jupiter': lambda c: (302.650461 + 0.0830898769 * c * 36525) % 360 * deg_to_rad,
                     'saturn': lambda c: (66.251797 + 0.0334442397 * c * 36525) % 360 * deg_to_rad,
                     'uranus': lambda c: (288.465359 + 0.0117258558 * c * 36525) % 360 * deg_to_rad,
                     'neptune': lambda c: (150.769275 + 0.0059952644 * c * 36525) % 360 * deg_to_rad,
                     'pluto': lambda c: (301.687570 + 0.0039892964 * c * 36525) % 360 * deg_to_rad
                     }

_ephemeris = {'a': _semimajor_axis_km,
              'e': _eccentricity,
              'i': _inclination_rad,
              'O': _long_asc_node_rad,
              'w': _arg_of_peri_rad,
              'M': _mean_anomaly_rad}


def ephem(elems: list, planets: list, times: np.ndarray) -> np.ndarray:
    """
    Get orbital elements from planets at multiple times. The following example returns semimajor axis, eccentricity,
    argument of periapsis, and mean anomaly for both Earth and Mars at 26000 and 26200 JD (16 total outputs).
    Ex:
        elems = ['a', 'e', 'w', 'M']  # define orbital elements to retrieve
        planets = ['earth', 'mars']  # define planets for which elements are retrieved
        times = np.array([26000, 26200]) / 36525  # define times at which elements are retrieved (in Julian century)
        states = ephem(elems, planets, times)
    The output has shape (len(elems), len(planets), len(times)) - unless a dimension is one, since np.squeeze is called
    at the end.
    The options for elems are ['a', 'e', 'i', 'O', 'w', 'M']
    The options for planets are ['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto']
    Times is an array of one or more times in Julian century
    :param elems:
    :param planets:
    :param times:
    :return:
    """
    return np.squeeze(np.array([planet(times) for elem in list(map(_ephemeris.get, elems))
                                for planet in list(map(elem.get, planets))], float
                               ).reshape((len(elems), len(planets), len(times))))


r_soi_mercury = 46 * r_mercury_km
r_soi_venus = 102 * r_venus_km
r_soi_earth = 145 * r_earth_km
r_soi_moon = 38 * r_moon_km
r_soi_mars = 170 * r_mars_km
r_soi_jupiter = 687 * r_jupiter_km
r_soi_saturn = 1025 * r_saturn_km
r_soi_uranus = 2040 * r_uranus_km
r_soi_neptune = 3525 * r_neptune_km
