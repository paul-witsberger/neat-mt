import numpy as np
from numpy import cos, sin
from math import gamma
from traj_config import gm
from numba import njit
from copy import copy
from constants import ephem, sec_to_day, year_to_sec, reference_date_jd1950, day_to_jc, u_earth_km3s2, r_earth_km
import constants as c
from typing import Union


@njit
def hohmann_circ(a1: float, a2: float, gm: float = gm) -> float:
    """
    Returns the delta V to transfer between two circular orbits using a Hohmann transfer
    :param a1:
    :param a2:
    :param gm:
    :return:
    """
    v1 = np.sqrt(gm / a1)
    v2 = np.sqrt(gm / a2)
    at = 0.5 * (a1 + a2)
    vpt = np.sqrt(2 * (-gm / 2 / at + gm / a1))
    vat = np.sqrt(2 * (-gm / 2 / at + gm / a2))
    dv1 = np.abs(vpt - v1)
    dv2 = np.abs(vat - v2)
    dv = dv1 + dv2
    return dv


@njit
def hohmann_rp_ra(rp1: float, vp1: float, ra2: float, va2: float, gm: float = gm) -> float:
    """
    Returns the delta V to transfer between two co-linear elliptical orbits using a Hohmann transfer
    :param rp1:
    :param vp1:
    :param ra2:
    :param va2:
    :param gm:
    :return:
    """
    at = 0.5 * (rp1 + ra2)
    vpt = np.sqrt(2 * (-gm / 2 / at + gm / rp1))
    vat = np.sqrt(2 * (-gm / 2 / at + gm / ra2))
    dv1 = np.abs(vpt - vp1)
    dv2 = np.abs(vat - va2)
    dv = dv1 + dv2
    return dv


@njit
def coe4_from_rv(r_vec: np.ndarray, v_vec: np.ndarray, gm: float = gm) -> np.ndarray:
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    # Get a from energy
    eps = v**2 / 2 - gm / r
    a = - gm / 2 / eps
    # Get p from ang mom
    h = np.linalg.norm(cross(r_vec, v_vec))
    p = h**2 / gm
    # Get e from a and p
    e = np.sqrt(1 - p / a)
    # Get sign of rdot
    rdot = np.dot(v_vec, r_vec / r)
    if e > 1e-5:
        # Get true anomaly from orbit equation
        f = np.arccos((p / r - 1) / e) * np.sign(rdot)
        # Get w from theta and f
        theta = np.arctan2(r_vec[1], r_vec[0])
        w = theta - f
    else:
        w = 0.0
        f = np.arctan2(r_vec[1], r_vec[0])
    coe = np.array([a, e, w, f])
    return coe


@njit
def period_from_inertial(state: np.ndarray, gm: float = gm, max_time_sec: float = 10 * year_to_sec) -> float:
    """
    Computes the period of an orbit given its 3D state vector.
    :param state:
    :param gm:
    :param max_time_sec:
    :return:
    """
    r_vec, v_vec = state[:3], state[3:]
    r, v = mag3(r_vec), mag3(v_vec)

    # Eccentricity
    e_vec = ((v * v - gm / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / gm
    e = mag3(e_vec)
    eps = v * v / 2 - gm / r

    # Semi-major axis
    tol = 1e-6
    if e < (1 - tol):
        a = - gm / 2 / eps
        per = 2 * np.pi * (a * a * a / gm) ** 0.5
    else:
        per = max_time_sec

    return per


@njit
def inertial_to_local(state: np.ndarray, has_extra: bool = True) -> np.ndarray:
    """
    Convert two 2D state vectors from inertial frame to spacecraft frame, and pass back extra values.
    :param state:
    :param has_extra:
    :return:
    """
    s1i = state[:4]
    s2i = state[4:8]
    if has_extra:
        extra = state[8:]
    r1, v1, th1, al1 = inertial_to_keplerian_2d(s1i)
    r2, v2, th2, al2 = inertial_to_keplerian_2d(s2i)
    if has_extra:
        return np.array([r1, r2, v1, v2, th1, th2, al1, al2, *extra])
    else:
        return np.array([r1, r2, v1, v2, th1, th2, al1, al2])


@njit
def inertial_to_keplerian(state: np.ndarray, gm: float = gm) -> np.ndarray:
    """
    Convert two 2D state vectors plus mass and time from inertial to Keplerian frame. Specific implementation
    for frame conversions for use by the neural network.
    :param state:
    :param gm:
    :return:
    """
    r1, r2, v1, v2, th1, th2, al1, al2, mr, tr = inertial_to_local(state)
    eps1 = v1 ** 2 / 2 - gm / r1
    eps2 = v2 ** 2 / 2 - gm / r2
    a1 = - gm / (2 * eps1)
    a2 = - gm / (2 * eps2)
    fpa1 = al1 - th1 - np.pi / 2
    fpa2 = al2 - th2 - np.pi / 2
    h1 = r1 * v1 * np.cos(fpa1)
    h2 = r2 * v2 * np.cos(fpa2)
    p1 = h1 ** 2 / gm
    p2 = h2 ** 2 / gm
    e1 = np.sqrt(1 - np.min((p1 / a1, 1.0)))
    e2 = np.sqrt(1 - np.min((p2 / a2, 1.0)))
    if e1 < 1e-6:
        arg1 = np.sign(p1 / r1 - 1)
    else:
        arg1 = (p1 / r1 - 1) / e1
        if np.abs(arg1) > 1:
            arg1 = np.sign(arg1)
    if e2 < 1e-6:
        arg2 = np.sign(p2 / r2 - 1)
    else:
        arg2 = (p2 / r2 - 1) / e2
        if np.abs(arg2) > 1:
            arg2 = np.sign(arg2)
    ta1 = np.arccos(arg1) * np.sign(fpa1+1e-8)
    ta2 = np.arccos(arg2) * np.sign(fpa2+1e-8)
    w1 = th1 - ta1
    w2 = th2 - ta2
    return np.array([a1, a2, e1, e2, w1, w2, ta1, ta2, mr, tr])


# @njit
def inertial_to_local_2d(state: np.ndarray) -> np.ndarray:
    """
    Convert a 2D state vector from inertial to (r, theta) frame.
    :param state:
    :return:
    """
    x, y, vx, vy = state
    r = (x * x + y * y) ** 0.5
    v = (vx * vx + vy * vy) ** 0.5
    th = np.arctan2(y, x)
    al = np.arctan2(vy, vx)
    return np.array([r, v, th, al])


@njit
def inertial_to_keplerian_2d(state: np.ndarray, gm: float = gm) -> np.ndarray:
    """
    Convert a 2D state vector from inertial to Keplerian
    :param state:
    :param gm:
    :return:
    """
    r, v, th, al = inertial_to_local_2d(state)
    eps = v ** 2 / 2 - gm / r
    a = - gm / (2 * eps)
    fpa = -fix_angle(al - th - np.pi / 2)
    # h = r * v * np.cos(fpa)
    p = r / gm * (v * v) * r * np.cos(fpa) ** 2
    e = np.sqrt(1 - min(p / a, 1.0))
    if e < 1e-6:
        arg = np.sign(p / r - 1)
        w = 0
        ta = th
    else:
        arg = (p / r - 1) / e
        if np.abs(arg) > 1:
            arg = np.sign(arg)
        ta = np.arccos(arg) * np.sign(fpa + 1e-8)
        w = fix_angle(th - ta)
    return np.array([a, e, w, ta])


@njit
def cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Compute the cross product of two vectors.
    :param left:
    :param right:
    :return:
    """
    x = ((left[1] * right[2]) - (left[2] * right[1]))
    y = ((left[2] * right[0]) - (left[0] * right[2]))
    z = ((left[0] * right[1]) - (left[1] * right[0]))
    return np.array([x, y, z])


# @njit
def inertial_to_keplerian_3d(state: np.ndarray, gm: float = gm) -> np.ndarray:
    """
    Convert a 3D state vector from inertial to Keplerian
    :param state:
    :param gm:
    :return:
    """
    r_vec, v_vec = state[:3], state[3:]
    tol = 1e-6
    r = mag3(r_vec)
    v = mag3(v_vec)
    h_vec = cross(r_vec, v_vec)
    h = mag3(h_vec)
    k_vec = np.array([0, 0, 1])
    n_vec = cross(k_vec, h_vec)
    n = mag3(n_vec)

    # Eccentricity
    e_vec =  ((v * v - gm / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / gm
    e = mag3(e_vec)
    eps = v * v / 2 - gm / r

    # Semi-major axis
    if abs(e - 1) < tol:
        a = np.infty
    else:
        a = - gm / 2 / eps

    # Inclination
    i = np.arccos(h_vec[2] / h)

    # Longitude of ascending node
    if (i < tol or np.abs(np.pi - i) < tol):
        # Special case - equatorial
        om = 0.
    else:
        # General
        om = np.arccos(n_vec[0] / n)
        if n_vec[1] < 0:
            om = 2 * np.pi - om

    # Argument of periapsis
    if n == 0 and False: # TODO verify where this statement came from
        # Special case - equatorial
        w = np.arccos(np.dot(np.array([1, 0, 0]), e_vec))
        # w = 0.
    elif (i < tol or np.abs(np.pi - i) < tol):
        # Special case: elliptical equatorial - true longitude of periapsis
        w = np.arccos(e_vec[0] / e)
        if e_vec[1] < 0:
            w = 2 * np.pi - w
    else:
        # General
        w = np.arccos(np.dot(n_vec, e_vec) / (n * e))
        if e_vec[2] < 0:
            w = 2 * np.pi - w

    # True anomaly
    if e < tol and (i < tol or np.abs(np.pi - i) < tol):
        # Special case: circular equatorial - use true longitude
        f = np.arccos(r_vec[0] / r)
        if r_vec[1] < 0:
            f = 2 * np.pi - f
    elif e < tol:
        # Special case: circular inclined - argument of latitude
        f = np.arccos(np.dot(n_vec, r_vec) / (n * r))
        if r_vec[2] < 0:
            f = 2 * np.pi - f
    else:
        # General
        f = np.arccos(max(min(np.dot(e_vec, r_vec) / (e * r), 1.), -1.))
        if np.dot(r_vec, v_vec) < 0:  # TODO verify this line - I changed this from "<0" to ">0" and then reverted
            f = 2 * np.pi - f

    return np.array([a, e, i, w, om, f])


def keplerian_to_perifocal_3d(state: np.ndarray, gm: float = gm) -> tuple:
    """
    Convert a 3D state vector from Keplerian to perifocal
    :param state:
    :param gm:
    :return:
    """
    a, e, i, w, om, f = state[0], state[1], state[2], state[3], state[4], state[5]
    p = a * (1 - e ** 2)
    r11 = p * cos(f) / (1 + e * cos(f))
    r12 = p * sin(f) / (1 + e * cos(f))
    r13 = np.zeros(int(np.array(state.shape).prod() / 6))
    v11 = -np.sqrt(gm / p) * sin(f)
    v12 = np.sqrt(gm / p) * (e + cos(f))
    r_p = np.vstack((r11, r12, r13))
    v_p = np.vstack((v11, v12, r13))
    return r_p, v_p


def keplerian_to_inertial_3d(state: np.ndarray, gm: float = gm, mean_or_true: str = 'true') -> np.ndarray:
    """
    Convert a 3D state vector from Keplerian to inertial
    :param state:
    :param gm:
    :return:
    """
    a, e, i, w, om, f = state
    if mean_or_true == 'mean':
        m = f
        if len(state.shape) > 1:
            state[5, :] = np.array([mean_to_true_anomaly(mm, ee) for mm, ee, in zip(m, e)]).T
        else:
            state[5] = mean_to_true_anomaly(m, e)
    r_p, v_p = keplerian_to_perifocal_3d(state, gm=gm)
    r_i = euler313(r_p, om, i, w)
    v_i = euler313(v_p, om, i, w)
    return np.hstack((r_i, v_i))


def keplerian_to_perifocal_2d(state: np.ndarray, gm: float = gm, mean_or_true: str = 'true') -> np.ndarray:
    """
    Convert a 2D state vector from Keplerian to perifocal
    :param state:
    :param gm:
    :return:
    """
    if mean_or_true == 'true':
        a, e, w, f = state
    else:
        a, e, w, m = state
        f = np.array([mean_to_true_anomaly(mm, e) for mm in m], float)
    p = a * (1 - e ** 2)
    r_p = np.hstack((np.array(p * np.cos(f) / (1 + e * np.cos(f))), np.array(p * np.sin(f) / (1 + e * np.cos(f)))))
    v_p = np.hstack((np.array(-np.sqrt(gm / p) * np.sin(f)), np.array(np.sqrt(gm / p) * (e + np.cos(f)))))
    return np.hstack((r_p, v_p))


def keplerian_to_inertial_2d(state: np.ndarray, gm: float = gm, mean_or_true: str = 'true') -> np.ndarray:
    """
    Convert a 2D state vector from Keplerian to inertial
    :param state:
    :param gm:
    :return:
    """
    # Convert to perifocal frame
    a, e, w, f = state
    state_peri = keplerian_to_perifocal_2d(state, gm=gm, mean_or_true=mean_or_true)
    r_p, v_p = state_peri[:2], state_peri[2:]
    # Construct DCM
    dcm = np.zeros((2, 2))
    dcm[0, 0] = np.cos(w)
    dcm[0, 1] = -np.sin(w)
    dcm[1, 0] = np.sin(w)
    dcm[1, 1] = np.cos(w)
    r_i = dcm.dot(r_p)
    v_i = dcm.dot(v_p)
    return np.hstack((r_i, v_i))


@njit
def keplerian_to_mee_3d(state: np.ndarray) -> np.ndarray:
    """
    Convert a 3D state vector from Keplerian to Modified Equinoctial Elements
    :param state:
    :return:
    """
    a, e, i, w, om, ta = state
    # Check that values are acceptable
    assert (a > 0 and e < 1) or (a < 0 and e > 1) or (e == 1), "semimajor axis and eccentricity do not agree"
    assert e >= 0., "eccentricity is below zero"
    assert i >= 0. and i <= np.pi, "inclination is outside of acceptable bounds"
    w = fix_angle(w, upper_bound=2*np.pi, lower_bound=0.)
    om = fix_angle(om, upper_bound=2*np.pi, lower_bound=0.)
    ta = fix_angle(ta, upper_bound=2*np.pi, lower_bound=0.)
    # Convert to MEE
    p = a * (1 - e ** 2)
    f = e * cos(w + om)
    g = e * sin(w + om)
    h = np.tan(i / 2) * cos(om)
    k = np.tan(i / 2) * sin(om)
    L = om + w + ta
    return np.array([p, f, g, h, k, L])


@njit
def mee_to_keplerian_3d(state: np.ndarray) -> np.ndarray:
    """
    Convert a 3D state vector from Modified Equinoctial Elements to Keplerian
    :param state:
    :return:
    """
    p, f, g, h, k, L = state
    # Convert to Keplerian
    om = np.arctan2(k, h)
    i = 2 * np.arctan(np.sqrt(h ** 2 + k ** 2))
    w = np.arctan2(g, f) - om
    e = np.sqrt(g ** 2 + f ** 2)
    a = p / (1 - g ** 2 - f ** 2)
    v = L - om - w
    # Make sure angles are properly scaled
    w = fix_angle(w, 2 * np.pi, 0.)
    om = fix_angle(w, 2 * np.pi, 0.)
    v = fix_angle(v, 2 * np.pi, 0.)
    return np.array([a, e, i, w, om, v])


# @njit
def rotate_vnc_to_inertial_3d(vec: np.ndarray, state: np.ndarray) -> np.ndarray:
    """
    Rotates the current velocity vector to an inertial frame.
    :param vec:
    :param state:
    :return:
    """
    r_vec, v_vec = state[:3], state[3:6]      # radius and velocity vectors
    v_hat = v_vec / mag3(v_vec)               # velocity unit vector
    h_vec = cross(r_vec, v_vec)               # angular momentum vector
    n_hat = h_vec / mag3(h_vec)               # angular momentum unit vector; also, normal unit vector
    c_hat = cross(v_hat, n_hat)               # co-normal unit vector
    dcm = np.vstack((v_hat, n_hat, c_hat)).T  # direction cosine matrix
    return np.dot(dcm, vec)


@njit
def fix_angle(angle: float, upper_bound: float = np.pi, lower_bound: float = -np.pi) -> float:
    """
    Forces an angle to be within a set 2*pi radian window.
    :param angle:
    :param upper_bound:
    :param lower_bound:
    :return:
    """
    # Check that bounds are properly defined
    assert upper_bound - lower_bound == 2 * np.pi
    assert not np.isnan(angle)
    while True:
        angle += 2 * np.pi if angle < lower_bound else 0.  # add 2pi if too negative
        angle -= 2 * np.pi if angle > upper_bound else 0.  # subtract 2pi if too positive
        if angle <= upper_bound and angle >= lower_bound:
            return angle


def min_energy_lambert(r0: np.ndarray, r1: np.ndarray, gm: float = gm) -> (np.ndarray, np.ndarray, float):
    """
    Computes the minimum-energy transfer between two points.
    :param r0:
    :param r1:
    :param gm:
    :return:
    """
    assert r0[2] == 0 and r1[2] == 0
    # Get magnitude of position vectors
    r0mag = np.linalg.norm(r0)
    r1mag = np.linalg.norm(r1)

    # Get properties of transfer angle
    cos_dnu = np.dot(r0, r1) / r0mag / r1mag
    sin_dnu = np.linalg.norm(cross(r0, r1)) / r0mag / r1mag
    dnu = np.arctan2(sin_dnu, cos_dnu)
    if dnu < 0:
        print('dnu is negative')

    # Calculate p of minimum energy transfer
    pmin = r0mag * r1mag / np.sqrt(r0mag ** 2 + r1mag ** 2 - 2 * np.dot(r0, r1)) * (1 - cos_dnu)

    # Calculate necessary velocity at initial position
    v0min = np.sqrt(gm * pmin) / r0mag / r1mag / sin_dnu * (r1 - r0 * (1 - r1mag / pmin * (1 - cos_dnu)))
    v0minmag = np.linalg.norm(v0min)

    # Check direction of transfer - make sure angular momentum is positive
    hmin = cross(r0, v0min)
    hminmag = np.linalg.norm(hmin)

    if hmin[-1] < 0:
        v0min = -v0min

    # Get other properties of transfer
    amin = -(gm / 2) / (v0minmag ** 2 / 2 - gm / r0mag)
    emin = np.sqrt(1 - pmin / amin)
    n = np.sqrt(gm / amin ** 3)

    # Calculate true anomaly of initial point - numerical safety check for cosine, and ascending/descending check
    sign0 = np.sign(np.dot(r0, v0min) / r0mag / v0minmag)
    arg0 = (pmin / r0mag - 1) / emin
    arg0 = min(max(-1., arg0), 1.)
    nu0 = np.arccos(arg0) * sign0
    # Calculate true anomaly of final point - rotate nu in a positive sense
    if hmin[-1] > 0:
        nu1 = nu0 + dnu
    else:
        nu1 = nu0 - dnu + 2 * np.pi

    # Calculate velocity at second position
    v1minmag = np.sqrt(max(2 * gm / r1mag - gm / amin, 0))
    arg = min(max(hminmag / r1mag / v1minmag, -1), 1)
    fpa = np.arccos(arg) * int(nu1 < np.pi)
    dcm = np.array([[np.cos(np.pi / 2 - fpa), -np.sin(np.pi / 2 - fpa)], [np.sin(np.pi / 2 - fpa), np.cos(np.pi / 2 - fpa)]])
    v1minunit = np.hstack((np.matmul(dcm, r1[:2] / r1mag), 0))
    v1min = v1minunit * v1minmag

    # Calculate eccentric anomaly using true anomaly
    E0 = 2 * np.arctan(np.sqrt((1 - emin) / (1 + emin)) * np.tan(nu0 / 2))
    E1 = 2 * np.arctan(np.sqrt((1 - emin) / (1 + emin)) * np.tan(nu1 / 2))
    E0 = fix_angle(E0, 2 * np.pi, 0)
    E1 = fix_angle(E1, 2 * np.pi, 0)

    # Calculate time of flight
    t0 = (E0 - emin * np.sin(E0)) / n
    t1 = (E1 - emin * np.sin(E1)) / n

    # Calculate time of flight - add a revolution if tof < 0
    tof = t1 - t0
    if hmin[-1] < 0:
        per = 2 * np.pi / n
        tof += per

    return v0min, v1min, tof


@njit
def c2(psi: float) -> float:
    """
    Helper function for vallado().
    :param psi:
    :return:
    """
    eps = 1.0
    if psi > eps:
        res = (1 - np.cos(np.sqrt(psi))) / psi
    elif psi < -eps:
        res = (np.cosh(np.sqrt(-psi)) - 1) / (-psi)
    else:
        res = 1.0 / 2.0
        delta = (-psi) / gamma(2 + 2 + 1)
        k = 1
        while res + delta != res:
            res = res + delta
            k += 1
            delta = (-psi) ** k / gamma(2 * k + 2 + 1)

    return res


@njit
def c3(psi: float) -> float:
    """
    Helper function for vallado().
    :param psi:
    :return:
    """
    eps = 1.0
    if psi > eps:
        res = (np.sqrt(psi) - np.sin(np.sqrt(psi))) / (psi * np.sqrt(psi))
    elif psi < -eps:
        res = (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi)) / (-psi * np.sqrt(-psi))
    else:
        res = 1.0 / 6.0
        delta = (-psi) / gamma(2 + 3 + 1)
        k = 1
        while res + delta != res:
            res = res + delta
            k += 1
            delta = (-psi) ** k / gamma(2 * k + 3 + 1)
    return res


# @njit
def vallado(k: float, r0: np.ndarray, r: np.ndarray, tof: float, short: bool, numiter: int, rtol: float) \
        -> (np.ndarray, np.ndarray):
    """
    Computes a Lambert solution based on the implementation in Vallado.
    :param k:
    :param r0:
    :param r:
    :param tof:
    :param short:
    :param numiter:
    :param rtol:
    :return:
    """
    if short:
        t_m = +1
    else:
        t_m = -1

    norm_r0 = np.dot(r0, r0) ** 0.5
    norm_r = np.dot(r, r) ** 0.5
    norm_r0_times_norm_r = norm_r0 * norm_r
    norm_r0_plus_norm_r = norm_r0 + norm_r

    cos_dnu = np.dot(r0, r) / norm_r0_times_norm_r

    A = t_m * (norm_r * norm_r0 * (1 + cos_dnu)) ** 0.5

    if A == 0.0:
        raise RuntimeError("Cannot compute orbit, phase angle is 180 degrees")

    psi = 0.0
    psi_low = -4 * np.pi
    psi_up = 4 * np.pi ** 2

    count = 0

    while count < numiter:
        y = norm_r0_plus_norm_r + A * (psi * c3(psi) - 1) / c2(psi) ** 0.5
        if A > 0.0:
            # Readjust xi_low until y > 0.0
            # Translated directly from Vallado
            while y < 0.0:
                psi_low = psi
                psi = (
                    0.8
                    * (1.0 / c3(psi))
                    * (1.0 - norm_r0_times_norm_r * np.sqrt(c2(psi)) / A)
                )
                y = norm_r0_plus_norm_r + A * (psi * c3(psi) - 1) / c2(psi) ** 0.5

        xi = np.sqrt(y / c2(psi))
        tof_new = (xi ** 3 * c3(psi) + A * np.sqrt(y)) / np.sqrt(k)

        # Convergence check
        if np.abs((tof_new - tof) / tof) < rtol:
            break
        count += 1
        # Bisection check
        condition = tof_new <= tof
        psi_low = psi_low + (psi - psi_low) * condition
        psi_up = psi_up + (psi - psi_up) * (not condition)

        psi = (psi_up + psi_low) / 2
    else:
        raise RuntimeError("Maximum number of iterations reached")

    f = 1 - y / norm_r0
    g = A * np.sqrt(y / k)

    gdot = 1 - y / norm_r

    v0 = (r - f * r0) / g
    v = (gdot * r - r0) / g

    return v0, v


def lambert_min_dv(k: float, state: np.ndarray, state_f: np.ndarray, short: bool = True, do_print: bool = False) \
        -> (np.ndarray, np.ndarray, float):
    """
    Computes the minimum delta V transfer between two states assuming a two-impulse maneuver.
    :param k:
    :param r0:
    :param v0:
    :param rf:
    :param vf:
    :param short:
    :param do_print:
    :return:
    """
    r0, v0 = state[:3], state[3:6]
    rf, vf = state_f[:3], state_f[3:6]
    # Define parameters for search
    count = 0
    max_count = 20
    min_dv = 1e10
    rtol = 1e-8
    numiter = 50
    r0_mag = np.linalg.norm(r0)
    high = np.pi * np.sqrt(r0_mag ** 3 / k)
    low = high * 0.1

    # Compute initial guesses
    tof = [low, (high + low) / 3, (high + low) * 2 / 3, high]
    dv = list()
    for _tof in tof:
        try:
            sol = vallado(k, r0, rf, _tof, short=short, numiter=numiter, rtol=rtol)
        except RuntimeError:
            sol = ([10, 10, 10], [10, 10, 10])
        dv.append(np.linalg.norm((sol[0] - v0)) + np.linalg.norm((sol[1] - vf)))

    # Adjust bounds and start direction of search
    dv = np.array(dv)
    if dv[0] > dv[1] and dv[1] > dv[2]:
        if dv[2] > dv[3]:
            low = tof[2]
        else:
            low = tof[1]
    elif dv[1] < dv[2] and dv[2] < dv[3]:
        if dv[0] < dv[1]:
            high = tof[1]
        else:
            high = tof[2]
    else:
        high = tof[2]
        low = tof[1]
    best_tof, last_best = tof[np.argsort(dv)[0]], tof[np.argsort(dv)[1]]
    last_tof = best_tof
    best_dv = np.min(dv)

    # Main loop
    while True:
        tof = (high + low) / 2
        if do_print:
            print('TOF = %.3f sec' % tof)
        try:
            dv = vallado(k, r0, rf, tof, short=short, numiter=numiter, rtol=rtol)
        except RuntimeError:
            dv = ([10, 10, 10], [10, 10, 10])
        cost = np.linalg.norm((dv[0] - v0)) + np.linalg.norm((dv[1] - vf))
        if do_print:
            print('Cost = %.3f km/s' % cost)
            dif = cost - min_dv
            print('Diff = %.3e km/s' % dif)

        if cost <= min_dv:
            min_dv = cost
            best_dv = copy(dv)
            last_best = best_tof
            best_tof = tof
        if count >= max_count:
            break

        dir_last_best = np.sign(tof - last_best)
        dir_best = np.sign(tof - best_tof)
        dir_last = np.sign(tof - last_tof)

        if count > 0:
            if dir_best == 0:
                if dir_last_best > 0:
                    low = last_best
                elif dir_last_best < 0:
                    high = last_best
                else:
                    high *= 1.1
            elif dir_last > 0 and dir_best > 0:
                high = tof
            elif dir_last < 0 and dir_best < 0:
                low = tof
            elif dir_last > 0 and dir_best < 0:
                low = last_tof
            elif dir_last < 0 and dir_best > 0:
                high = last_tof
            elif dir_last == 0 and dir_best > 0:
                high = tof
            elif dir_last == 0 and dir_best < 0:
                low = tof
            else:
                print('Reached else.')
                print('dir_last = ' % dir_last)
                print('dir_last_best = ' % dir_last_best)

        last_tof = tof
        count += 1
        if do_print:
            print('[{0:0.4e}, {1:0.4e}]'.format(low, high))
            print()

    dv1 = best_dv[0] - v0
    dv2 = vf - best_dv[1]
    return dv1, dv2, best_tof


def gamma_from_r_v(r_vec: np.ndarray, v_vec: np.ndarray) -> float:
    """
    Computes the current flight path angle from radius and velocity vectors.
    :param r_vec:
    :param v_vec:
    :return:
    """
    r_vec = np.append(r_vec, 0) if r_vec.size == 2 else r_vec
    r_mag = mag3(r_vec)
    r_hat = r_vec / r_mag

    v_vec = np.append(v_vec, 0) if v_vec.size == 2 else v_vec
    v_mag = mag3(v_vec)
    v_hat = v_vec / v_mag

    h_vec = cross(r_vec, v_vec)
    h_mag = mag3(h_vec)
    assert h_mag > 0
    h_hat = h_vec / h_mag

    t_hat = cross(h_hat, r_hat)
    gamma_mag = np.arccos(np.dot(t_hat, v_hat))
    gamma_sign = np.sign(np.dot(r_vec, v_vec))
    gamma = gamma_mag * gamma_sign
    return gamma


@njit
def mag2(array: np.ndarray) -> float:
    """
    Computes the magnitude of a 2-dimensional array.
    :param array:
    :return:
    """
    return (array[0] * array[0] + array[1] * array[1]) ** 0.5


@njit
def mag3(array: np.ndarray) -> float:
    """
    Computes the magnitude of a 3-dimensional array.
    :param array:
    :return:
    """
    return (array[0] * array[0] + array[1] * array[1] + array[2] * array[2]) ** 0.5


def true_anomaly_from_r_v(r_vec: np.ndarray, v_vec: np.ndarray) -> float:
    # TODO streamline this
    if r_vec.size == 2:
        a, e, w, f = inertial_to_keplerian_2d(np.hstack((r_vec, v_vec)))
    else:
        try:
            a, e, i, w, om, f = inertial_to_keplerian_3d(np.hstack((r_vec, v_vec)))
        except ZeroDivisionError as err:
            print(r_vec)
            print(v_vec)
            raise err
    return f


def time_to_periapsis_from_r_v(r_vec: np.ndarray, v_vec: np.ndarray, gm: float = gm) -> float:
    """
    Computes the time until the current state reaches periapsis. Assumes 2D.
    :param r_vec:
    :param v_vec:
    :param gm:
    :param r_periapsis_km:
    :return:
    """
    # TODO streamline this
    if r_vec.size == 2:
        a, e, w, f = inertial_to_keplerian_2d(np.hstack((r_vec, v_vec)), gm=gm)
    else:
        a, e, i, w, om, f = inertial_to_keplerian_3d(np.hstack((r_vec, v_vec)), gm=gm)

    if e >= 0.99 and e <= 1.01:  # parabolic
        if f > 0 and f < np.pi:  # ascending and escaping
            print(f)
            print(e)
            raise RuntimeError('Spacecraft is parabolic, past periapsis, and escaping.')
        else:
            B = np.tan(f / 2)
            h = mag3(cross(r_vec, v_vec))
            p = h ** 2 / gm
            n = 2 * (gm / p ** 3) ** 0.5
            time_to_periapsis = - (B ** 3 / 3 + B) / n
    elif e >= 1.01:  # hyperbolic
        if f > 0 and f < np.pi:  # ascending and escaping
            print(f)
            print(e)
            raise RuntimeError('Spacecraft is hyperbolic, past periapsis, and escaping.')
        else:
            H = np.arctanh(np.sin(f) * (e * e - 1) ** 0.5 / (e + np.cos(f)))
            M = e * np.sinh(H) - H
            time_to_periapsis = (2 * np.pi - M) * np.sqrt(gm / - a ** 3)
    else:  # elliptical
        # Calculate eccentric anomaly
        E = np.arctan2(np.sin(f) * (1 - e * e) ** 0.5, e + np.cos(f))
        E = fix_angle(E, 2 * np.pi, 0)
        M = E - e * np.sin(E)
        time_to_periapsis = (2 * np.pi - M) * np.sqrt(a ** 3 / gm)
    return time_to_periapsis


def min_dv_capture(state_sc: np.ndarray, state_target: np.ndarray, gm: float, r_periapsis_km: float) \
        -> (np.ndarray, np.ndarray, float):
    """
    Computes the delta V required to capture into a desired circular orbit. The assumed inputs are radius and velocity
    vectors of the spacecraft with respect to the sun, and the radius and velocity vectors of target planet with respect
    to the sun. Also, the gravitational parameter of the target planet and the desired final circular orbital radius.
    Outputs the two delta V vectors, and the time of flight between them.
    :param state_sc:
    :param state_target:
    :param gm:
    :param r_periapsis_km:
    :return:
    """
    # Get relative vectors
    r_sc_vec, v_sc_vec = state_sc[:3], state_sc[3:6]
    r_target_vec, v_target_vec = state_target[:3], state_target[3:6]
    r_vec = r_sc_vec - r_target_vec
    v_vec = v_sc_vec - v_target_vec
    # Compute current energy
    r_mag_km = mag2(r_vec) if r_vec.size == 2 else mag3(r_vec)
    v_mag_kms = mag2(v_vec) if v_vec.size == 2 else mag3(v_vec)
    v_hat = v_vec / v_mag_kms
    epsilon_current = v_mag_kms * v_mag_kms / 2 - gm / r_mag_km  # with respect to target
    # Define target orbit
    v_final_kms = (gm / r_periapsis_km) ** 0.5  # target circular speed
    # Check if current state is hyperbolic
    if epsilon_current > 0:
        v_periapsis_kms = (2 * gm / r_periapsis_km) ** 0.5  # parabolic speed at target periapsis
        v_mag_capture_kms = (2 * gm / r_mag_km) ** 0.5      # speed along parabolic orbit at current distance (?)
        dv1_mag = (v_mag_capture_kms - v_mag_kms) * 1.0     # final - initial = parabolic capture (! TODO !)
        gamma = gamma_from_r_v(r_vec, v_vec)                # check if ascending or descending
        if gamma > 0:
            dv1_mag -= v_mag_capture_kms  # (! TODO !) don't need to completely reverse direction - this will send you around the other side of the planet
            # which side will this capture around?
    else:
        dv1_mag = 0
        v_periapsis_kms = (2 * (epsilon_current + gm / r_periapsis_km)) ** 0.5
    # Compute velocity vector after the maneuvers
    dv1_vec = v_hat * dv1_mag
    v_transfer_vec = v_vec + dv1_vec
    # Compute delta v to get into target orbit at periapsis
    dv2_mag = v_final_kms - v_periapsis_kms
    # Compute delta v vector and final velocity vector
    true_anomaly = true_anomaly_from_r_v(r_vec, v_transfer_vec)
    a, e, i, w, om , f = inertial_to_keplerian_3d(state_sc[:6] - state_target, gm=gm)
    v_final_vec = euler313(np.array([v_final_kms, 0, 0]), om, i, w + true_anomaly + np.pi / 2)
    dv2_vec = v_final_vec / v_final_kms * dv2_mag
    # Compute time-of-flight to reach periapsis after first maneuver
    tof = time_to_periapsis_from_r_v(r_vec, v_transfer_vec, gm)
    return dv1_vec, dv2_vec, tof


def rotate_vector_2d(vec: np.ndarray, angle: float) -> np.ndarray:
    """
    Assumes a 2D vector or a 3D vector with zero Z component. Rotates vector 'vec' in-plane by angle 'angle' ccw.
    :param vec:
    :param angle:
    :return:
    """
    c, s = np.cos(angle), np.sin(angle)
    if vec.size == 2:
        dcm = np.array([[c, s], [-s, c]])
    else:
        dcm = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    return np.matmul(dcm, vec)


@njit
def shift_vector_origin_single(cb1_to_sc: np.ndarray, cb2_to_cb1: np.ndarray) -> np.ndarray:
    """
    Change the central body by which a vector is defined.
    :param cb1_to_sc:
    :param cb2_to_cb1:
    :return:
    """
    cb2_to_sc = np.empty_like(cb1_to_sc, float)
    for i in range(cb1_to_sc.shape[0]):
        cb2_to_sc[i, :] = cb2_to_cb1[i, :] + cb1_to_sc
    return cb2_to_sc


def shift_vector_origin_multiple(cb1_to_sc: np.ndarray, cb2_to_cb1: np.ndarray) -> np.ndarray:
    """
    Change the central body by which a vector is defined. Assumes cb2_to_cb1 has same shape as cb1_to_sc.
    :param cb1_to_sc:
    :param cb2_to_cb1:
    :return:
    """
    return cb2_to_cb1 + cb1_to_sc


def change_central_body(states: np.ndarray, times: np.ndarray, cur_cb: str, new_cb: str, gm: float = gm) -> np.ndarray:
    """
    Shifts an array of states to be defined by a different reference over a time history.
    :param states:
    :param times:
    :param cur_cb:
    :param new_cb:
    :return:
    """
    # Determine state history of new central body with respect to current central body
    n_dim = states.shape[1]
    assert n_dim == 4 or n_dim == 6
    elements = ['a', 'e', 'w', 'M'] if n_dim == 2 else ['a', 'e', 'i', 'w', 'O', 'M']
    assert cur_cb == 'sun' or new_cb == 'sun'
    if cur_cb == 'sun':
        planets = [new_cb]
        flip = False
    else:
        planets = [cur_cb]
        flip = True
    times_jc = (times * sec_to_day + reference_date_jd1950) * day_to_jc
    non_sun_cb_states = ephem(elements, planets, times_jc)
    # Convert to inertial coordinates
    if n_dim == 4:
        cb2_to_cb1_i = keplerian_to_inertial_2d(non_sun_cb_states, gm=gm, mean_or_true='mean')
    else:
        cb2_to_cb1_i = keplerian_to_inertial_3d(non_sun_cb_states, gm=gm, mean_or_true='mean')
    # Flip relative direction if the new body is 'sun' # TODO but this was meant for cartesian -> how does this work for keplerian?
    if flip:
        cb2_to_cb1_i *= -1.
    # Shift vectors
    new_states = shift_vector_origin_multiple(states, cb2_to_cb1_i)
    return new_states


@njit
def mean_to_true_anomaly(m: float, e: float, tol: float = 1e-8) -> float:
    # Assume small eccentricity
    ea_guess = m
    # ea_next = ea_guess
    for i in range(20):
        ea_next = ea_guess + (m + e * sin(ea_guess) - ea_guess) / (1 - e * cos(ea_guess))
        diff = ea_next - ea_guess
        if abs(diff) < tol:
            ea = ea_next
            f = 2 * np.arctan(((1 + e) / (1 - e)) ** 0.5 * np.tan(ea / 2))
            f = fix_angle(f, 2 * np.pi, 0)
            return f
        else:
            ea_guess = ea_next
    raise RuntimeError('Newton''s method did not converge.')


def euler313(vector: np.ndarray, psi: float, theta: float, phi: float) -> np.ndarray:
    dcm = np.array([[  cos(psi) * cos(phi) - sin(psi) * sin(phi) * cos(theta),  cos(psi) * sin(phi) + sin(psi) * cos(theta) * cos(phi), sin(psi) * sin(theta) ],
                    [ -sin(psi) * cos(phi) - cos(psi) * sin(phi) * cos(theta), -sin(psi) * sin(phi) + cos(psi) * cos(theta) * cos(phi), cos(psi) * sin(theta) ],
                    [                                   sin(theta) * sin(phi),                                  -sin(theta) * cos(phi),            cos(theta) ]]).T
    return np.matmul(dcm, vector.T.reshape(-1, 3, 1)).squeeze()


def euler1(vector: np.ndarray, theta: float) -> np.ndarray:
    dcm = np.array([[1,              0,             0],
                    [0,  np.cos(theta), np.sin(theta)],
                    [0, -np.sin(theta), np.cos(theta)]])
    return np.matmul(dcm, vector)


def E_from_f_e(f: float, e: float) -> float:
    return np.arctan2(sin(f) * (1 - e * e) ** 0.5, e + cos(f))


def f_from_E_e(E: float, e: float) -> float:
    return np.arctan2(sin(E) * (1 - e - e) ** 0.5, cos(E) - e)


def fpa_from_E_e(E: float, e: float) -> float:
    return np.arctan2(e * sin(E), (1 - e * e) ** 0.5)


def fpa_from_h_r_v(h: float, r: float, v: float) -> float:
    return np.arccos(h / r / v)


def fpa_parabolic(f: float) -> float:
    return f / 2


def f_parabolic_from_B_p_r(B: float, p: float, r: float) -> float:
    return np.arctan2(p * B, p - r)


def E_from_M_e(M: float, e: float, tol: float = 1e-8) -> float:
    E_guess = M
    for i in range(20):
        E_next = E_guess + (M + e * sin(E_guess) - E_guess) / (1 - e * cos(E_guess))
        if abs(E_next - E_guess) < tol:
            return E_next
        else:
            E_guess = E_next
    raise RuntimeError('Newton''s method did not converge.')


def M_from_E_e(E: float, e: float) -> float:
    return E - e * sin(E)


def r_from_a_e_f(a: float, e: float, f: float) -> float:
    return a * (1 - e * e) / (1 + e * cos(f))


def rp_from_a_e(a: float, e: float) -> float:
    return a * (1 - e)


def ra_from_a_e(a: float, e: float) -> float:
    return a * (1 + e)


def p_from_a_e(a: float, e: float) -> float:
    return a * (1 - e * e)


def a_from_gm_energy(gm: float, energy: float) -> float:
    return - gm / 2 / energy


def v_from_gm_eps_r(gm: float, eps: float, r: float) -> float:
    return (2 * (gm / r + eps)) ** 0.5


def v_from_gm_r_a(gm: float, r: float, a: float) -> float:
    return (2 * gm / r - gm / a) ** 0.5


def v_from_gm_r_e_f(gm: float, r: float, e: float, f: float) -> float:
    return (gm / r * (2 - (1 - e - e) / (1 + e * cos(f)))) ** 0.5


def n_from_gm_a(gm: float, a: float) -> float:
    return (gm / a / a / a) ** 0.5


def n_from_per(per: float) -> float:
    return 2 * np.pi / per


def per_from_gm_a(gm: float, a: float) -> float:
    return 2 * np.pi * (a * a * a / gm) ** 0.5


def v_parabolic(gm: float, r: float) -> float:
    return (2 * gm / r) ** 0.5


def per_from_rp_ra(rp: float, ra: float, gm: float) -> float:
    a = (rp + ra) / 2
    return 2 * np.pi * (a * a * a / gm) ** 0.5


def a_from_gm_per(gm: float, per: float) -> float:
    return (gm * (per / 2 / np.pi) ** 2) ** (1 / 3)


def ra_from_rp_per(rp: float, per: float, gm: float) -> float:
    a = a_from_gm_per(gm, per)
    return 2 * a - rp


def f_from_r_a_e(r: float, a: float, e: float, sign: float) -> float:
    return np.arccos((a * (1 - e * e) / r - 1) / e) * sign


def energy_from_gm_r_v(gm: float, r: float, v: float) -> float:
    return v * v / 2 - gm / r


def energy_from_gm_a(gm: float, a: float) -> float:
    return - gm / 2 / a


def e_vec_from_gm_v_r(gm: float, v_vec: np.ndarray, r_vec: np.ndarray) -> np.ndarray:
    r_mag, v_mag = mag3(r_vec), mag3(v_vec)
    return ((v_mag * v_mag - gm / r_mag) * r_vec - np.dot(r_vec, v_vec) * v_vec) / gm


def a_e_from_rp_ra(rp: float, ra: float) -> (float, float):
    e = (ra - rp) / (rp + ra)
    a = rp / (1 - e)
    return a, e


def a_from_rp_ra(rp: float, ra: float) -> float:
    return 0.5 * (rp + ra)


def e_from_rp_ra(rp: float, ra: float) -> float:
    return (ra - rp) / (ra + rp)


def h_from_gm_a_e(gm: float, a: float, e: float) -> float:
    return (gm * a * (1 - e * e)) ** 0.5


def gamma_from_h_r_v_qcheck(h: float, r: float, v: float, r_vec: np.ndarray, v_vec: np.ndarray) -> float:
    return np.arccos(h / r / v) * sign_check(r_vec, v_vec)


def gamma_from_h_r_v(h: float, r: float, v: float) -> float:
    return np.arccos(h / r / v)


def f_from_r_p_e_qcheck(r: float, p: float, e: float, r_vec: np.ndarray, v_vec: np.ndarray) -> float:
    return np.arccos((p / r - 1) / e) * sign_check(r_vec, v_vec)


def f_from_r_p_e(r: float, p: float, e: float) -> float:
    return np.arccos((p / r - 1) / e)


def sign_check(r_vec: np.ndarray, v_vec: np.ndarray) -> float:
    return np.sign(np.dot(r_vec, v_vec))


def f_from_gamma_r_v_gm(gamma: float, r: float, v: float, gm: float) -> float:
    tmp = r * v * v / gm * cos(gamma)
    return np.arctan2(tmp * sin(gamma), tmp * cos(gamma) - 1)


def time_to_apoapsis_from_per_M_n(per: float, M: float, n: float) -> float:
    return per / 2 - M / n


def time_to_periapsis_from_M_n(M: float, n: float) -> float:
    return M / n


def H_from_f_e(f: float, e: float) -> float:
    return np.arctanh(sin(f) * (e * e - 1) ** 0.5 / (e + cos(f)))


def lower_periapsis(ra_cur: float, a_cur: float, rp_new: float, gm: float) -> float:
    a_new = (ra_cur + rp_new) / 2
    va_cur = (2 * gm / ra_cur - gm / a_cur) ** 0.5
    va_new = (2 * gm / ra_cur - gm / a_new) ** 0.5
    return va_new - va_cur


def raise_periapsis(ra_cur: float, a_cur: float, rp_new: float, gm: float) -> float:
    return lower_periapsis(ra_cur, a_cur, rp_new, gm)


def lower_apoapsis(rp_cur: float, a_cur: float, ra_new: float, gm: float) -> float:
    a_new = (rp_cur + ra_new) / 2
    vp_cur = (2 * gm / rp_cur - gm / a_cur) ** 0.5
    vp_new = (2 * gm / rp_cur - gm / a_new) ** 0.5
    return vp_new - vp_cur


def raise_apoapsis(rp_cur: float, a_cur: float, ra_new: float, gm: float) -> float:
    return lower_apoapsis(rp_cur, a_cur, ra_new, gm)


# TODO make functions for various capture types
#    - starting conic: hyperbolic or elliptical
#    - starting inside or outside SOI (elliptical can't start outside)
#    - final capture orbit: low circular or high elliptical


def capture(state_rel: np.ndarray, rp_target: float, per_target: float, gm: float, r_SOI: float, capture_low: bool,
            current: bool) -> list:
    """
    Calls the appropriate function to capture a spacecraft around the target body based on the spacecraft's current
    orbit. Decisions include starting from hyperbolic or elliptical orbit, targeting a low circular or high elliptical
    final orbit, starting inside or outside the sphere of influence (hyperbolic only), and whether the maneuver could
    occur at the optimal point of the current orbit or force it to be at the current location.
    :param state_rel:
    :param rp_target:
    :param per_target:
    :param gm:
    :param r_SOI:
    :param capture_low:
    :param current:
    :return:
    """
    r_mag, v_mag = mag3(state_rel[:3]), mag3(state_rel[3:])
    energy = v_mag * v_mag / 2 - gm / r_mag
    char1 = 'h' if energy > 0 else 'e'
    char2 = 'o' if r_mag > r_SOI else 'i'
    char3 = 'l' if capture_low else 'h'
    char4 = 'c' if current else 'o'
    capture_type = char1 + char2 + char3 + char4
    # TODO the difference between low and high capture is an additional maneuver at periapsis - so just worry about high
    #  capture and then make another function that can optionally compute the final maneuver
    capture_methods = {'holc': _hyperbolic_out_low_current,  'holo': _hyperbolic_out_low_optimal,
                       'hilc': _hyperbolic_in_low_current,   'hilo': _hyperbolic_in_low_optimal,
                       'hohc': _hyperbolic_out_high_current, 'hoho': _hyperbolic_out_high_optimal,
                       'hihc': _hyperbolic_in_high_current,  'hiho': _hyperbolic_in_high_optimal,
                       'eilc': _elliptical_low_current,      'eilo': _elliptical_low_optimal,
                       'eihc': _elliptical_high_current,     'eiho': _elliptical_high_optimal}
    maneuvers = capture_methods[capture_type](state_rel, rp_target, per_target, gm)
    return maneuvers


# Finished - but I may want to change first maneuver
def _hyperbolic_out_low_current(state_0: np.ndarray, rp_target: float, per_target: float, gm: float) -> list:
    """
    Determine flight path angle that will set periapsis to desired periapsis distance on a parabolic trajectory.
    At periapsis, circularize. This should work whether we are ascending or descending.
    """
    r_vec, v_vec = state_0[:3], state_0[3:]
    r_mag, v_mag = mag3(r_vec), mag3(v_vec)
    vp_para = (2 * gm / rp_target) ** 0.5  # parabolic speed at target periapsis
    v_mag_para = (2 * gm / r_mag) ** 0.5  # speed along parabolic orbit at current distance
    # Compute new flight path angle (gamma)
    p = 2 * rp_target
    h_mag = mag3(cross(r_vec, v_vec))
    gamma = np.arccos(h_mag / r_mag / v_mag) * np.sign(np.dot(r_vec, v_vec))
    n = 2 * (gm / p ** 3) ** 0.5
    B = (2 * r_mag / p - 1) ** 0.5
    tof = (B ** 3 / 3 + + B) / n  # time to periapsis
    f = abs(np.arctan2(p * B, p - r_mag)) * -1.  # make sure we are descending
    gamma_para = f / 2
    d_gamma = gamma_para - gamma
    # Use law of cosines to determine dv1_mag and angle of dv1 in VNC coords (alpha)
    dv1_mag = (v_mag ** 2 + v_mag_para ** 2 - 2 * v_mag * v_mag_para * cos(d_gamma)) ** 0.5
    beta = np.arccos((v_mag ** 2 + dv1_mag ** 2 - v_mag_para ** 2) / 2 / v_mag / dv1_mag)
    alpha = np.pi - beta
    theta = alpha + np.arctan2(r_vec[1], r_vec[0]) + np.pi / 2 - gamma  # TODO how should arctan(y/x) be changed for 3D?
    dv1_vec = np.array([dv1_mag * cos(theta), dv1_mag * sin(theta), 0])  # TODO how should dv1_vec be changed for 3D?
    v_transfer_vec = v_vec + dv1_vec
    # Compute delta v to get into target orbit at periapsis
    v_final = (gm / rp_target) ** 0.5
    dv2_mag = v_final - vp_para
    # Compute delta v vector and final velocity vector
    v_periapsis_vec = v_transfer_vec / v_mag_para * vp_para
    v_periapsis_vec = euler313(v_periapsis_vec, 0, 0, - f + gamma_para)
    dv2_vec = v_periapsis_vec / vp_para * dv2_mag
    maneuvers = [dv1_vec, dv2_vec, tof]
    return maneuvers


# Finished - there *may* be a way to optimally adjust energy and orientation instead of just orientation
def _hyperbolic_out_low_optimal(state_0: np.ndarray, rp_target: float, per_target: float, gm: float) -> (list, np.ndarray):
    """
    :param state_0:
    :return:
    """
    # Define initial orbit in Keplerian elements - then determine Cartesian state
    state_kep = inertial_to_keplerian_3d(state_0, gm)
    # Then determine f when r = r_far_away
    r_far = 10 * c.r_soi_mars
    assert mag3(state_0) < r_far, 'Spacecraft is too far from planet for this capture method to apply.'
    sign = -1
    f_opt = f_from_r_a_e(r_far, state_kep[0], state_kep[1], sign)
    state_kep[-1] = f_opt
    # Then determine Cartesian state
    state_opt = keplerian_to_inertial_3d(state_kep, gm)
    # Now calculate TCM from new location
    r_vec, v_vec = state_opt[:3], state_opt[3:]
    r_mag, v_mag = mag3(r_vec), mag3(v_vec)
    vinf2 = v_mag * v_mag - 2 * gm / r_mag
    a = -gm / vinf2
    e = 1 - rp_target / a
    h = (gm * a * (1 - e * e)) ** 0.5
    gamma0 = gamma_from_r_v(r_vec, v_vec)
    gamma1 = np.arccos(h / r_mag / v_mag) * -1
    dgamma = gamma1 - gamma0
    f = np.arccos((a / r_mag * (1 - e * e) - 1) / e) * -1
    ha = np.arccosh((r_mag / abs(a) + 1) / e) * -1
    ma = e * np.sinh(ha) - ha
    tof = abs((-(a ** 3) / gm) ** 0.5 * ma)
    dv1_mag = (2 * v_mag * v_mag * (1 - np.cos(dgamma))) ** 0.5
    beta = np.pi - np.arcsin(v_mag / dv1_mag * sin(abs(dgamma)))
    dv1_vec = np.array([cos(beta), 0., -sin(beta)]) * dv1_mag  # NOTE: ASSUMES 2D
    dv1_vec = rotate_vnc_to_inertial_3d(dv1_vec, state_opt)
    vp_hyper = (vinf2 + 2 * gm / rp_target) ** 0.5
    vp_capture = (gm / rp_target) ** 0.5
    # print('vp_hyper estimate = %f' % vp_hyper)
    dv2_mag = vp_capture - vp_hyper
    dv2_vec = (v_vec + dv1_vec) / mag3(v_vec + dv1_vec) * dv2_mag
    dalpha = (0 - f) - (0 - gamma1)
    dv2_vec = euler313(dv2_vec, 0, 0, dalpha * np.sign(cross(r_vec, v_vec)[-1]))
    maneuvers = [dv1_vec, dv2_vec, tof]
    return maneuvers, state_opt


# Finished - but I may want to change first maneuver
def _hyperbolic_out_high_current(state_0: np.ndarray, rp_target: float, per_target: float, gm: float) -> list:
    """
    :param state_0:
    :return:
    """
    r_vec, v_vec = state_0[:3], state_0[3:]
    r_mag, v_mag = mag3(r_vec), mag3(v_vec)
    vp_para = (2 * gm / rp_target) ** 0.5  # parabolic speed at target periapsis
    v_mag_para = (2 * gm / r_mag) ** 0.5  # speed along parabolic orbit at current distance
    # Compute new flight path angle (gamma)
    p = 2 * rp_target
    h_mag = mag3(cross(r_vec, v_vec))
    gamma = np.arccos(h_mag / r_mag / v_mag) * np.sign(np.dot(r_vec, v_vec))
    n = 2 * (gm / p ** 3) ** 0.5
    B = (2 * r_mag / p - 1) ** 0.5
    tof = (B ** 3 / 3 + + B) / n  # time to periapsis
    f = abs(np.arctan2(p * B, p - r_mag)) * -1.  # make sure we are descending
    gamma_para = f / 2
    d_gamma = gamma_para - gamma
    # Use law of cosines to determine dv1_mag and angle of dv1 in VNC coords (alpha)
    dv1_mag = (v_mag ** 2 + v_mag_para ** 2 - 2 * v_mag * v_mag_para * cos(d_gamma)) ** 0.5
    beta = np.arccos((v_mag ** 2 + dv1_mag ** 2 - v_mag_para ** 2) / 2 / v_mag / dv1_mag)
    alpha = np.pi - beta
    theta = alpha + np.arctan2(r_vec[1], r_vec[0]) + np.pi / 2 - gamma  # TODO how should arctan(y/x) be changed for 3D?
    dv1_vec = np.array([dv1_mag * cos(theta), dv1_mag * sin(theta), 0])  # TODO how should dv1_vec be changed for 3D?
    v_transfer_vec = v_vec + dv1_vec
    # Compute delta v to get into target orbit at periapsis
    ra_target = ra_from_rp_per(rp_target, per_target, gm)
    a_target = (ra_target + rp_target) / 2
    v_final = v_from_gm_r_a(gm, rp_target, a_target)
    dv2_mag = v_final - vp_para
    # Compute delta v vector and final velocity vector
    v_periapsis_vec = v_transfer_vec / v_mag_para * vp_para
    v_periapsis_vec = euler313(v_periapsis_vec, 0, 0, - f + gamma_para)
    dv2_vec = v_periapsis_vec / vp_para * dv2_mag
    maneuvers = [dv1_vec, dv2_vec, tof]
    return maneuvers


# Finished - there *may* be a way to optimally adjust energy and orientation instead of just orientation
def _hyperbolic_out_high_optimal(state_0, rp_target: float, per_target: float, gm: float) -> (list, np.ndarray):
    """
    :param state_0:
    :return:
    """
    # Define initial orbit in Keplerian elements - then determine Cartesian state
    state_kep = inertial_to_keplerian_3d(state_0, gm)
    # Then determine f when r = r_far_away
    r_far = 10 * c.r_soi_mars
    sign = -1
    f_opt = f_from_r_a_e(r_far, state_kep[0], state_kep[1], sign)
    state_kep[-1] = f_opt
    # Then determine Cartesian state
    state_opt = keplerian_to_inertial_3d(state_kep, gm)
    # Now calculate TCM from new location
    r_vec, v_vec = state_opt[:3], state_opt[3:]
    r_mag, v_mag = mag3(r_vec), mag3(v_vec)
    vinf2 = v_mag * v_mag - 2 * gm / r_mag
    a = -gm / vinf2
    e = 1 - rp_target / a
    h = (gm * a * (1 - e * e)) ** 0.5
    gamma0 = gamma_from_r_v(r_vec, v_vec)
    gamma1 = np.arccos(h / r_mag / v_mag) * -1
    dgamma = gamma1 - gamma0
    f = np.arccos((a / r_mag * (1 - e * e) - 1) / e) * -1
    ha = np.arccosh((r_mag / abs(a) + 1) / e) * -1
    ma = e * np.sinh(ha) - ha
    tof = abs((-(a ** 3) / gm) ** 0.5 * ma)
    dv1_mag = (2 * v_mag * v_mag * (1 - np.cos(dgamma))) ** 0.5
    beta = np.pi - np.arcsin(v_mag / dv1_mag * sin(abs(dgamma)))
    dv1_vec = np.array([cos(beta), 0., -sin(beta)]) * dv1_mag  # NOTE: ASSUMES 2D
    dv1_vec = rotate_vnc_to_inertial_3d(dv1_vec, state_opt)
    vp_hyper = (vinf2 + 2 * gm / rp_target) ** 0.5
    a_capture = a_from_gm_per(gm, per_target)
    vp_capture = v_from_gm_r_a(gm, rp_target, a_capture)
    # print('vp_hyper estimate = %f' % vp_hyper)
    dv2_mag = vp_capture - vp_hyper
    dv2_vec = (v_vec + dv1_vec) / mag3(v_vec + dv1_vec) * dv2_mag
    dalpha = (0 - f) - (0 - gamma1)
    dv2_vec = euler313(dv2_vec, 0, 0, dalpha * np.sign(cross(r_vec, v_vec)[-1]))
    maneuvers = [dv1_vec, dv2_vec, tof]
    return maneuvers, state_opt


# Finished
def _hyperbolic_in_low_current(state_0: np.ndarray, rp_target: float, per_target: float, gm: float) -> list:
    """
    :param state_0:
    :param rp_target:
    :return:
    """
    r0_vec, v0_vec = state_0[:3], state_0[3:]
    r0_mag, v0_mag = mag3(r0_vec), mag3(v0_vec)
    h0_vec = cross(r0_vec, v0_vec)
    h0_mag = mag3(h0_vec)
    gamma = gamma_from_h_r_v_qcheck(h0_mag, r0_mag, v0_mag, r0_vec, v0_vec)
    ra_target = ra_from_rp_per(rp_target, per_target, gm)
    # Conserve r_vec, gamma; only change v_mag such that ra becomes ra_target
    v1_mag = ((2 * gm * ra_target * (1 - ra_target / r0_mag)) / (r0_mag ** 2 * cos(gamma) ** 2 - ra_target ** 2)) ** 0.5
    dv1_mag = v1_mag - v0_mag
    dv1_vec = v0_vec / v0_mag * dv1_mag
    v1_vec = v0_vec + dv1_vec

    energy = energy_from_gm_r_v(gm, r0_mag, v1_mag)
    a = a_from_gm_energy(gm, energy)
    e_vec = e_vec_from_gm_v_r(gm, v1_vec, r0_vec)
    e_mag = mag3(e_vec)
    rp = rp_from_a_e(a, e_mag)

    # Check that periapsis is above surface
    if rp < rp_target:  # Assume that if rp < rp_target, then rp < r_planet
        v1_mag = ((2 * gm * rp_target * (1 - rp_target / r0_mag)) / (r0_mag ** 2 * cos(gamma) ** 2 - rp_target ** 2)) ** 0.5
        dv1_mag = v1_mag - v0_mag
        dv1_vec = v0_vec / v0_mag * dv1_mag
        v1_vec = v0_vec + dv1_vec

        # At periapsis, circularize
        energy = energy_from_gm_r_v(gm, r0_mag, v1_mag)
        a = a_from_gm_energy(gm, energy)
        e_vec = e_vec_from_gm_v_r(gm, v1_vec, r0_vec)
        e_mag = mag3(e_vec)
        rp = rp_from_a_e(a, e_mag)
        vp = v_from_gm_r_a(gm, rp, a)
        v2_mag = (gm / rp_target) ** 0.5
        dv2_mag = v2_mag - vp
        a, e, i, w, om, f = inertial_to_keplerian_3d(np.hstack((r0_vec, v1_vec)), gm)
        f = fix_angle(f)
        df = 0 - f
        dgamma = 0 - gamma
        at_periapsis = True
    else:
        # At apoapsis, lower periapsis to target
        ra = a * (1 + e_mag)
        dv2_mag = lower_periapsis(ra, a, rp_target, gm)
        a, e, i, w, om, f = inertial_to_keplerian_3d(np.hstack((r0_vec, v1_vec)), gm)
        df = np.pi - f
        dgamma = 0 - gamma
        at_periapsis = False

    # Account for sign difference if orbit is "backwards"
    if i > np.pi / 2 or i < - np.pi / 2:
        df *= -1
        dgamma *= -1

    # Compute 2nd delta V
    dv2_vec = v1_vec / v1_mag * dv2_mag
    dv2_vec = euler313(dv2_vec, 0, 0, df - dgamma)

    # Compute TOF between 1st and 2nd burns, accounting for elliptic vs. hyperbolic
    if e < 1:
        E = E_from_f_e(f, e)
        M = M_from_E_e(E, e)
        per = per_from_gm_a(gm, a)
        n = n_from_per(per)
        tof12 = time_to_apoapsis_from_per_M_n(per, M, n)
    else:
        H = H_from_f_e(f, e) * sign_check(r0_vec, v1_vec)
        M = e * np.sinh(H) - H
        n = (gm / - a ** 3) ** 0.5
        tof12 = time_to_periapsis_from_M_n(M, n)

    if at_periapsis:
        maneuvers = [dv1_vec, dv2_vec, tof12]
    else:
        # At new periapsis, circularize
        a = (ra + rp_target) / 2
        vp_transfer = (gm * (2 / rp_target -  1 / a)) ** 0.5
        dv3_mag = (gm / rp_target) ** 0.5 - vp_transfer
        dv3_vec = dv2_vec / dv2_mag * -dv3_mag
        per = 2 * np.pi * (a * a * a / gm) ** 0.5
        tof23 = per / 2
        maneuvers = [dv1_vec, dv2_vec, dv3_vec, tof12, tof23]

    return maneuvers


# Finished
def _hyperbolic_in_low_optimal(state_0: np.ndarray, rp_target: float, per_target: float, gm: float) -> (list, np.ndarray):
    """
    :param state_0:
    :param rp_target:
    :return:
    """
    # Define initial orbit in Keplerian elements
    state_kep = inertial_to_keplerian_3d(state_0, gm)
    state_kep[-1] = 0
    state_opt = keplerian_to_inertial_3d(state_kep, gm)
    maneuvers = _hyperbolic_in_low_current(state_opt, rp_target, per_target, gm)
    return maneuvers, state_opt


# Finished
def _hyperbolic_in_high_current(state_0, rp_target: float, per_target: float, gm: float) -> list:
    """
    :param state_0:
    :param rp_target:
    :param per_target:
    :param gm:
    :return:
    """
    r0_vec, v0_vec = state_0[:3], state_0[3:]
    r0_mag, v0_mag = mag3(r0_vec), mag3(v0_vec)
    h0_vec = cross(r0_vec, v0_vec)
    h0_mag = mag3(h0_vec)
    gamma = gamma_from_h_r_v_qcheck(h0_mag, r0_mag, v0_mag, r0_vec, v0_vec)
    ra_target = ra_from_rp_per(rp_target, per_target, gm)
    # Conserve r_vec, gamma; only change v_mag such that ra becomes ra_target
    v1_mag = ((2 * gm * ra_target * (1 - ra_target / r0_mag)) / (r0_mag ** 2 * cos(gamma) ** 2 - ra_target ** 2)) ** 0.5
    dv1_mag = v1_mag - v0_mag
    dv1_vec = v0_vec / v0_mag * dv1_mag
    v1_vec = v0_vec + dv1_vec

    energy = energy_from_gm_r_v(gm, r0_mag, v1_mag)
    a = a_from_gm_energy(gm, energy)
    e_vec = e_vec_from_gm_v_r(gm, v1_vec, r0_vec)
    e_mag = mag3(e_vec)
    rp = rp_from_a_e(a, e_mag)

    # Check that periapsis is above surface
    if rp < rp_target:  # Assume that if rp < rp_target, then rp < r_planet
        v1_mag = ((2 * gm * rp_target * (1 - rp_target / r0_mag)) / (r0_mag ** 2 * cos(gamma) ** 2 - rp_target ** 2)) ** 0.5
        dv1_mag = v1_mag - v0_mag
        dv1_vec = v0_vec / v0_mag * dv1_mag
        v1_vec = v0_vec + dv1_vec

        # At periapsis, raise apoapsis to target
        energy = energy_from_gm_r_v(gm, r0_mag, v1_mag)
        a = a_from_gm_energy(gm, energy)
        e_vec = e_vec_from_gm_v_r(gm, v1_vec, r0_vec)
        e_mag = mag3(e_vec)
        rp = rp_from_a_e(a, e_mag)
        dv2_mag = lower_apoapsis(rp, a, ra_target, gm)
        a, e, i, w, om, f = inertial_to_keplerian_3d(np.hstack((r0_vec, v1_vec)), gm)
        f = fix_angle(f)
        df = 0 - f
        dgamma = 0 - gamma
    else:
        # At apoapsis, lower periapsis to target
        ra = a * (1 + e_mag)
        dv2_mag = lower_periapsis(ra, a, rp_target, gm)
        a, e, i, w, om, f = inertial_to_keplerian_3d(np.hstack((r0_vec, v1_vec)), gm)
        df = np.pi - f
        dgamma = 0 - gamma
    if i > np.pi / 2:
        df *= -1
        dgamma *= -1
    dv2_vec = v1_vec / v1_mag * dv2_mag
    dv2_vec = euler313(dv2_vec, 0, 0, df - dgamma)
    if e < 1:
        E = E_from_f_e(f, e)
        M = M_from_E_e(E, e)
        per = per_from_gm_a(gm, a)
        n = n_from_per(per)
        tof = time_to_apoapsis_from_per_M_n(per, M, n)
    else:
        H = H_from_f_e(f, e) * sign_check(r0_vec, v1_vec)
        M = e * np.sinh(H) - H
        n = (gm / - a ** 3) ** 0.5
        tof = time_to_periapsis_from_M_n(M, n)
    maneuvers = [dv1_vec, dv2_vec, tof]
    return maneuvers


# Finished
def _hyperbolic_in_high_optimal(state_0, rp_target: float, per_target: float, gm: float) -> (list, np.ndarray):
    """
    :param state_0:
    :param rp_target:
    :param per_target:
    :return:
    """
    state_kep = inertial_to_keplerian_3d(state_0, gm)
    state_kep[-1] = 0
    state_opt = keplerian_to_inertial_3d(state_kep, gm)
    maneuvers = _hyperbolic_in_high_current(state_opt, rp_target, per_target, gm)
    return maneuvers, state_opt


# Not finished - not needed at this time
def _elliptical_low_current(state_0: np.ndarray, rp_target: float, per_target: float, gm: float) -> list:
    """
    TODO lower energy such that new orbit has a periapsis at target distance, then circularize
    TODO this implementation is not correct
    :param state_0:
    :param rp_target:
    :return:
    """
    # r_vec, v_vec = state_0[:3], state_0[3:]
    # r0, v0 = mag3(r_vec), mag3(v_vec)
    # a0, e0, i0, w0, om0, f0 = inertial_to_keplerian_3d(np.hstack((r_vec, v_vec)), gm)
    # eps0 = - gm / 2 / a0
    # ra0 = a0 * (1 + e0)
    # va0 = (2 * (eps0 + gm / ra0)) ** 0.5
    # h0 = (gm * a0 * (1 - e0 * e0)) ** 0.5
    # gamma0 = np.arccos(h0 / r0 / v0)
    # gamma0 *= 1 if f0 < np.pi else -1
    #
    # at = ra0 + r_final
    # epst = - gm / 2 / at
    # vpt = (2 * (epst + gm / r_final)) ** 0.5
    # vat = (2 * (epst + gm / ra0)) ** 0.5
    #
    # dv1_mag = vat - va0
    # dv2_mag = v_final - vpt
    #
    # dv1_vec = np.array([dv1_mag, 0, 0])
    # dv1_vec = euler313(dv1_vec, 0, 0, f0 - np.pi - gamma0)
    #
    # dv2_vec = np.array([dv2_mag, 0, 0])
    #
    # # Compute velocity vector after the maneuvers
    # dv1_vec = v_hat * dv1_mag
    # v_transfer_vec = v_vec + dv1_vec
    # v_mag_capture_kms = mag3(v_transfer_vec)
    # a, e, i, w, o, f = inertial_to_keplerian_3d(np.hstack((r_vec, v_transfer_vec)), gm)
    # gamma_para = gamma
    # n = (gm / a ** 3) ** 0.5
    # E = E_from_f_e(f, e)
    # M = M_from_E_e(E, e)
    # dt = abs(fix_angle(M)) / n
    dv1_vec = np.zeros(3)
    dv2_vec = np.zeros(3)
    tof = 0.
    maneuvers = [dv1_vec, dv2_vec, tof]
    return maneuvers


# Not finished - not needed at this time
def _elliptical_low_optimal(state_0: np.ndarray, rp_target: float, per_target: float, gm: float) -> list:
    """
    TODO Two options - at apo, adjust peri, then at peri finish the capture; or, at peri, bring apo down to target peri.
         Is the first option always better?
    :param state_0:
    :param rp_target:
    :return:
    """
    dv1_vec = np.zeros(3)
    dv2_vec = np.zeros(3)
    tof = 0.
    maneuvers = [dv1_vec, dv2_vec, tof]
    return maneuvers


# Not finished - not needed at this time
def _elliptical_high_current(state_0, rp_target: float, per_target: float, gm: float) -> list:
    """
    TODO compute new energy and flight path angle required to rotate and resize orbit to be in target orbit. Probably
         a very expensive option compared to _elliptical_high_optimal().
    :param state_0:
    :param rp_target:
    :param per_target:
    :return:
    """
    # r0_vec, v0_vec = state_0[:3], state_0[3:]
    # r0_mag, v0_mag = mag3(r0_vec), mag3(v0_vec)
    # h0_vec = cross(r0_vec, v0_vec)
    # h0_mag = mag3(h0_vec)
    # gamma = np.arccos(h0_mag / r0_mag / v0_mag) * np.sign(h0_vec[2])
    # # gamma = np.dot(r0_vec, v0_vec) / (r0_mag * v0_mag)
    # ra_target = ra_from_rp_per(rp_target, per_target, gm)
    # # Option 1: conserve r_vec, gamma; only change v_mag such that ra becomes ra_target
    # v1_mag = ((2 * gm * ra_target * (1 - ra_target / r0_mag)) / (r0_mag ** 2 * cos(gamma) ** 2 - ra_target ** 2)) ** 0.5
    # dv1_mag = v1_mag - v0_mag
    # dv1_vec = v0_vec / v0_mag * dv1_mag
    # v1_vec = v0_vec + dv1_vec
    #
    # # At apoapsis, lower periapsis to target
    # energy = energy_from_gm_r_v(gm, r0_mag, v1_mag)
    # a = a_from_gm_energy(gm, energy)
    # e_vec = e_vec_from_gm_v_r(gm, v1_vec, r0_vec)
    # e_mag = mag3(e_vec)
    # rp = rp_from_a_e(a, e_mag)
    # a, e_mag = a_e_from_rp_ra(rp_target, ra_target)
    # energy = energy_from_gm_a(gm, a)
    # v1_mag = v_from_gm_eps_r(gm, energy, r0_mag)
    # h = h_from_gm_a_e(gm, a, e_mag)
    # gamma = gamma_from_h_r_v_qcheck(h, r0_mag, v1_mag, r0_vec, v1_vec)
    # f = f_from_gamma_r_v_gm(gamma, r0_mag, v1_mag, gm)
    # v1_vec = np.array([sin(gamma), cos(gamma), 0]) * v1_mag
    # v1_vec = euler313(v1_vec, 0, 0, (f - 0))
    # dv1_vec = v1_vec - v0_vec
    # ra = a * (1 + e_mag)
    # dv2_mag = lower_periapsis(ra, a, rp_target, gm)
    # a, e, i, w, om, f = inertial_to_keplerian_3d(np.hstack((r0_vec, v1_vec)), gm)
    # df = np.pi - f
    # dgamma = 0 - gamma
    # dv2_vec = v1_vec / v1_mag * dv2_mag
    # dv2_vec = euler313(dv2_vec, 0, 0, df - dgamma)  # * np.sign(h0_vec[2])
    # E = np.arctan2(sin(f) * (1 - e * e) ** 0.5, e + cos(f))
    # M = E - e * sin(E)
    # per = 2 * np.pi * (a * a * a / gm) ** 0.5
    # n = 2 * np.pi / per
    # tof = per / 2 - M / n

    dv1_vec = np.zeros(3)
    dv2_vec = np.zeros(3)
    tof = 0.
    maneuvers = [dv1_vec, dv2_vec, tof]
    return maneuvers


# Not finished - not needed at this time
def _elliptical_high_optimal(state_0, rp_target: float, per_target: float, gm: float) -> list:
    """
    TODO There are two options - at apo adjust peri, then at peri adjust apo; or, vice versa. Calculate both strats and
         then choose the lower total cost. Is one necessarily better than the other depending on where you are in an
         orbit? What about tof cost? Maybe just default to doing each maneuver as they come - i.e. if descending, will
         adjust apoapsis first (arrive at periapsis first), and then adjust periapsis second (arrive at apoapsis second)
    :param state_0:
    :param rp_target:
    :param per_target:
    :return:
    """
    dv1_vec = np.zeros(3)
    dv2_vec = np.zeros(3)
    tof = 0.
    maneuvers = [dv1_vec, dv2_vec, tof]
    return maneuvers


if __name__ == "__main__":
    test1 = False  # Test min_dv_capture
    if test1:
        r = np.random.rand(2) * 4e5 - 2e5
        v = np.random.rand(2) * 4 - 2
        gm = 42328.372
        print(min_dv_capture(r, v, gm))

    test2 = False  # Test change_central_body
    if test2:
        r1 = np.array([10000, 5000, 0.])
        v1 = np.array([-3, -4, 0.])
        r2 = np.array([-150000000, 0, 0.])
        v2 = np.array([0, -30, 0.])
        r3 = change_central_body(r1, r2)
        v3 = change_central_body(v1, v2)
        print(r3)
        print(v3)

    test3 = False  # Test mean_to_true_anomaly
    if test3:
        m = 135 * np.pi / 180
        e = 0.1
        ta = mean_to_true_anomaly(m, e)
        print(ta)
        # Quick and dirty approximation
        ta2 = m + (2 * e - 0.25 * e ** 3) * sin(m) + 1.25 * e ** 2 * sin(2 * m) + 13 / 12 * e ** 3 * sin(3 * m)
        print(ta2)

    test4 = False  # Test euler313
    if test4:
        vec = np.array([1, 0, 0], float)
        a = np.pi / 2
        print(euler313(vec, a, 0, 0))
        print(euler313(vec, 0, a, 0))
        print(euler313(vec, 0, 0, a))
        print(euler313(vec, a, a, 0))
        print(euler313(vec, a, 0, a))
        print(euler313(vec, 0, a, a))
        print(euler313(vec, a, a, a))

    test5 = False  # Test keplerian to LLA
    if test5:
        gm = u_earth_km3s2
        n = 15.4891975521933 * 2 * np.pi / 24 / 3600  # rad / sec
        a = (gm / n / n) ** (1. / 3)  # km
        e = 0.0005270  # -
        i = 51.6460 * np.pi / 180 # deg
        w = 61.9928 * np.pi / 180  # deg
        om = 33.2488 * np.pi / 180  # deg
        m = 83.3154 * np.pi / 180  # deg
        state_i = np.squeeze(keplerian_to_inertial_3d(np.array([a, e, i, w, om, m]), gm=gm, mean_or_true='mean'))
        # state_p = np.squeeze(keplerian_to_perifocal_3d(np.array([a, e, i, w, om, m]), gm=gm, mean_or_true='mean'))
        # state_i = np.hstack((euler313(state_p[:3], om, i, w), euler313(state_p[3:], om, i, w)))
        print('Inertial:')
        print(state_i)
        print()

        r = np.linalg.norm(state_i[:3])
        lat = np.arcsin(state_i[2] / r) * 180 / np.pi
        lon = np.arctan2(state_i[1], state_i[0]) * 180 / np.pi
        alt = (r - 6378.135) * 1000
        print('LLA')
        print([lat, lon, alt])

    test6 = False  # Test rotate_vnc_to_inertial_3d
    if test6:
        vec_vnc = np.array([1, 0, 0], float)
        state = np.array([1e8, 0, 0, 0, 29, 0], float)
        vec_i = rotate_vnc_to_inertial_3d(vec_vnc, state)
        print(vec_i)

    test7 = False  # Test capture maneuvers
    if test7:
        from scipy import integrate
        from eom import eom2BP
        from constants import u_mars_km3s2, r_mars_km

        # state_0 = np.array([10000,  14000, 0., -1,  5, 0])  # ascending, positive momentum
        # state_0 = np.array([10000,  14000, 0., -1, -5, 0])  # descending, negative momentum
        # state_0 = np.array([10000, -14000, 0., -1,  5, 0])  # descending, positive momentum
        state_0 = np.array([10000, -14000, 0., -1, -5, 0])  # ascending, negative momentum
        r_vec, v_vec = state_0[:3], state_0[3:]
        gm = u_mars_km3s2

        # Do maneuver
        rp_target = c.r_mars_km + 500
        per_target = 10 * 86400
        maneuvers, *args = _hyperbolic_in_low_current(state_0, rp_target, per_target, gm)
        if len(args) == 1:
            state_opt = args[0]
            state_0 = state_opt
        elif len(args) == 2 or len(args) == 4:
            maneuvers = [maneuvers, *args]
        else:
            raise ValueError('Unknown or unprepared outputs from capture function.')
        r_vec, v_vec = state_0[:3], state_0[3:]
        if len(maneuvers) == 3:
            dv1_vec, dv2_vec, tof12 = maneuvers
            dv1_mag, dv2_mag = mag3(dv1_vec), mag3(dv2_vec)
        else:
            dv1_vec, dv2_vec, dv3_vec, tof12, tof23 = maneuvers
            dv1_mag, dv2_mag, dv3_mag = mag3(dv1_vec), mag3(dv2_vec), mag3(dv3_vec)

        print('DV1 mag (km/s):\t\t%f' % abs(dv1_mag))
        print('DV2 mag (km/s):\t\t%f' % abs(dv2_mag))
        if len(maneuvers) == 3:
            print('Total DV mag (km/s):\t%f' % (abs(dv1_mag) + abs(dv2_mag)))
        else:
            print('DV3 mag (km/s):\t\t%f' % abs(dv3_mag))
            print('Total DV mag (km/s):\t%f' % (abs(dv1_mag) + abs(dv2_mag) + abs(dv3_mag)))

        print('\nTime to second maneuver (hr):\t%f' % (tof12 / 3600))
        if len(maneuvers) > 3:
            print('Time to third maneuver (hr):\t%f' % (tof23 / 3600))

        tf = 15000.
        tol = 1e-10
        traj1 = integrate.solve_ivp(eom2BP, [0, -tf*1], state_0, atol=tol, rtol=tol)
        traj11 = integrate.solve_ivp(eom2BP, [0, tf*1], state_0, atol=tol, rtol=tol)
        v_transfer_vec = state_0[3:6] + dv1_vec
        state_transfer = np.hstack((r_vec, v_transfer_vec))
        # per_transfer = period_from_inertial(state_transfer, gm)
        traj2 = integrate.solve_ivp(eom2BP, [0, tof12], state_transfer, atol=tol, rtol=tol)
        print('vp_hyper actual = %f' % (mag3(traj2.y[3:, -1])))
        print('rp_hyper actual = %f' % (mag3(traj2.y[:3, -1])))

        vp_hat = traj2.y[3:, -1] / mag3(traj2.y[3:, -1])
        dv2_necessary = - mag3(dv2_vec) * vp_hat
        print('dv2_vec actual =')
        print(dv2_vec)
        print('dv2_vec necessary =')
        print(dv2_necessary)
        v3_vec = traj2.y[3:, -1] + dv2_vec
        ra_target = ra_from_rp_per(rp_target, per_target, gm)
        state3 = np.hstack((traj2.y[:3, -1], v3_vec))
        if len(maneuvers) == 3:
            tof23 = period_from_inertial(state3, gm, max_time_sec=tf)
        traj3 = integrate.solve_ivp(eom2BP, [0, tof23], state3, atol=tol, rtol=tol)
        if len(maneuvers) > 3:
            v4_vec = traj3.y[3:, -1] + dv3_vec
            state4 = np.hstack((traj3.y[:3, -1], v4_vec))
            per4 = period_from_inertial(state4, gm)
            traj4 = integrate.solve_ivp(eom2BP, [0, tf], state4, atol=tol, rtol=tol)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        circ = plt.Circle((0, 0), r_mars_km, color='red')
        ax.add_artist(circ)
        plt.plot(traj1.y[0], traj1.y[1])
        plt.gca().set_prop_cycle(None)
        plt.plot(traj11.y[0], traj11.y[1], '--')
        plt.scatter(traj1.y[0, 0], traj1.y[1, 0])
        plt.plot(traj2.y[0], traj2.y[1])
        plt.scatter(traj2.y[0, -1], traj2.y[1, -1])
        plt.plot(traj3.y[0], traj3.y[1])
        if len(maneuvers) > 3:
            plt.scatter(traj3.y[0, -1], traj3.y[1, -1])
            plt.plot(0, 0)
            plt.plot(traj4.y[0], traj4.y[1])
        ax.axis('equal')
        # x, y = traj2.y[0, -1], traj2.y[1, -1]
        # plt.quiver(x, y, dv2_vec[0], dv2_vec[1])
        plt.show()

    test8 = False  # Test _hyperbolic_in_high_current
    if test8:
        state_0 = np.array([20000, 20000, 0, -1, 6, 0])
        rp_target = 3000
        per_target = 20 * 86400
        gm = c.u_mars_km3s2
        maneuvers = _hyperbolic_in_high_current(state_0, rp_target, per_target, gm)
        [print(m) for m in maneuvers]

    test9 = False  # Test perpendicular line visualization
    if test9:
        import numpy as np
        import matplotlib.pyplot as plt


        # A point object that has an x and y coordinate
        class Point:
            def __init__(self, x, y):
                self.x: float = x
                self.y: float = y

            def __add__(self, p):
                return Point(self.x + p.x, self.y + p.y)

            def __sub__(self, p):
                return Point(self.x - p.x, self.y - p.y)

            def __mul__(self, p):
                return Point(self.x * p, self.y * p)

            def __truediv__(self, p):
                return Point(self.x / p, self.y / p)

            def dot(self, p):
                return self.x * p.x + self.y * p.y

            def mag(self):
                return mag2(self.x, self.y)


        # Magnitude of a 2-vector
        def mag2(x: float, y: float):
            return (x * x + y * y) ** 0.5


        # Plot point of intersection from orthogonal line between two vectors
        def shortest_distance(a: Point, b: Point, p: Point):
            num = abs((b.y - a.y) * p.x - (b.x - a.x) * p.y + b.x * a.y - b.y * a.x)
            den = mag2(b.y - a.y, b.x - a.x)
            return num / den


        test1 = True
        if test1:
            a = Point(1, 0)
            b = Point(2, 2)
            n = b - a
            n /= n.mag()
            p = Point(0.5, 1)
            d = shortest_distance(a, b, p)
            print('distance = %f' % d)

            plt.figure()
            plt.plot([a.x, b.x], [a.y, b.y])
            plt.scatter(p.x, p.y)

            a_minus_p = a - p
            ampdnn = n * a_minus_p.dot(n)
            amp_minus_ampdnn = a_minus_p - ampdnn
            perp = amp_minus_ampdnn
            print('mag(perp) = %f' % perp.mag())

            plt.plot([p.x, p.x + perp.x], [p.y, p.y + perp.y])
            plt.axis('equal')
            plt.show()

        test2 = True
        if test2:
            plt.figure()
            a = Point(0, 0)
            p = Point(0, 1)
            plt.scatter(p.x, p.y)
            a_minus_p = a - p

            mags = list()
            n_lines = 101
            for i in range(n_lines):
                b = Point(np.cos(np.pi / 2 * i / (n_lines - 1)), np.sin(np.pi / 2 * i / (n_lines - 1)))
                n = b - a
                n /= n.mag()
                plt.plot([a.x, b.x], [a.y, b.y])
                ampdnn = n * a_minus_p.dot(n)
                amp_minus_ampdnn = a_minus_p - ampdnn
                perp = amp_minus_ampdnn
                mags.append(perp.mag())
                # print('mag(perp) = %f' % perp.mag())
                plt.plot([p.x, p.x + perp.x], [p.y, p.y + perp.y])
                plt.scatter(p.x + perp.x, p.y + perp.y)
            plt.axis('equal')
            plt.show()

            plt.figure()
            mags = np.array(mags)
            angles = np.arange(90, 0, -90 / n_lines)
            plt.plot(angles, mags)
            plt.xlabel('Angle between old and new vectors (deg)')
            plt.ylabel('Normalized Delta V')
            plt.show()
