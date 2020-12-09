import numpy as np
from numpy import cos, sin
# from math import gamma
import math
from traj_config import gm
import traj_config as tc
from numba import njit, vectorize, float64
from copy import copy
from constants import ephem, sec_to_day, year_to_sec, reference_date_jd1950, day_to_jc, u_earth_km3s2
import constants as c
import boost_tbp
import warnings

lambert = boost_tbp.maneuvers().lambert
tbp = boost_tbp.TBP()


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

    eps = v * v / 2 - gm / r

    # Eccentricity
    e_vec = ((v * v - gm / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / gm
    e = mag3(e_vec)

    # Semi-major axis
    tol = 1e-6
    if e < (1 - tol):
        a = - gm / 2 / eps
        return 2 * np.pi * (a * a * a / gm) ** 0.5
    else:
        return max_time_sec


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
    r1, v1, th1, al1 = inertial_to_keplerian_2d(s1i)
    r2, v2, th2, al2 = inertial_to_keplerian_2d(s2i)
    if has_extra:
        extra = state[8:]
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


@njit
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
        w = 0
        ta = th
    else:
        arg = (p / r - 1) / e
        if abs(arg) > 1:
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


@njit
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
    k_vec = np.array([0., 0, 1])
    n_vec = cross(k_vec, h_vec)
    n = mag3(n_vec)

    # Eccentricity
    e_vec = ((v * v - gm / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / gm
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
    if i < tol or np.abs(np.pi - i) < tol:
        # Special case - equatorial
        om = 0.
    else:
        # General
        om = np.arccos(n_vec[0] / n)
        if n_vec[1] < 0:
            om = 2 * np.pi - om

    # Argument of periapsis
    # if n == 0 and False:  # TODO verify where this statement came from
    #     # Special case - equatorial
    #     w = np.arccos(np.dot(np.array([1, 0, 0]), e_vec))
    #     # w = 0.
    if i < tol or np.abs(np.pi - i) < tol:
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
        if np.dot(r_vec, v_vec) < 0:
            f = 2 * np.pi - f

    return np.array([a, e, i, w, om, f])


# TODO njit this
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
    r_p = np.vstack((r11, r12, r13))  # TODO speed up
    v_p = np.vstack((v11, v12, r13))
    return r_p, v_p


# TODO njit this
def keplerian_to_inertial_3d(state: np.ndarray, gm: float = gm, mean_or_true: str = 'true') -> np.ndarray:
    """
    Convert a 3D state vector from Keplerian to inertial.
    :param state:
    :param gm:
    :param mean_or_true:
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
    return np.hstack((r_i, v_i))  # TODO speed up


# TODO njit this
def keplerian_to_perifocal_2d(state: np.ndarray, gm: float = gm, mean_or_true: str = 'true') -> np.ndarray:
    """
    Convert a 2D state vector from Keplerian to perifocal
    :param state:
    :param gm:
    :param mean_or_true:
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


# TODO njit this
def keplerian_to_inertial_2d(state: np.ndarray, gm: float = gm, mean_or_true: str = 'true') -> np.ndarray:
    """
    Convert a 2D state vector from Keplerian to inertial
    :param state:
    :param gm:
    :param mean_or_true:
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
    assert abs(i - np.pi / 2) <= np.pi / 2, "inclination is outside of acceptable bounds"  # 0 <= i <= np.pi
    w = fix_angle(w, upper_bound=2*np.pi, lower_bound=0.)
    om = fix_angle(om, upper_bound=2*np.pi, lower_bound=0.)
    ta = fix_angle(ta, upper_bound=2*np.pi, lower_bound=0.)
    # Convert to MEE
    p = a * (1 - e ** 2)
    f = e * cos(w + om)
    g = e * sin(w + om)
    h = np.tan(i / 2) * cos(om)
    k = np.tan(i / 2) * sin(om)
    el = om + w + ta
    return np.array([p, f, g, h, k, el])


@njit
def mee_to_keplerian_3d(state: np.ndarray) -> np.ndarray:
    """
    Convert a 3D state vector from Modified Equinoctial Elements to Keplerian
    :param state:
    :return:
    """
    p, f, g, h, k, el = state
    # Convert to Keplerian
    om = np.arctan2(k, h)
    i = 2 * np.arctan(np.sqrt(h ** 2 + k ** 2))
    w = np.arctan2(g, f) - om
    e = np.sqrt(g ** 2 + f ** 2)
    a = p / (1 - g ** 2 - f ** 2)
    v = el - om - w
    # Make sure angles are properly scaled
    w = fix_angle(w, 2 * np.pi, 0.)
    om = fix_angle(w, 2 * np.pi, 0.)
    v = fix_angle(v, 2 * np.pi, 0.)
    return np.array([a, e, i, w, om, v])


@njit
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
    # dcm = np.vstack((v_hat, n_hat, c_hat)).T  # direction cosine matrix
    dcm = np.vstack((c_hat, v_hat, n_hat)).T  # direction cosine matrix
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
        if upper_bound >= angle >= lower_bound:
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
    dcm = np.array([[np.cos(np.pi / 2 - fpa), -np.sin(np.pi / 2 - fpa)], [np.sin(np.pi / 2 - fpa),
                                                                          np.cos(np.pi / 2 - fpa)]])
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
    eps = 1e-6
    if psi > eps:
        res = (1 - math.cos(psi ** 0.5)) / psi
    elif psi < -eps:
        # res = (1 - math.cosh((-psi) ** 0.5)) / psi
        res = -math.exp((-psi) ** 0.5) / 2 / psi
    else:
        res = 0.5
        # delta = (-psi) / gamma(2 + 2 + 1)  # TODO what is this crap?
        # k = 1
        # while res + delta != res:
        #     res = res + delta
        #     k += 1
        #     delta = (-psi) ** k / gamma(2 * k + 2 + 1)
    return res


@njit
def c3(psi: float) -> float:
    """
    Helper function for vallado().
    :param psi:
    :return:
    """
    eps = 1e-6
    if psi > eps:
        res = (psi ** 0.5 - math.sin(psi ** 0.5)) / (psi * psi ** 0.5)
    elif psi < -eps:
        # res = (math.sinh((-psi) ** 0.5) - (-psi) ** 0.5) / (-psi * (-psi) ** 0.5)
        res = math.exp((-psi) ** 0.5) / 2. / (-psi ** 3) ** 0.5
    else:
        res = 1. / 6.
        # delta = (-psi) / gamma(2 + 3 + 1)  # TODO what is this crap?
        # k = 1
        # while res + delta != res:
        #     res = res + delta
        #     k += 1
        #     delta = (-psi) ** k / gamma(2 * k + 3 + 1)
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

    A = t_m * (norm_r0_times_norm_r * (1 + cos_dnu)) ** 0.5
    y = 0.0

    if A == 0.0:
        raise RuntimeError("Cannot compute orbit, phase angle is 180 degrees")

    psi = 0.0
    psi_low = -4 * np.pi ** 2
    psi_up = 4 * np.pi ** 2
    count = 0
    psi_low_increment = 1e0  # TODO this counter has a large effect on overall runtime if too small (!!!!)

    while count < numiter:
        y = norm_r0_plus_norm_r + A * (psi * c3(psi) - 1) / c2(psi) ** 0.5
        if A > 0.0:
            # Readjust psi_low until y > 0.0
            while y < 0.0:  # TODO implement a shooting method here for speed
                psi_low += psi_low_increment
                psi = psi_low
                assert psi_low <= psi <= psi_up, "psi exceeds bounds"
                y = norm_r0_plus_norm_r + A * (psi * c3(psi) - 1) / c2(psi) ** 0.5

        xi = (y / c2(psi)) ** 0.5
        tof_new = (xi ** 3 * c3(psi) + A * y ** 0.5) / k ** 0.5

        # Convergence check
        if np.abs((tof_new - tof) / tof) < rtol:
            break
        count += 1

        # Bisection check
        condition = tof_new <= tof
        if condition:
            psi_low = psi
        else:
            psi_up = psi
        psi = (psi_up + psi_low) / 2
    else:
        # raise RuntimeError("Maximum number of iterations reached")
        # warnings.warn('Maximum number of iterations reached in vallado')
        pass
    # print('count = %i | tof = %.4f' % (count, tof / 86400))

    f = 1 - y / norm_r0
    g = A * (y / k) ** 0.5

    gdot = 1 - y / norm_r

    v0 = (r - f * r0) / g
    v = (gdot * r - r0) / g

    return v0, v


def find_min(f, low: float, high: float, num_iter: int = 10):
    assert high > low
    # Initialize guesses at [0, 1/3, 2/3, 1]
    coords = np.empty((2, 4))
    coords[0, :] = [low, (high - low) / 3 + low, (high - low) * 2 / 3 + low, high]
    _min = np.infty
    # Evaluate initial guesses
    for i in range(4):
        coords[1, i] = f(coords[0, i])
    # Main loop
    for i in range(num_iter):
        # Identify best solution and its index
        _min = min(coords[1])
        ind_min = np.argmin(coords[1])
        # Adjust search region based on new min
        low = coords[0, max(ind_min - 1, 0)]
        high = coords[0, min(ind_min + 1, coords.shape[1] - 1)]
        x_min = coords[0, ind_min]
        new_coord_low = (x_min - low) / 2 + low
        new_coord_high = (high - x_min) / 2 + x_min
        # Evaluate the lower of the two new points
        new_pt_low = f(new_coord_low)
        coords = np.insert(coords, ind_min, [new_coord_low, new_pt_low], axis=1)
        # If lower point is better, can skip to next iteration; otherwise evaluate higher point
        if new_pt_low < _min:
            continue
        else:
            new_pt_high = f(new_coord_high)
            coords = np.insert(coords, ind_min + 2, [new_coord_high, new_pt_high], axis=1)
    # Record best ever
    min_x, min_y = coords[0, np.argmin(coords[1])], min(coords[1])
    return min_x, min_y


def _dv_from_tof(tof, t0, targ_planet, gm, r0, v0, short, numiter, rtol):
    time = tof * c.day_to_jc + t0
    state_f = c.ephem(['a', 'e', 'i', 'w', 'O', 'M'], [targ_planet], np.array([time]))
    state_f[2] = 0.  # make coplanar
    state_f = keplerian_to_inertial_3d(state_f, gm, 'mean')
    # sol = vallado(gm, r0, state_f[:3], tof * c.day_to_sec, short=short, numiter=numiter, rtol=rtol)
    sol = lambert(float(gm), r0.tolist(), state_f[:3].tolist(), float(tof * c.day_to_sec), short)
    if sol[2] == 0:
        # return mag3(sol[0] - v0) + mag3(state_f[3:6] - sol[1])
        return mag3(sol[0] - v0) + hyperbolic_capture_from_infinity(mag3(state_f[3:6] - sol[1]),
                                     tc.capture_periapsis_radius_km, tc.capture_period_day * c.day_to_sec, tc.gm_target)
    else:
        return np.infty


# TODO Do I really need to find the min dv TOF, or can I just set it to e.g. 60 days and have the network learn around
#      that? Because for most of the early and mid stages of training, it will be at the max bound that I set which
#      will probably be in the ballpark of 60 days.
def lambert_min_dv(gm: float, state_0: np.ndarray, t0: float, low: float, high: float, max_count: int = 10,
                   targ_planet: str = tc.target_body, short: bool = True) -> (np.ndarray, np.ndarray, float):
    """
    Computes the minimum delta V transfer between two states assuming a two-impulse maneuver.
    :param max_count:
    :param gm:
    :param state_0:
    :param t0:
    :param low:
    :param high:
    :param targ_planet:
    :param short:
    :return:
    """
    # Define/initialize parameters for search
    r0, v0 = state_0[:3], state_0[3:6]

    # Fill in all inputs except TOF
    t0_jc = tc.times_jd1950_jc[-1]

    def f_short(tof):
        return _dv_from_tof(tof, t0_jc, targ_planet, gm, r0, v0, True, tc.vallado_numiter, tc.vallado_rtol)

    # def f_long(tof):
    #     return _dv_from_tof(tof, t0_jc, targ_planet, gm, r0, v0, False, tc.vallado_numiter, tc.vallado_rtol)

    # Find minimum dv TOF
    tof_of_min_dv_s, min_dv_s = find_min(f_short, low, high, num_iter=max_count)
    # tof_of_min_dv_l, min_dv_l = find_min(f_long, low, high, num_iter=max_count)
    # print('TOF short = %f\t|\tmin_dv_s = %e' % (tof_of_min_dv_s, min_dv_s))
    # print('TOF long = %f\t|\tmin_dv_l = %e\n' % (tof_of_min_dv_l, min_dv_l))

    # if min_dv_s < min_dv_l:
    tof_of_min_dv = tof_of_min_dv_s
    short = True
    # else:
    #     tof_of_min_dv = tof_of_min_dv_l
    #     short = False

    # Recompute best case to get dv vectors
    tf_jc = tof_of_min_dv * c.day_to_jc + t0_jc
    state_f = c.ephem(['a', 'e', 'i', 'w', 'O', 'M'], [targ_planet], np.array([tf_jc]))
    state_f[2] = 0.
    state_f = keplerian_to_inertial_3d(state_f, gm, 'mean')
    sol = lambert(float(gm), r0.tolist(), state_f[:3].tolist(), float(tof_of_min_dv * c.day_to_sec), short)
    dv1 = sol[0] - v0
    dv2 = state_f[3:6] - sol[1]
    dv1[np.isnan(dv1) + np.isinf(dv1)] = 1e6
    dv2[np.isnan(dv2) + np.isinf(dv2)] = 1e6
    # dv2 = hyperbolic_capture_from_infinity(state_f[3:6] - sol[1], tc.capture_periapsis_radius_km, tc.capture_period_day, gm,
    #                                        tc.capture_low_not_high)
    return dv1, dv2, tof_of_min_dv


def _lambert_min_dv(k: float, state_0: np.ndarray, target_planet: str = 'mars', short: bool = True,
                    do_print: bool = False) -> (np.ndarray, np.ndarray, float):
    """
    Computes the minimum delta V transfer between two states assuming a two-impulse maneuver.
    :param k:
    :param state_0:
    :param target_planet:
    :param short:
    :param do_print:
    :return:
    """
    r0, v0 = state_0[:3], state_0[3:6]
    # rf, vf = state_f[:3], state_f[3:6]
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

    # Compute corresponding final state at these times
    times = (tof * c.sec_to_day + c.reference_date_jd1950) * c.day_to_jc
    states_f = c.ephem(['a', 'e', 'i', 'w', 'O', 'M'], [target_planet], times)
    states_f = keplerian_to_inertial_3d(states_f, k, 'mean')

    dv = list()
    for _tof, _state_f in zip(tof, states_f):
        try:
            sol = vallado(k, r0, _state_f[:3], _tof, short=short, numiter=numiter, rtol=rtol)
        except RuntimeError:
            sol = ([10, 10, 10], [10, 10, 10])
        dv.append(np.linalg.norm((sol[0] - v0)) + np.linalg.norm((sol[1] - _state_f[3:6])))

    # Adjust bounds and start direction of search
    dv = np.array(dv)
    if dv[0] > dv[1] > dv[2]:
        if dv[2] > dv[3]:
            low = tof[2]
        else:
            low = tof[1]
    elif dv[1] < dv[2] < dv[3]:
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
        time = (tof * c.sec_to_day + c.reference_date_jd1950) * c.day_to_jc
        state_f = c.ephem(['a', 'e', 'i', 'w', 'O', 'M'], [target_planet], time)
        state_f = keplerian_to_inertial_3d(state_f, k, 'mean')
        rf, vf = state_f[:3], state_f[3:6]
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
            elif dir_last > 0 > dir_best:
                low = last_tof
            elif dir_last < 0 < dir_best:
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


@vectorize([float64(float64)])
def fast_square(elem):
    """
    A fast method to square each element of a matrix. NOTE: This is slower when used on a (zero-D) float.
    :param elem:
    :return:
    """
    return elem * elem


@vectorize([float64(float64)])
def fast_abs(elem):
    """
    A fast method to take the absolute value of each element of a matrix. NOTE: This is slower when used on a float.
    :param elem:
    :return:
    """
    return abs(elem)


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
    :return:
    """
    # TODO streamline this
    if r_vec.size == 2:
        a, e, w, f = inertial_to_keplerian_2d(np.hstack((r_vec, v_vec)), gm=gm)
    else:
        a, e, i, w, om, f = inertial_to_keplerian_3d(np.hstack((r_vec, v_vec)), gm=gm)

    if 0.99 <= e <= 1.01:  # parabolic
        if 0 < f < np.pi:  # ascending and escaping
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
        if 0 < f < np.pi:  # ascending and escaping
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
            # TODO don't need to completely reverse direction - this will send you around the other side of the planet
            dv1_mag -= v_mag_capture_kms
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
    a, e, i, w, om, f = inertial_to_keplerian_3d(state_sc[:6] - state_target, gm=gm)
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
    :param gm:
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
    # Flip relative direction if the new body is 'sun'
    # TODO but this was meant for cartesian -> how does this work for keplerian?
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


# TODO can this be njit'd?
def euler313(vector: np.ndarray, psi: float, theta: float, phi: float) -> np.ndarray:
    dcm = np.array([[cos(psi) * cos(phi) - sin(psi) * sin(phi) * cos(theta),
                     cos(psi) * sin(phi) + sin(psi) * cos(theta) * cos(phi),
                     sin(psi) * sin(theta)],
                    [-sin(psi) * cos(phi) - cos(psi) * sin(phi) * cos(theta),
                     -sin(psi) * sin(phi) + cos(psi) * cos(theta) * cos(phi),
                     cos(psi) * sin(theta)],
                    [sin(theta) * sin(phi),
                     -sin(theta) * cos(phi),
                     cos(theta)]]).T
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
    return np.arccos(max(min(h / r / v, 1), -1)) * sign_check(r_vec, v_vec)


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
    frac = M / n
    half_per = per / 2
    return (half_per - frac) if half_per > frac else (3 * half_per - frac)


def time_to_periapsis_from_per_M_n(per: float, M: float, n: float) -> float:
    return per - M / n


def H_from_f_e(f: float, e: float) -> float:
    return np.arctanh(sin(f) * (e * e - 1) ** 0.5 / (e + cos(f)))


def inc_from_h(h_vec: np.array) -> float:
    return np.arccos(h_vec[2] / mag3(h_vec))


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


@njit
def v_from_gm_r_rp_gamma(gm: float, r: float, rp: float, gamma: float) -> float:
    assert r > rp, "Radius is smaller than the target periapsis."
    return (2 * gm * rp * (1 - rp / r) / (r * r * cos(gamma) ** 2 - rp * rp)) ** 0.5


@njit
def v_from_gm_r_ra_gamma(gm: float, r: float, ra: float, gamma: float) -> float:
    assert r < ra, "Radius is greater than the target apoapsis."
    return (2 * gm * ra * (1 - ra / r) / (r * r * cos(gamma) ** 2 - ra * ra)) ** 0.5


# TODO
def capture(state_rel: np.ndarray, rp_target: float, per_target: float, gm: float, r_soi: float, capture_low: bool,
            current: bool) -> list:
    """
    Calls the appropriate function to capture a spacecraft around the target body based on the spacecraft's current
    orbit. Decisions include starting from hyperbolic or elliptical orbit, targeting a low circular or high elliptical
    final orbit, starting inside or outside the sphere of influence, and whether the maneuver could
    occur at the optimal point of the current orbit or force it to be at the current location.
    :param state_rel:
    :param rp_target:
    :param per_target:
    :param gm:
    :param r_soi:
    :param capture_low:
    :param current:
    :return:
    """
    r_mag, v_mag = mag3(state_rel[:3]), mag3(state_rel[3:])
    energy = v_mag * v_mag / 2 - gm / r_mag
    char1 = 'h' if energy > 0 else 'e'
    char2 = 'o' if r_mag > r_soi else 'i'
    char3 = 'l' if capture_low else 'h'
    char4 = 'c' if current else 'o'
    if char1 == 'e' and char2 == 'o':
        char1 = 'h'
        warnings.warn('Flipped char1 in capture()')
    capture_type = char1 + char2 + char3 + char4
    print(capture_type)
    capture_methods = {'holc': _hyperbolic_out_low_current,  'holo': _hyperbolic_out_low_optimal,
                       'hilc': _hyperbolic_in_low_current,   'hilo': _hyperbolic_in_low_optimal,
                       'hohc': _hyperbolic_out_high_current, 'hoho': _hyperbolic_out_high_optimal,
                       'hihc': _hyperbolic_in_high_current,  'hiho': _hyperbolic_in_high_optimal,
                       # 'eolc': _elliptical_out_low_current,  'eolo': _elliptical_out_low_optimal,
                       'eilc': _elliptical_in_low_current,   'eilo': _elliptical_in_low_optimal,
                       # 'eohc': _elliptical_out_high_current, 'eoho': _elliptical_out_high_optimal,
                       'eihc': _elliptical_in_high_current,  'eiho': _elliptical_in_high_optimal}
    maneuvers = capture_methods[capture_type](state_rel, rp_target, per_target, gm)
    return maneuvers


def get_capture_final_values(maneuvers: list, m0: float) -> (float, float):
    """
    Takes the maneuvers from capture() and computes the final mass and time after the capture. Does not compute states.
    :param maneuvers:
    :param m0:
    :return:
    """
    # maneuvers, *args = maneuvers
    # if len(args) == 2 or len(args) == 4:
    #     maneuvers = [maneuvers, *args]
    # elif len(args) != 1:
    #     raise ValueError('Unknown or unprepared outputs from capture function.')

    if len(maneuvers) == 3:
        dv1_vec, dv2_vec, tof12 = maneuvers
        dv3_vec, tof23 = np.zeros(3), 0.
        dv1_mag, dv2_mag, dv3_mag = mag3(dv1_vec), mag3(dv2_vec), 0.
    else:
        dv1_vec, dv2_vec, dv3_vec, tof12, tof23 = maneuvers
        dv1_mag, dv2_mag, dv3_mag = mag3(dv1_vec), mag3(dv2_vec), mag3(dv3_vec)

    m1 = m0 / np.exp(dv1_mag * 1000 / c.g0_ms2 / tc.isp_chemical)
    m2 = m1 / np.exp(dv2_mag * 1000 / c.g0_ms2 / tc.isp_chemical)
    m3 = m2 / np.exp(dv3_mag * 1000 / c.g0_ms2 / tc.isp_chemical)

    tf = tof12 + tof23

    return tf, m3


# TODO account for tof01, which is the time to go from current location to the initial maneuver
def propagate_capture(maneuvers: list, state_0: np.ndarray, m0: float, du: float = 20000., gm: float = tc.gm)\
        -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Takes the maneuvers from capture() and generates the corresponding state history. state_0 and m0 are the state
    (position and velocity only) and the mass at the end of trajectory propagation.
    :param maneuvers:
    :param state_0:
    :param m0:
    :param du:
    :param gm:
    :return:
    """
    maneuvers, *args = maneuvers
    if len(args) == 1:
        state_opt = args[0]
        state_0 = state_opt
    elif len(args) == 2 or len(args) == 4:
        maneuvers = [maneuvers, *args]
    else:
        raise ValueError('Unknown or unprepared outputs from capture function.')

    if len(maneuvers) == 3:
        dv1_vec, dv2_vec, tof12 = maneuvers
        dv3_vec, tof23 = np.zeros(3), 0.
        dv1_mag, dv2_mag, dv3_mag = mag3(dv1_vec), mag3(dv2_vec), 0.
        n_maneuvers = 2
    else:
        dv1_vec, dv2_vec, dv3_vec, tof12, tof23 = maneuvers
        dv1_mag, dv2_mag, dv3_mag = mag3(dv1_vec), mag3(dv2_vec), mag3(dv3_vec)
        n_maneuvers = 3
    tof12 *= c.day_to_sec
    y = np.empty((n_maneuvers, 7))
    full_traj = np.empty(((n_maneuvers - 1) * tc.n_terminal_steps + 1, 8))
    # ti = np.empty(n_maneuvers + 1)

    # Compute masses after each maneuver
    m1 = m0 / np.exp(dv1_mag * 1000 / c.g0_ms2 / tc.isp_chemical)
    m2 = m1 / np.exp(dv2_mag * 1000 / c.g0_ms2 / tc.isp_chemical)
    m3 = m2 / np.exp(dv3_mag * 1000 / c.g0_ms2 / tc.isp_chemical)

    step_type, eom_type = 0, 3  # fixed step, TBP only

    state_1, state_2, state_3 = np.empty((3, 6))

    # Before and after initial maneuver on "original" orbit
    # traj01m = tbp.prop(list(state_0 / state_scales), [0., -tf / tu], [], 6, 2, 0, tol, tol, step_size / tu, step_type,
    #                  eom_type)
    # traj01m = (np.array(traj01m)[:, 1:] * state_scales).T
    # traj01p = tbp.prop(list(state_0 / state_scales), [0.,  tf / tu], [], 6, 2, 0, tol, tol, step_size / tu, step_type,
    #                   eom_type)
    # traj01p = (np.array(traj01p)[:, 1:] * state_scales).T

    # Transfer from initial location to 2nd maneuver location
    state_1[:3], state_1[3:6] = state_0[:3], state_0[3:6] + dv1_vec
    traj12 = tbp.prop(list(state_1 / tc.state_scales[:-1]), [0., tof12 / tc.tu], [], 6, 2, 0, tc.rtol, tc.atol,
                      tof12 / (tc.n_terminal_steps - 1) / tc.tu, step_type, eom_type)
    traj12 = np.array(traj12)

    # Compute state after 2nd maneuver
    state_2[:3], state_2[3:6] = traj12[-1, 1:4] * tc.state_scales[:3], traj12[-1, 4:7] * tc.state_scales[3:6] + dv2_vec

    # Compute propagate until 3rd maneuver if necessary, and save states
    if n_maneuvers == 3:
        traj23 = tbp.prop(list(state_2 / tc.state_scales[:-1]), [0., tof23 / tc.tu], [], 6, 2, 0, tc.rtol, tc.atol,
                          tof23 / (tc.n_terminal_steps - 1) / tc.tu, step_type, eom_type)
        traj23 = np.array(traj23)
        traj23[:, 0] += tof12 / tc.tu

        state_3[:3] = traj23[-1, 1:4] * tc.state_scales[:3]
        state_3[3:6] = traj23[-1, 4:7] * tc.state_scales[3:6] + dv3_vec
        y[:, :-1] = state_1, state_2, state_3
        y[:, -1] = m1, m2, m3
        full_traj[:-1, :-1] = traj12, traj23
        full_traj[:tc.n_terminal_steps, -1] = m1 / tc.mu
        full_traj[tc.n_terminal_steps:-1, -1] = m2 / tc.mu
        full_traj[-1, :-1] = traj23[-1]
        full_traj[-1, 4:7] += dv3_vec / tc.state_scales[3:6]
        full_traj[-1, -1] = m3 / tc.mu
        # ti[:] = 0, tof12, tof23

    else:
        y[:, :-1] = state_1, state_2
        y[:, -1] = m1, m2
        full_traj[:-1, :-1] = traj12
        full_traj[:-1, -1] = m1 / tc.mu
        full_traj[-1, :-1] = traj12[-1]
        full_traj[-1, 4:7] += dv2_vec / tc.state_scales[3:6]
        full_traj[-1, -1] = m2 / tc.mu
        # ti[:] = 0, tof12

    return y, full_traj


# Finished - but I may want to change first maneuver # Tested 4
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
    h_vec = cross(r_vec, v_vec)
    h_mag = mag3(h_vec)
    gamma = gamma_from_h_r_v_qcheck(h_mag, r_mag, v_mag, r_vec, v_vec)
    n = 2 * (gm / p ** 3) ** 0.5
    B = (2 * r_mag / p - 1) ** 0.5
    tof = (B ** 3 / 3 + + B) / n  # time to periapsis
    f = abs(np.arctan2(p * B, p - r_mag)) * -1.  # make sure we are descending
    gamma_para = f / 2
    d_gamma = gamma_para - gamma
    # Use law of cosines to determine dv1_mag and angle of dv1 in VNC coords (alpha)
    dv1_mag = (v_mag ** 2 + v_mag_para ** 2 - 2 * v_mag * v_mag_para * cos(abs(d_gamma))) ** 0.5
    beta = np.arccos((v_mag ** 2 + dv1_mag ** 2 - v_mag_para ** 2) / 2 / v_mag / dv1_mag)
    alpha = np.pi - beta
    dv1_vec = np.array([cos(alpha), 0, -sin(alpha)]) * dv1_mag  # TODO how should dv1_vec be changed for 3D?
    dv1_vec = rotate_vnc_to_inertial_3d(dv1_vec, state_0)
    v_transfer_vec = v_vec + dv1_vec
    # Compute delta v to get into target orbit at periapsis
    v_final = (gm / rp_target) ** 0.5
    dv2_mag = v_final - vp_para
    # Compute delta v vector and final velocity vector
    v_periapsis_vec = v_transfer_vec / v_mag_para * vp_para
    df = (0 - f)
    dgamma = (0 - gamma_para)
    # Account for sign difference if orbit is "backwards"
    i = inc_from_h(h_vec)
    if i > np.pi / 2 or i < - np.pi / 2:
        df *= -1
        dgamma *= -1
    v_periapsis_vec = euler313(v_periapsis_vec, 0, 0, df - dgamma)
    dv2_vec = v_periapsis_vec / vp_para * dv2_mag
    maneuvers = [dv1_vec, dv2_vec, tof]
    return maneuvers


# Finished - there *may* be a way to optimally adjust energy and orientation instead of just orientation # Tested 4
def _hyperbolic_out_low_optimal(state_0: np.ndarray, rp_target: float, per_target: float, gm: float)\
        -> (list, np.ndarray):
    """
    :param state_0:
    :return:
    """
    # Define initial orbit in Keplerian elements - then determine Cartesian state
    state_kep = inertial_to_keplerian_3d(state_0, gm)
    # Then determine f when r = r_far_away
    r_far = tc.r_limit_soi * c.r_soi_mars
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


# Finished - but I may want to change first maneuver # Tested 4
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
    h_vec = cross(r_vec, v_vec)
    h_mag = mag3(h_vec)
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
    dv1_vec = np.array([cos(alpha), 0, -sin(alpha)]) * dv1_mag  # TODO how should dv1_vec be changed for 3D?
    dv1_vec = rotate_vnc_to_inertial_3d(dv1_vec, state_0)
    v_transfer_vec = v_vec + dv1_vec
    # Compute delta v to get into target orbit at periapsis
    ra_target = ra_from_rp_per(rp_target, per_target, gm)
    a_target = (ra_target + rp_target) / 2
    v_final = v_from_gm_r_a(gm, rp_target, a_target)
    dv2_mag = v_final - vp_para
    # Compute delta v vector and final velocity vector
    v_periapsis_vec = v_transfer_vec / v_mag_para * vp_para
    df = (0 - f)
    dgamma = (0 - gamma_para)
    # Account for sign difference if orbit is "backwards"
    i = inc_from_h(h_vec)
    if i > np.pi / 2 or i < - np.pi / 2:
        df *= -1
        dgamma *= -1
    v_periapsis_vec = euler313(v_periapsis_vec, 0, 0, df - dgamma)
    dv2_vec = v_periapsis_vec / vp_para * dv2_mag
    maneuvers = [dv1_vec, dv2_vec, tof]
    return maneuvers


# Finished - there *may* be a way to optimally adjust energy and orientation instead of just orientation # Tested 4
def _hyperbolic_out_high_optimal(state_0: np.ndarray, rp_target: float, per_target: float, gm: float)\
        -> (list, np.ndarray):
    """
    :param state_0:
    :return:
    """
    # Define initial orbit in Keplerian elements - then determine Cartesian state
    state_kep = inertial_to_keplerian_3d(state_0, gm)
    # Then determine f when r = r_far_away
    r_far = tc.r_limit_soi * c.r_soi_mars
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


# Finished # Tested 4 # TODO broken when outside of ra_target
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
    v1_mag = v_from_gm_r_ra_gamma(gm, r0_mag, ra_target, gamma)
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
        v1_mag = v_from_gm_r_rp_gamma(gm, r0_mag, rp_target, gamma)
        dv1_mag = v1_mag - v0_mag
        dv1_vec = v0_vec / v0_mag * dv1_mag
        v1_vec = v0_vec + dv1_vec

        # At periapsis, circularize
        energy = energy_from_gm_r_v(gm, r0_mag, v1_mag)
        a = a_from_gm_energy(gm, energy)
        e_vec = e_vec_from_gm_v_r(gm, v1_vec, r0_vec)
        e_mag = mag3(e_vec)
        ra = ra_from_a_e(a, e_mag)
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
        per = per_from_gm_a(gm, a)
        tof12 = time_to_periapsis_from_per_M_n(per, M, n)

    if at_periapsis:
        maneuvers = [dv1_vec, dv2_vec, tof12]
    else:
        a = a_from_rp_ra(rp_target, ra)
        vp_transfer = v_from_gm_r_a(gm, rp_target, a)
        dv3_mag = (gm / rp_target) ** 0.5 - vp_transfer
        dv3_vec = dv2_vec / dv2_mag * -dv3_mag
        per = per_from_gm_a(gm, a)
        tof23 = per / 2
        maneuvers = [dv1_vec, dv2_vec, dv3_vec, tof12, tof23]

    return maneuvers


# Finished # Tested 4  # TODO possibly broken when outside ra_target
def _hyperbolic_in_low_optimal(state_0: np.ndarray, rp_target: float, per_target: float, gm: float)\
        -> (list, np.ndarray):
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


# Finished # Tested 4 # TODO probably broken when outside ra_target
def _hyperbolic_in_high_current(state_0: np.ndarray, rp_target: float, per_target: float, gm: float) -> list:
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
        v1_mag = ((2 * gm * rp_target * (1 - rp_target / r0_mag)) /
                  (r0_mag ** 2 * cos(gamma) ** 2 - rp_target ** 2)) ** 0.5
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
        per = per_from_gm_a(gm, a)
        tof = time_to_periapsis_from_per_M_n(per, M, n)
    maneuvers = [dv1_vec, dv2_vec, tof]
    return maneuvers


# Finished # Tested 4 # TODO possibly broken when outside ra_target
def _hyperbolic_in_high_optimal(state_0: np.ndarray, rp_target: float, per_target: float, gm: float)\
        -> (list, np.ndarray):
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


# Finished # Tested 12 (normal + 2 types of ascending problems)
def _elliptical_in_low_current(state_0: np.ndarray, rp_target: float, per_target: float, gm: float) -> list:
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
    a, e, i, w, om, f = inertial_to_keplerian_3d(state_0, gm)
    assert e < 1, 'Initial orbit is hyperbolic'
    ra = ra_from_a_e(a, e)
    rp = rp_from_a_e(a, e)
    tmp_state = np.empty(6)

    # Adjust periapsis at current location
    v1_mag = v_from_gm_r_rp_gamma(gm, r0_mag, rp_target, gamma)
    if not np.isnan(v1_mag):
        dv1_mag = v1_mag - v0_mag
        dv1_vec = v0_vec / v0_mag * dv1_mag
        v1_vec = v0_vec + dv1_vec
    # else s/c is coming in too sharply - the periapsis can't be raised while maintaining constant flight path angle
    else:
        a, e = a_e_from_rp_ra(rp_target, ra)
        v1_mag = v_from_gm_r_a(gm, r0_mag, a)
        h = h_from_gm_a_e(gm, a, e)
        gamma0 = gamma
        gamma = gamma_from_h_r_v(h, r0_mag, v1_mag) * -1
        v1_vec = v0_vec / v0_mag * v1_mag
        dgamma = gamma - gamma0
        if inc_from_h(h0_vec) > np.pi / 2:
            dgamma *= -1
        v1_vec = euler313(v1_vec, 0, 0, -dgamma)
        dv1_vec = v1_vec - v0_vec

    # At periapsis, adjust apoapsis
    energy = energy_from_gm_r_v(gm, r0_mag, v1_mag)
    if energy > 0 and gamma > 0:
        # Ascending: maneuvers will be to (attempt to) lower apoapsis at current point
        #   and then lower periapsis once reaching apoapsis
        if r0_mag < ra_target:
            # Conserve r_vec and gamma; only change v_mag such that ra becomes ra_target
            v1_mag = v_from_gm_r_ra_gamma(gm, r0_mag, ra_target, gamma)
            dv1_mag = v1_mag - v0_mag
            dv1_vec = v0_vec / v0_mag * dv1_mag
            v1_vec = v0_vec + dv1_vec

            # At apoapsis, change periapsis to target altitude
            energy = energy_from_gm_r_v(gm, r0_mag, v1_mag)
            a = a_from_gm_energy(gm, energy)
            dv2_mag = lower_periapsis(ra_target, a, rp_target, gm)
            tmp_state[:3], tmp_state[3:] = r0_vec, v1_vec
            a, e, i, w, om, f = inertial_to_keplerian_3d(tmp_state, gm)
            df1 = np.pi - f
            dgamma1 = 0 - gamma
            to_apoapsis = True

            # Check signs
            if i > np.pi / 2:
                df1 *= -1
                dgamma1 *= -1

            # Rotate dv2 vector into correct orientation
            dv2_vec = v1_vec / v1_mag * dv2_mag
            dv2_vec = euler313(dv2_vec, 0, 0, df1 - dgamma1)

            # Compute time between maneuvers
            E = E_from_f_e(f, e)
            E = fix_angle(E, 2 * np.pi, 0)
            M = M_from_E_e(E, e)
            per = per_from_gm_a(gm, a)
            n = n_from_per(per)
            if to_apoapsis:
                tof12 = time_to_apoapsis_from_per_M_n(per, M, n)
            else:
                tof12 = time_to_periapsis_from_per_M_n(per, M, n)

            dv3_mag = lower_apoapsis(rp_target, a, rp_target, gm)
            a, e = a_e_from_rp_ra(rp_target, ra_target)
            f = np.pi
            df2 = 2 * np.pi - f
            dgamma2 = 0
            to_apoapsis = False

            # Check signs
            if i > np.pi / 2:
                df2 *= -1
                dgamma2 *= -1

            # Rotate dv2 vector into correct orientation
            dv3_vec = v1_vec / v1_mag * dv3_mag
            dv3_vec = euler313(dv3_vec, 0, 0, df1 + df2 - dgamma1 - dgamma2)

            # Compute time between maneuvers
            E = E_from_f_e(f, e)
            E = fix_angle(E, 2 * np.pi, 0)
            M = M_from_E_e(E, e)
            per = per_from_gm_a(gm, a)
            n = n_from_per(per)
            if to_apoapsis:
                tof23 = time_to_apoapsis_from_per_M_n(per, M, n)
            else:
                tof23 = time_to_periapsis_from_per_M_n(per, M, n)

            maneuvers = [dv1_vec, dv2_vec, dv3_vec, tof12, tof23]
            return maneuvers

        # If we're already super far away and ascending and would be hyperbolic, just gotta pull the e-brake and set
        # current location as new apoapsis
        else:
            a = a_from_rp_ra(rp_target, r0_mag)
            v1_mag = v_from_gm_r_a(gm, r0_mag, a)
            v1_vec = r0_vec.copy() / r0_mag * v1_mag
            v1_vec = euler313(v1_vec, 0, 0, np.pi / 2)
            dv1_vec = v1_vec - v0_vec

            # At periapsis, circularize
            dv2_mag = lower_apoapsis(rp_target, a, rp_target, gm)
            tmp_state[:3], tmp_state[3:] = r0_vec, v1_vec
            i = inc_from_h(cross(r0_vec, v0_vec))
            df1 = np.pi
            dgamma1 = 0
            to_apoapsis = False

            # Check signs
            if i > np.pi / 2:
                df1 *= -1
                dgamma1 *= -1

            # Rotate dv2 vector into correct orientation
            dv2_vec = v1_vec / v1_mag * dv2_mag
            dv2_vec = euler313(dv2_vec, 0, 0, df1 - dgamma1)

            # Compute time between maneuvers
            e = e_from_rp_ra(rp_target, ra_target)
            E = E_from_f_e(np.pi, e)
            E = fix_angle(E, 2 * np.pi, 0)
            M = M_from_E_e(E, e)
            per = per_from_gm_a(gm, a)
            n = n_from_per(per)
            if to_apoapsis:
                tof12 = time_to_apoapsis_from_per_M_n(per, M, n)
            else:
                tof12 = time_to_periapsis_from_per_M_n(per, M, n)

            maneuvers = [dv1_vec, dv2_vec, tof12]
            return maneuvers

    # Descending, or above target apoapsis: maneuvers will be to set periapsis to correct altitude at current location,
    #     and then lower apoapsis to target altitude once reaching periapsis
    else:

        a = a_from_gm_energy(gm, energy)
        dv2_mag = lower_apoapsis(rp_target, a, rp_target, gm)
        tmp_state[:3], tmp_state[3:] = r0_vec, v1_vec
        a, e, i, w, om, f = inertial_to_keplerian_3d(tmp_state, gm)
        # assert e < 1, 'Transfer orbit is hyperbolic'
        ra = ra_from_a_e(a, e)
        rp = rp_from_a_e(a, e)

        f = fix_angle(f, 2 * np.pi, 0)
        df = 2 * np.pi - f
        dgamma = 0 - gamma
        # to_apoapsis = False

        # Check signs
        if i > np.pi / 2:
            df *= -1
            dgamma *= -1

        # Rotate dv2 vector into correct orientation
        dv2_vec = v1_vec / v1_mag * dv2_mag
        dv2_vec = euler313(dv2_vec, 0, 0, df - dgamma)

        # Compute time between maneuvers
        E = E_from_f_e(f, e)
        E = fix_angle(E, 2 * np.pi, 0)
        M = M_from_E_e(E, e)
        per = per_from_gm_a(gm, a)
        n = n_from_per(per)
        # if to_apoapsis:
        #     tof = time_to_apoapsis_from_per_M_n(per, M, n)
        # else:
        tof = time_to_periapsis_from_per_M_n(per, M, n)

        maneuvers = [dv1_vec, dv2_vec, tof]
        return maneuvers


# Finished # Tested 8
def _elliptical_in_low_optimal(state_0: np.ndarray, rp_target: float, per_target: float, gm: float)\
        -> (list, np.ndarray):
    """
    :param state_0:
    :param rp_target:
    :return:
    """
    state_kep = inertial_to_keplerian_3d(state_0, gm)
    state_kep[-1] = np.pi
    state_opt = keplerian_to_inertial_3d(state_kep, gm)
    maneuvers = _elliptical_in_low_current(state_opt, rp_target, per_target, gm)
    return maneuvers, state_opt


# Finished # Tested 8
def _elliptical_in_high_current(state_0: np.ndarray, rp_target: float, per_target: float, gm: float) -> list:
    """
    :param state_0:
    :param rp_target:
    :param per_target:
    :return:
    """
    r0_vec, v0_vec = state_0[:3], state_0[3:]
    r0_mag, v0_mag = mag3(r0_vec), mag3(v0_vec)
    h0_vec = cross(r0_vec, v0_vec)
    h0_mag = mag3(h0_vec)
    gamma = gamma_from_h_r_v_qcheck(h0_mag, r0_mag, v0_mag, r0_vec, v0_vec)
    ra_target = ra_from_rp_per(rp_target, per_target, gm)
    a, e, i, w, om, f = inertial_to_keplerian_3d(state_0, gm)
    assert e < 1, 'Initial orbit is hyperbolic'
    ra = ra_from_a_e(a, e)
    rp = rp_from_a_e(a, e)
    tmp_state = np.empty(6)

    # If the s/c is within acceptable bounds, just leave it - job well done already
    if ra <= ra_target and rp >= rp_target:
        dv1_vec = np.zeros(3)
        dv2_vec = np.zeros(3)
        tof12 = 0.
        return [dv1_vec, dv2_vec, tof12]

    # Adjust periapsis at current location
    v1_mag = v_from_gm_r_rp_gamma(gm, r0_mag, rp_target, gamma)
    if not np.isnan(v1_mag):
        dv1_mag = v1_mag - v0_mag
        dv1_vec = v0_vec / v0_mag * dv1_mag
        v1_vec = v0_vec + dv1_vec
    # else s/c is coming in too sharply - the periapsis can't be raised while maintaining constant flight path angle
    else:
        a, e = a_e_from_rp_ra(rp_target, ra)
        v1_mag = v_from_gm_r_a(gm, r0_mag, a)
        h = h_from_gm_a_e(gm, a, e)
        gamma0 = gamma
        gamma = gamma_from_h_r_v(h, r0_mag, v1_mag) * -1
        v1_vec = v0_vec / v0_mag * v1_mag
        dgamma = gamma - gamma0
        if inc_from_h(h0_vec) <= np.pi / 2:
            dgamma *= -1
        v1_vec = euler313(v1_vec, 0, 0, dgamma)
        dv1_vec = v1_vec - v0_vec

    # At periapsis, adjust apoapsis
    energy = energy_from_gm_r_v(gm, r0_mag, v1_mag)
    if energy > 0 and gamma > 0:
        # Ascending: maneuvers will be to (attempt to) lower apoapsis at current point
        #   and then lower periapsis once reaching apoapsis
        if r0_mag < ra_target:
            # Conserve r_vec and gamma; only change v_mag such that ra becomes ra_target
            v1_mag = v_from_gm_r_ra_gamma(gm, r0_mag, ra_target, gamma)
            dv1_mag = v1_mag - v0_mag
            dv1_vec = v0_vec / v0_mag * dv1_mag
            v1_vec = v0_vec + dv1_vec

            # At apoapsis, change periapsis to target altitude
            energy = energy_from_gm_r_v(gm, r0_mag, v1_mag)
            a = a_from_gm_energy(gm, energy)
            dv2_mag = lower_periapsis(ra_target, a, rp_target, gm)
            tmp_state[:3], tmp_state[3:] = r0_vec, v1_vec
            a, e, i, w, om, f = inertial_to_keplerian_3d(tmp_state, gm)
            df1 = np.pi - f
            dgamma1 = 0 - gamma
            to_apoapsis = True

            # Check signs
            if i > np.pi / 2:
                df1 *= -1
                dgamma1 *= -1

            # Rotate dv2 vector into correct orientation
            dv2_vec = v1_vec / v1_mag * dv2_mag
            dv2_vec = euler313(dv2_vec, 0, 0, df1 - dgamma1)

            # Compute time between maneuvers
            E = E_from_f_e(f, e)
            E = fix_angle(E, 2 * np.pi, 0)
            M = M_from_E_e(E, e)
            per = per_from_gm_a(gm, a)
            n = n_from_per(per)
            if to_apoapsis:
                tof12 = time_to_apoapsis_from_per_M_n(per, M, n)
            else:
                tof12 = time_to_periapsis_from_per_M_n(per, M, n)

            maneuvers = [dv1_vec, dv2_vec, tof12]
            return maneuvers

        # If we're already super far away and ascending and would be hyperbolic, just gotta pull the e-brake and set
        # current location as new apoapsis
        else:
            a = a_from_rp_ra(rp_target, r0_mag)
            v1_mag = v_from_gm_r_a(gm, r0_mag, a)
            v1_vec = r0_vec.copy() / r0_mag * v1_mag
            v1_vec = euler313(v1_vec, 0, 0, np.pi / 2)
            dv1_vec = v1_vec - v0_vec

            # At periapsis, adjust apoapsis
            dv2_mag = lower_apoapsis(rp_target, a, ra_target, gm)
            tmp_state[:3], tmp_state[3:] = r0_vec, v1_vec
            i = inc_from_h(cross(r0_vec, v0_vec))
            df1 = np.pi
            dgamma1 = 0
            to_apoapsis = False

            # Check signs
            if i > np.pi / 2:
                df1 *= -1
                dgamma1 *= -1

            # Rotate dv2 vector into correct orientation
            dv2_vec = v1_vec / v1_mag * dv2_mag
            dv2_vec = euler313(dv2_vec, 0, 0, df1 - dgamma1)

            # Compute time between maneuvers
            e = e_from_rp_ra(rp_target, ra_target)
            E = E_from_f_e(np.pi, e)
            E = fix_angle(E, 2 * np.pi, 0)
            M = M_from_E_e(E, e)
            per = per_from_gm_a(gm, a)
            n = n_from_per(per)
            if to_apoapsis:
                tof12 = time_to_apoapsis_from_per_M_n(per, M, n)
            else:
                tof12 = time_to_periapsis_from_per_M_n(per, M, n)

            maneuvers = [dv1_vec, dv2_vec, tof12]
            return maneuvers

    # Descending, or above target apoapsis: maneuvers will be to set periapsis to correct altitude at current location,
    #     and then lower apoapsis to target altitude once reaching periapsis
    else:

        a = a_from_gm_energy(gm, energy)
        dv2_mag = lower_apoapsis(rp_target, a, ra_target, gm)
        a, e, i, w, om, f = inertial_to_keplerian_3d(np.hstack((r0_vec, v1_vec)), gm)
        # assert e < 1, 'Transfer orbit is hyperbolic'
        ra = ra_from_a_e(a, e)
        rp = rp_from_a_e(a, e)

        f = fix_angle(f, 2 * np.pi, 0)
        df = 2 * np.pi - f
        dgamma = 0 - gamma
        # to_apoapsis = False

        # Check signs
        if i > np.pi / 2:
            df *= -1
            dgamma *= -1

        # Rotate dv2 vector into correct orientation
        dv2_vec = v1_vec / v1_mag * dv2_mag
        dv2_vec = euler313(dv2_vec, 0, 0, df - dgamma)

        # Compute time between maneuvers
        E = E_from_f_e(f, e)
        E = fix_angle(E, 2 * np.pi, 0)
        M = M_from_E_e(E, e)
        per = per_from_gm_a(gm, a)
        n = n_from_per(per)
        # if to_apoapsis:
        #     tof = time_to_apoapsis_from_per_M_n(per, M, n)
        # else:
        tof = time_to_periapsis_from_per_M_n(per, M, n)

        maneuvers = [dv1_vec, dv2_vec, tof]
        return maneuvers


# Finished # Tested 8
def _elliptical_in_high_optimal(state_0: np.ndarray, rp_target: float, per_target: float, gm: float)\
        -> (list, np.ndarray):
    """
    :param state_0:
    :param rp_target:
    :param per_target:
    :return:
    """
    # dv1_vec = np.zeros(3)
    # dv2_vec = np.zeros(3)
    # tof = 0.
    # maneuvers = [dv1_vec, dv2_vec, tof]
    # return maneuvers

    state_kep = inertial_to_keplerian_3d(state_0, gm)
    state_kep[-1] = np.pi
    state_opt = keplerian_to_inertial_3d(state_kep, gm)
    maneuvers = _elliptical_in_high_current(state_opt, rp_target, per_target, gm)
    return maneuvers, state_opt


def _get_time_to_maneuver(a, e, f, gm, to_apoapsis):
    # Compute time between maneuvers
    if e < 1:  # elliptical
        E = E_from_f_e(f, e)
        E = fix_angle(E, 2 * np.pi, 0)
        M = M_from_E_e(E, e)
        per = per_from_gm_a(gm, a)
        n = n_from_per(per)
        if to_apoapsis:
            tof = time_to_apoapsis_from_per_M_n(per, M, n)
        else:
            tof = time_to_periapsis_from_per_M_n(per, M, n)
    else:  # hyperbolic
        H = H_from_f_e(f, e)  # * sign_check(r0_vec, v1_vec)
        M = e * np.sinh(H) - H
        n = (gm / - a ** 3) ** 0.5
        per = per_from_gm_a(gm, -a)
        # tof = time_to_periapsis_from_per_M_n(per, M, n)
        tof = - M / n  # TODO when to use above line vs. this line?
    return tof


def _get_maneuver_vector(r_vec, v_vec, v_mag, dv_mag, gamma, gm, to_apoapsis):
    """
    r_vec, v_vec, v_mag, and gamma all correspond to a known state vector. dv_mag is the magnitude of the dv vector to
    compute. to_apoapsis defines if the maneuver will occur at periapsis or apoapsis.
    :param r_vec:
    :param v_vec:
    :param v_mag:
    :param dv_mag:
    :param gamma:
    :param gm:
    :param to_apoapsis:
    :return:
    """
    # Compute orbital elements at after first maneuver
    tmp_state2 = np.empty(6)
    tmp_state2[:3], tmp_state2[3:] = r_vec, v_vec
    # gamma = np.arccos(max(min(mag3(cross(r_vec, v_vec)) / mag3(r_vec) / mag3(v_vec), 1), -1))
    a, e, i, w, om, f = inertial_to_keplerian_3d(tmp_state2, gm)
    f = fix_angle(f, 2 * np.pi, 0)

    # Compute change in velocity vector angle
    if to_apoapsis:
        df = np.pi - f
    else:
        df = 2 * np.pi - f
    dgamma = 0 - gamma

    # Flip signs if retrograde
    if i > np.pi / 2:
        df *= -1
        dgamma *= -1

    # Rotate dv2 vector into correct orientation
    dv_vec = v_vec / v_mag * dv_mag
    dv_vec = euler313(dv_vec, 0, 0, df - dgamma)

    tof = _get_time_to_maneuver(a, e, f, gm, to_apoapsis)

    maneuver = [dv_vec, tof]
    return maneuver


def _get_capture_at_periapsis_magnitude(rp_target, ra_target, a, e, gm, low_not_high):
    # Compute dv to capture into low or high orbit
    if low_not_high:  # circularize
        dv_mag = lower_apoapsis(rp_target, a, rp_target, gm)
    else:  # lower apoapsis to target - if already lower, don't do anything
        ra = ra_from_a_e(a, e)
        if not rp_target <= ra <= ra_target:
            dv_mag = lower_apoapsis(rp_target, a, ra_target, gm)
        else:
            dv_mag = 0.
    return dv_mag


def _in_current(state_0: np.ndarray, rp_target: float, per_target: float, gm: float, low_not_high: bool = True):
    r0_vec, v0_vec = state_0[:3], state_0[3:]
    r0_mag, v0_mag = mag3(r0_vec), mag3(v0_vec)
    h0_vec = cross(r0_vec, v0_vec)
    h0_mag = mag3(h0_vec)
    gamma0 = gamma_from_h_r_v_qcheck(h0_mag, r0_mag, v0_mag, r0_vec, v0_vec)
    ra_target = ra_from_rp_per(rp_target, per_target, gm)
    a, e, i, w, om, f = inertial_to_keplerian_3d(state_0, gm)
    f = fix_angle(f, 2 * np.pi, 0)
    ra = ra_from_a_e(a, e)
    rp = rp_from_a_e(a, e)
    tmp_state = np.empty(6)

    # Define some flags to guide logic below
    is_retrograde = i > np.pi / 2
    is_hyperbolic = e > 0.997
    is_descending = gamma0 < 0
    need_to_raise_rp = rp < rp_target
    need_to_lower_rp = rp > ra_target
    need_to_lower_ra = ra > ra_target or is_hyperbolic
    above_ra_target = r0_mag > ra_target

    # First, start with descending case.
    # 1. Check that rp > rp_target. If not, do a periapsis raise maneuver at current location. If rp is raised to
    #       rp_target, then capture after reaching periapsis.
    # 2. Check that rp < ra_target. If not, at periapsis do a maneuver to set rp to rp_target. At new periapsis, do
    #       capture maneuver.
    # 3. Check if hyperbolic or ra > ra_target - if so, then do a maneuver at periapsis to set ra to ra_target.
    # 4. At apoapsis, do periapsis lower maneuver to set rp to rp_target.
    # 5. If capturing into a low circular orbit, do a maneuver at rp to circularize.

    # Then look at ascending case.
    # 6. Check if r < ra_target. If not, do a maneuver to set current location as apoapsis and rp = rp_target. At
    #       periapsis, capture.
    # 7. Check that rp_target <= ra <= ra_target. If not, at current location do an apoapsis lower maneuver to set ra to
    #       ra_target.
    # 8. Steps 4-5.

    if is_descending:
        # 1. Check that rp >= rp_target. If not, do a periapsis raise maneuver at current location. If rp is raised to
        #       rp_target, then capture after reaching periapsis.
        if need_to_raise_rp:
            # # Try to adjust periapsis at current location with a tangential burn
            # v1_mag = v_from_gm_r_rp_gamma(gm, r0_mag, rp_target, gamma0)
            # # If it worked (returns a real number), compute the maneuver
            # if not np.isnan(v1_mag):
            #     dv1_mag = v1_mag - v0_mag
            #     dv1_vec = v0_vec / v0_mag * dv1_mag
            #     v1_vec = v0_vec + dv1_vec
            #     gamma1 = gamma0
            # # If it failed, the s/c is coming in too sharply, and the periapsis can't be raised enough while
            # # maintaining a constant flight path angle. Need a non-tangential maneuver to raise periapsis.
            # else:
            a, e = a_e_from_rp_ra(rp_target, ra)
            v1_mag = v_from_gm_r_a(gm, r0_mag, a)
            h = h_from_gm_a_e(gm, a, e)
            gamma1 = gamma_from_h_r_v(h, r0_mag, v1_mag) * -1  # we know we're descending
            v1_vec = v0_vec / v0_mag * v1_mag
            dgamma = gamma1 - gamma0
            if is_retrograde:  # account for retrograde orbits
                dgamma *= -1
            v1_vec = euler313(v1_vec, 0, 0, -dgamma)
            dv1_vec = v1_vec - v0_vec

            # Now s/c is en route to rp_target. The next maneuver will be capture at periapsis.
            energy = energy_from_gm_r_v(gm, r0_mag, v1_mag)
            a = a_from_gm_energy(gm, energy)
            dv2_mag = _get_capture_at_periapsis_magnitude(rp_target, ra_target, a, e, gm, low_not_high)
            dv2_vec, tof12 = _get_maneuver_vector(r0_vec, v1_vec, v1_mag, dv2_mag, gamma1, gm, to_apoapsis=False)
            maneuvers = [dv1_vec, dv2_vec, tof12]

    # 2. Check that rp <= ra_target. If not, at periapsis do a maneuver to set rp to rp_target. At new periapsis, do
    #       capture maneuver.
        elif need_to_lower_rp:
            # Coast from current point to periapsis
            dv1_vec = np.zeros(3)

            # At periapsis, set current point as new apoapsis and new periapsis as rp_target
            dv2_mag = lower_periapsis(rp, a, rp_target, gm)
            dv2_vec, tof12 = _get_maneuver_vector(r0_vec, v0_vec, v0_mag, dv2_mag, gamma0, gm, to_apoapsis=False)
            # Compute velocity at original periapsis after maneuver
            v2_mag = dv2_mag + v_from_gm_r_a(gm, rp, a)
            v2_vec = dv2_vec / dv2_mag * v2_mag
            r2_mag = rp
            df = 2 * np.pi - f
            if is_retrograde:
                df *= -1
            r2_vec = euler313(r0_vec, 0, 0, df) / r0_mag * r2_mag
            gamma2 = 0

            # Now s/c is en route to rp_target. The next maneuver will be capture at periapsis.
            energy = energy_from_gm_r_v(gm, r2_mag, v2_mag)
            a = a_from_gm_energy(gm, energy)
            e = e_from_rp_ra(rp_target, r2_mag)
            dv3_mag = _get_capture_at_periapsis_magnitude(rp_target, ra_target, a, e, gm, low_not_high)
            dv3_vec, tof23 = _get_maneuver_vector(r2_vec, v2_vec, v2_mag, dv3_mag, gamma2, gm, to_apoapsis=False)
            maneuvers = [dv1_vec, dv2_vec, dv3_vec, tof12, tof23]

    # 3. Check if hyperbolic or ra > ra_target - if so, then do a maneuver at periapsis to set ra to ra_target.
        elif need_to_lower_ra:
            # Coast from current point to periapsis
            dv1_vec = np.zeros(3)

            # At periapsis, lower apoapsis to ra_target
            dv2_mag = lower_apoapsis(rp, a, ra_target, gm)
            dv2_vec, tof12 = _get_maneuver_vector(r0_vec, v0_vec, v0_mag, dv2_mag, gamma0, gm, to_apoapsis=False)

            # Compute velocity at original periapsis after maneuver
            a = a_from_rp_ra(rp, ra_target)
            v2_mag = v_from_gm_r_a(gm, rp, a)
            v2_vec = dv2_vec / dv2_mag * v2_mag
            r2_mag = rp
            df = 0 - f  # (0 - f) vs. (2*np.pi - f)    ???
            if is_retrograde:
                df *= -1
            r2_vec = euler313(r0_vec, 0, 0, df) / r0_mag * r2_mag
            gamma2 = 0

            # At apoapsis, lower periapsis
            dv3_mag = lower_periapsis(ra_target, a, rp_target, gm)
            dv3_vec, tof23 = _get_maneuver_vector(r2_vec, v2_vec, v2_mag, dv3_mag, gamma2, gm, to_apoapsis=True)
            a = a_from_rp_ra(rp_target, ra_target)
            v3_mag = v_from_gm_r_a(gm, ra_target, a)
            v3_vec = dv3_vec / dv3_mag * v3_mag
            r3_vec = -r2_vec / r2_mag * ra_target
            gamma3 = 0

            # If necessary, circularize at periapsis
            if low_not_high:
                dv4_mag = lower_apoapsis(rp_target, a, rp_target, gm)
                dv4_vec, tof34 = _get_maneuver_vector(r3_vec, v3_vec, v3_mag, dv4_mag, gamma3, gm, to_apoapsis=False)
                maneuvers = [dv1_vec, dv2_vec, dv3_vec, dv4_vec, tof12, tof23, tof34]
                # raise NotImplementedError('Cannot handle 4 maneuvers')
            else:
                maneuvers = [dv1_vec, dv2_vec, dv3_vec, tof12, tof23]

    # 4. At apoapsis, do periapsis lower maneuver to set rp to rp_target.
        else:
            # Coast from current point to apoapsis
            dv1_vec = np.zeros(3)

            # At apoapsis, lower periapsis
            dv2_mag = lower_periapsis(ra, a, rp_target, gm)
            dv2_vec, tof12 = _get_maneuver_vector(r0_vec, v0_vec, v0_mag, dv2_mag, gamma0, gm, to_apoapsis=True)
            a = a_from_rp_ra(rp_target, ra)
            v2_mag = v_from_gm_r_a(gm, ra, a)
            v2_vec = dv2_vec / dv2_mag * v2_mag
            r2_mag = ra
            # df = 2 * np.pi - f
            df = 0 - f  # TODO when to use 2 * np vs 0 ???
            if is_retrograde:
                df *= -1
            r2_vec = euler313(r0_vec, 0, 0, df) / r0_mag * r2_mag
            gamma2 = 0

            # If necessary, circularize at periapsis.
            if low_not_high:
                dv3_mag = lower_apoapsis(rp_target, a, rp_target, gm)
                dv3_vec, tof23 = _get_maneuver_vector(r2_vec, v2_vec, v2_mag, dv3_mag, gamma2, gm, to_apoapsis=False)
                maneuvers = [dv1_vec, dv2_vec, dv3_vec, tof12, tof23]
            else:
                maneuvers = [dv1_vec, dv2_vec, tof12]

    else:
        # 6. Check if r < ra_target. If not, do a maneuver to set current location as apoapsis and rp = rp_target. At
        #       periapsis, capture.
        if above_ra_target:
            a, e = a_e_from_rp_ra(rp_target, r0_mag)
            v1_mag = v_from_gm_r_a(gm, r0_mag, a)
            # h = h_from_gm_a_e(gm, a, e)
            gamma1 = 0
            dgamma = 0 - np.pi / 2
            if is_retrograde:  # account for retrograde orbits
                dgamma *= -1
            v1_vec = euler313(r0_vec / r0_mag, 0, 0, -dgamma) * v1_mag
            dv1_vec = v1_vec - v0_vec

            # Now s/c is en route to rp_target. The next maneuver will be capture at periapsis.
            # energy = energy_from_gm_r_v(gm, r0_mag, v1_mag)
            # a = a_from_gm_energy(gm, energy)
            dv2_mag = _get_capture_at_periapsis_magnitude(rp_target, ra_target, a, e, gm, low_not_high)
            dv2_vec, tof12 = _get_maneuver_vector(r0_vec, v1_vec, v1_mag, dv2_mag, gamma1, gm, to_apoapsis=False)
            maneuvers = [dv1_vec, dv2_vec, tof12]

    # 7. Check that rp_target <= ra <= ra_target. If not, at current location do an apoapsis lower maneuver to set ra to
    #       ra_target.
        elif need_to_lower_ra:
            a, e = a_e_from_rp_ra(rp, ra_target)
            v1_mag = v_from_gm_r_a(gm, r0_mag, a)
            h = h_from_gm_a_e(gm, a, e)
            gamma1 = gamma_from_h_r_v(h, r0_mag, v1_mag)
            v1_vec = v0_vec / v0_mag * v1_mag
            dgamma = gamma1 - gamma0
            if is_retrograde:  # account for retrograde orbits
                dgamma *= -1
            v1_vec = euler313(v1_vec, 0, 0, -dgamma)
            dv1_vec = v1_vec - v0_vec

            # 8. At apoapsis, do periapsis lower maneuver to set rp to rp_target.
            # At apoapsis, lower periapsis
            dv2_mag = lower_periapsis(ra_target, a, rp_target, gm)
            dv2_vec, tof12 = _get_maneuver_vector(r0_vec, v1_vec, v1_mag, dv2_mag, gamma1, gm, to_apoapsis=True)
            tmp_state[:3], tmp_state[3:] = r0_vec, v1_vec
            a, e, i, w, o, f = inertial_to_keplerian_3d(tmp_state, gm)
            a = a_from_rp_ra(rp_target, ra_target)
            v2_mag = v_from_gm_r_a(gm, ra_target, a)
            v2_vec = -dv2_vec / abs(dv2_mag) * v2_mag
            r2_mag = ra_target
            df = np.pi - f
            if is_retrograde:
                df *= -1
            r2_vec = euler313(r0_vec / r0_mag * r2_mag, 0, 0, df)
            gamma2 = np.arccos(max(min(mag3(cross(r2_vec, v2_vec)) / mag3(r2_vec) / mag3(v2_vec), 1), -1))

            # If necessary, circularize at periapsis.
            if low_not_high:
                dv3_mag = lower_apoapsis(rp_target, a, rp_target, gm)
                dv3_vec, tof23 = _get_maneuver_vector(r2_vec, v2_vec, v2_mag, dv3_mag, gamma2, gm, to_apoapsis=False)
                if need_to_raise_rp:
                    # Normally, both maneuvers are anti-velocity, but in this case the apoapsis maneuver is parallel to
                    # velocity, so after rotating vector by pi, need to flip the other way
                    dv3_vec *= -1
                maneuvers = [dv1_vec, dv2_vec, dv3_vec, tof12, tof23]
            else:
                maneuvers = [dv1_vec, dv2_vec, tof12]

    # Otherwise coast to apoapsis.
        else:
            # Coast from current point to apoapsis
            dv1_vec = np.zeros(3)

            # 8. At apoapsis, do periapsis lower maneuver to set rp to rp_target.
            # At apoapsis, lower periapsis
            dv2_mag = lower_periapsis(ra, a, rp_target, gm)
            dv2_vec, tof12 = _get_maneuver_vector(r0_vec, v0_vec, v0_mag, dv2_mag, gamma0, gm, to_apoapsis=True)
            a = a_from_rp_ra(rp_target, ra)
            v2_mag = v_from_gm_r_a(gm, ra, a)
            v2_vec = dv2_vec / dv2_mag * v2_mag
            r2_mag = ra
            df = np.pi - f
            if is_retrograde:
                df *= -1
            r2_vec = euler313(r0_vec, 0, 0, df) / r0_mag * r2_mag
            gamma2 = 0

            # If necessary, circularize at periapsis.
            if low_not_high:
                dv3_mag = lower_apoapsis(rp_target, a, rp_target, gm)
                dv3_vec, tof23 = _get_maneuver_vector(r2_vec, v2_vec, v2_mag, dv3_mag, gamma2, gm, to_apoapsis=False)
                maneuvers = [dv1_vec, dv2_vec, dv3_vec, tof12, tof23]
            else:
                maneuvers = [dv1_vec, dv2_vec, tof12]

    return maneuvers


# TODO
def _in_optimal():
    pass


# TODO treat anything outside of sphere of influence as hyperbolic
def _out_optimal(state_0: np.ndarray, rp_target: float, per_target: float, gm_central: float, gm_target: float,
                 low_not_high: bool = True):
#     r_vec, v_vec = state_0[:3], state_0[3:6]
#     r_mag, v_mag = mag3(r_vec), mag3(v_vec)
#     # compute transfer to target SOI
#     dv1_vec, dv2_vec, tof12 = lambert_min_dv(gm, state_0, state_f, short=True, do_print=False)
#
#     # compute velocity when target reaches SOI
#     v_soi = 0
#     # then compute corresponding v_inf
#     v_rel_soi = v_mars - v_soi
#     energy = energy_from_gm_r_v(gm_target, c.r_soi_mars, v_rel_soi)
#     v_inf = (2 * energy) ** 0.5
#     # compute dv to capture
#     vp_hyper = (2 * gm_target / rp_target + v_inf * v_inf) ** 0.5
#     if low_not_high:
#         vp_capture = (gm_target / rp_target) ** 0.5
#     else:
#         a = a_from_gm_per(gm_target, per_target)
#         vp_capture = v_from_gm_r_a(gm_target, rp_target, a)
#     dv_mag = vp_hyper - vp_capture
    pass


# TODO the main still-to-be-solved problem is when I am outside the sphere-of-influence and want to get to it. Mars'
#          state is a function of time, so if I try to solve the min delta V Lambert arc by adjusting time, I also need
#          to adjust Mars' state accordingly. I do have the ephem function which could be used. Is this something
#          normally solved by SQP? Is there a fast solution? Is this where B-plane targeting comes into play?
# TODO Compare my machine learning missed thrust solution to a perturbation technique for TCMs (see pg. 534 in Battin)?
#          Is there an advantage to one way or another?
# TODO What if I had thrust stop some time (e.g. 30 days) before arrival. Then the capture occurs in those 30 days.
# TODO Look into relative motion/proximity operations for two satellites in orbit. There is no gravity between them but
#          the problems often involve rendezvous. There are probably relevant anwers to what I'm looking for.


def hyperbolic_capture_from_infinity(v_inf: float, rp_target: float, per_target: float, gm: float,
                                     low_not_high: bool = True) -> float:
    vp_hyper = (2 * gm / rp_target + v_inf * v_inf) ** 0.5
    if low_not_high:
        vp_capture = (gm / rp_target) ** 0.5
    else:
        a = a_from_gm_per(gm, per_target)
        vp_capture = v_from_gm_r_a(gm, rp_target, a)
    return vp_hyper - vp_capture


if __name__ == "__main__":
    test1 = False  # Test min_dv_capture
    if test1:
        r = np.random.rand(2) * 4e5 - 2e5
        v = np.random.rand(2) * 4 - 2
        gm = 42328.372
        rp_target = 3900
        print(min_dv_capture(r, v, gm, rp_target))

    test2 = False  # Test change_central_body  # TODO fix change_central_body
    if test2:
        r1 = np.array([10000, 5000, 0.])
        v1 = np.array([-3, -4, 0.])
        r2 = np.array([-150000000, 0, 0.])
        v2 = np.array([0, -30, 0.])
        # r3 = change_central_body(r1, r2)
        # v3 = change_central_body(v1, v2)
        # print(r3)
        # print(v3)

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
        i = 51.6460 * np.pi / 180  # deg
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
        def compute_capture_sequence(state_0):
            # r_vec, v_vec = state_0[:3], state_0[3:]
            gm = c.u_mars_km3s2

            # Compute capture
            rp_target = c.r_mars_km + 500
            per_target = 10 * 86400
            low_not_high = True
            # maneuvers, *args = _elliptical_in_high_optimal(state_0, rp_target, per_target, gm)
            maneuvers, *args = _in_current(state_0, rp_target, per_target, gm, low_not_high)

            # Parse outputs
            if len(args) == 1:
                state_opt = args[0]
                state_0 = state_opt
            elif len(args) == 2 or len(args) == 4 or len(args) == 6:
                maneuvers = [maneuvers, *args]
            else:
                raise ValueError('Unknown or unprepared outputs from capture function.')
            r_vec, v_vec = state_0[:3], state_0[3:]
            if len(maneuvers) == 3:
                dv1_vec, dv2_vec, tof12 = maneuvers
                # dv1_mag, dv2_mag = mag3(dv1_vec), mag3(dv2_vec)
                dv3_vec, dv4_vec, dv3_mag, dv4_mag, tof23, tof34 = np.zeros(3), np.zeros(3), 0, 0, 0, 0
            elif len(maneuvers) == 5:
                dv1_vec, dv2_vec, dv3_vec, tof12, tof23 = maneuvers
                # dv1_mag, dv2_mag, dv3_mag = mag3(dv1_vec), mag3(dv2_vec), mag3(dv3_vec)
                dv4_vec, dv4_mag, tof34 = np.zeros(3), 0, 0
            else:
                dv1_vec, dv2_vec, dv3_vec, dv4_vec, tof12, tof23, tof34 = maneuvers
                # dv1_mag, dv2_mag, dv3_mag, dv4_mag = mag3(dv1_vec), mag3(dv2_vec), mag3(dv3_vec), mag3(dv4_vec)

            # print('DV1 mag (km/s):\t\t%f' % abs(dv1_mag))
            # print('DV2 mag (km/s):\t\t%f' % abs(dv2_mag))
            # if len(maneuvers) == 3:
            #     print('Total DV mag (km/s):\t%f' % (abs(dv1_mag) + abs(dv2_mag)))
            # elif len(maneuvers) == 5:
            #     print('DV3 mag (km/s):\t\t%f' % abs(dv3_mag))
            #     print('Total DV mag (km/s):\t%f' % (abs(dv1_mag) + abs(dv2_mag) + abs(dv3_mag)))
            # else:
            #     print('DV3 mag (km/s):\t\t%f' % abs(dv3_mag))
            #     print('DV4 mag (km/s):\t\t%f' % abs(dv4_mag))
            #     print('Total DV mag (km/s):\t%f' % (abs(dv1_mag) + abs(dv2_mag) + abs(dv3_mag) + abs(dv4_mag)))
            #
            # print('\nTime to second maneuver (hr):\t%f' % (tof12 / 3600))
            # if len(maneuvers) > 3:
            #     print('Time to third maneuver (hr):\t%f' % (tof23 / 3600))
            #     if len(maneuvers) > 5:
            #         print('Time to fourth maneuver (hr):\t%f' % (tof34 / 3600))

            tf = 50000.
            tol = 1e-10
            e_tol = 1e-4
            a_tol = 1e-1
            n_steps = 101
            du = 20000
            tu = (du ** 3 / c.u_mars_km3s2) ** 0.5
            state_scales = np.array([du, du, du, du / tu, du / tu, du / tu])
            step_type = 1
            failed = False

            tof0 = period_from_inertial(state_0, gm, max_time_sec=tf)
            traj1 = tbp.prop(list(state_0 / state_scales), [0., -tof0 / tu], [], 6, 2, 0, tol, tol, tf / n_steps / tu,
                             step_type, 3)
            traj1 = (np.array(traj1)[:, 1:] * state_scales).T
            traj11 = []
            if tof0 == tf:
                traj11 = tbp.prop(list(state_0 / state_scales), [0., tof0 / tu], [], 6, 2, 0, tol, tol,
                                  tf / n_steps / tu, step_type, 3)
                traj11 = (np.array(traj11)[:, 1:] * state_scales).T

            v_transfer_vec = state_0[3:6] + dv1_vec
            state_transfer = np.hstack((r_vec, v_transfer_vec))
            traj2 = tbp.prop(list(state_transfer / state_scales), [0., tof12 / tu], [], 6, 2, 0, tol, tol,
                             tof12 / n_steps / tu, step_type, 3)
            traj2 = (np.array(traj2)[:, 1:] * state_scales).T

            v3_vec = traj2[3:, -1] + dv2_vec
            # ra_target = ra_from_rp_per(rp_target, per_target, gm)
            state3 = np.hstack((traj2[:3, -1], v3_vec))
            if len(maneuvers) == 3:
                tof23 = period_from_inertial(state3, gm, max_time_sec=tf)
            traj3 = tbp.prop(list(state3 / state_scales), [0., tof23 / tu], [], 6, 2, 0, tol, tol,
                             tof23 / n_steps / tu, step_type, 3)
            traj3 = (np.array(traj3)[:, 1:] * state_scales).T
            trajs = [traj1, traj11, traj2, traj3]
            if len(maneuvers) > 3:
                v4_vec = traj3[3:, -1] + dv3_vec
                state4 = np.hstack((traj3[:3, -1], v4_vec))
                if len(maneuvers) == 5:
                    tof34 = period_from_inertial(state4, gm, max_time_sec=tf)
                traj4 = tbp.prop(list(state4 / state_scales), [0., tof34 / tu], [], 6, 2, 0, tol, tol,
                                 tof34 / n_steps / tu, step_type, 3)
                traj4 = (np.array(traj4)[:, 1:] * state_scales).T
                trajs.append(traj4)
                if len(maneuvers) > 5:
                    v5_vec = traj4[3:, -1] + dv4_vec
                    state5 = np.hstack((traj4[:3, -1], v5_vec))
                    per5 = period_from_inertial(state5, gm)
                    traj5 = tbp.prop(list(state5 / state_scales), [0., per5 / tu], [], 6, 2, 0, tol, tol,
                                     per5 / n_steps / tu, step_type, 3)
                    traj5 = (np.array(traj5)[:, 1:] * state_scales).T
                    trajs.append(traj5)
                    a, e, i, w, o, f = inertial_to_keplerian_3d(traj5[:, -1], gm)
                    if abs(e) > e_tol or abs(a - rp_target) > a_tol:
                        failed = True
                else:
                    a, e, i, w, o, f = inertial_to_keplerian_3d(traj4[:, -1], gm)
                    if abs(e) > e_tol or abs(a - rp_target) > a_tol:
                        failed = True
            else:
                a, e, i, w, o, f = inertial_to_keplerian_3d(traj3[:, -1], gm)
                if abs(e) > e_tol or abs(a - rp_target) > a_tol:
                    failed = True

            return trajs, tof0 == tf, failed, [a, e]

        def plot_capture_sequence(trajs, has_traj11):
            traj1, traj11, traj2, traj3 = trajs[:4]
            traj4 = trajs[4] if len(trajs) > 4 else np.array([])
            traj5 = trajs[5] if len(trajs) > 5 else np.array([])

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            circ = plt.Circle((0, 0), c.r_mars_km, color='red')
            ax.add_artist(circ)
            plt.plot(traj1[0], traj1[1])
            if has_traj11:
                plt.gca().set_prop_cycle(None)
                plt.plot(traj11[0], traj11[1], '--')
            plt.scatter(traj1[0, 0], traj1[1, 0])
            plt.plot(traj2[0], traj2[1])
            plt.scatter(traj2[0, -1], traj2[1, -1])
            plt.plot(traj3[0], traj3[1])
            if len(trajs) > 4:
                plt.scatter(traj3[0, -1], traj3[1, -1])
                plt.plot(0, 0)
                plt.plot(traj4[0], traj4[1])
                if len(trajs) > 5:
                    plt.scatter(traj4[0, -1], traj4[1, -1])
                    plt.plot(0, 0)
                    plt.plot(traj5[0], traj5[1])
            ax.axis('equal')
            # x, y = traj2.y[0, -1], traj2.y[1, -1]
            # plt.quiver(x, y, dv2_vec[0], dv2_vec[1])
            plt.show()

        # x, y, vx, vy = 2e5, 2e5, -0.5, 0.3
        # tests = np.array([x,  y, 0., vx,  vy, 0])  # ascending, prograde

        tests = np.array([[2e4, 2e4, 0, -1, 1.2, 0],  # ascending + else, descending + need_to_raise_rp
                          [5e4, 2e4, 0, -1, 1.2, 0],  # descending + need_to_lower_ra (hyperbolic)
                          [2e5, 2e5, 0, 0.08, 0.1, 0],  # descending + need_to_raise_rp, ascending + above_ra_target
                          [1e4, 1e5, 0, -0.3, 0.5, 0],  # descending + else, ascending + else
                          [2e5, 2e5, 0, -0.5, 0.3, 0],  # descending + need_to_lower_rp, descending + need_to_lower_ra
                          [5e4, 0, 0, -0.1, 1.2, 0]])

        plot_failures = False
        fails = []
        for i in range(tests.shape[0]):
            num_subtests = 16
            subtests = np.array([tests[i].copy()] * num_subtests)
            subtests[[1, 5, 6, 7, 11, 12, 13, 15], 0] *= -1
            subtests[[2, 5, 8, 9, 11, 12, 14, 15], 1] *= -1
            subtests[[3, 6, 8, 10, 11, 13, 14, 15], 3] *= -1
            subtests[[4, 7, 9, 10, 12, 13, 14, 15], 4] *= -1
            # subtests[1, 0] *= -1
            # subtests[2, 1] *= -1
            # subtests[3, 3] *= -1
            # subtests[4, 4] *= -1
            # subtests[5, [0, 1]] *= -1
            # subtests[6, [0, 3]] *= -1
            # subtests[7, [0, 4]] *= -1
            # subtests[8, [1, 3]] *= -1
            # subtests[9, [1, 4]] *= -1
            # subtests[10, [3, 4]] *= -1
            # subtests[11, [0, 1, 3]] *= -1
            # subtests[12, [0, 1, 4]] *= -1
            # subtests[13, [0, 3, 4]] *= -1
            # subtests[14, [1, 3, 4]] *= -1
            # subtests[15, [0, 1, 3, 4]] *= -1
            for j in range(num_subtests):
                trajs, has_traj11, failed, [a, e] = compute_capture_sequence(subtests[j])
                if failed:
                    if plot_failures:
                        plot_capture_sequence(trajs, has_traj11)
                    fails.append(subtests[j])
        print('Failed cases:')
        if len(fails) == 0:
            print('None!')
        else:
            [print(f) for f in fails]

    test8 = False  # Test capture()
    if test8:
        # Define problem
        state_rel = np.array([1000000, -1400000, 0., -1, -5, 0])
        rp_target = c.r_mars_km + 500
        per_target = 10 * 86400
        gm = c.u_mars_km3s2
        r_soi = c.r_soi_mars
        capture_low_not_high = False
        capture_current_not_opt = False
        # Compute capture
        maneuvers = capture(state_rel, rp_target, per_target, gm, r_soi, capture_low_not_high, capture_current_not_opt)
        y, full_traj = propagate_capture(maneuvers, state_rel, m0=10000)
        # Plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.plot(full_traj[0], full_traj[1])
        ax.axis('equal')
        plt.show()

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

        test9_1 = True
        if test9_1:
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

        test9_2 = True
        if test9_2:
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

    test10 = False  # Test find_min()
    if test10:
        def f(x):
            return 0.4 * x ** 3 + 5 * x ** 2 - 7.2 * x + 0.4278
        tests = np.arange(20) + 1
        for i in range(len(tests)):
            x, y = find_min(f, -10, 20, num_iter=tests[i])
            print('Iterations: %i' % tests[i])
            print(x)
            print(y)
            print()

    test11 = True  # Test lambert_min_dv()
    if test11:
        r0_vec = np.array([0, -1, 0], float)
        r0_vec = r0_vec / mag3(r0_vec) * c.a_mars_km * 0.95
        v0_vec = np.matmul(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]]), r0_vec)
        v0_vec = v0_vec / mag3(v0_vec) * (c.u_sun_km3s2 / mag3(r0_vec)) ** 0.5
        state_0 = np.hstack((r0_vec, v0_vec))
        low, high = 60, 61
        max_count = 0
        short = False
        t0 = float(tc.times_jd1950_jc[-1] / c.day_to_jc * c.day_to_sec)

        # dv1_vec, dv2_vec, tof12 = lambert_min_dv(tc.gm, state_0, t0, low, high, max_count=max_count, short=short)
        # lambert_min_dv(tc.gm, state_0, t0, low, high, max_count=max_count, short=short)
        import time
        lambert_t0 = time.time()
        sol, tof12 = lambert_min_dv(tc.gm, state_0, t0, low, high, max_count=max_count, short=short)
        lambert_tf = time.time()
        print('\nElapsed time in lambert_min_dv(): %e sec\n' % (lambert_tf - lambert_t0))

        t0_jc = t0 * c.sec_to_day * c.day_to_jc
        tf_jc = tof12 * c.day_to_jc + t0_jc
        mars_f = c.ephem(['a', 'e', 'i', 'w', 'O', 'M'], ['mars'], np.array([tf_jc]))
        mars_f[2] = 0.
        mars_f = keplerian_to_inertial_3d(mars_f, gm=c.u_sun_km3s2, mean_or_true='mean')

        dv1_vec = sol[0] - v0_vec
        dv2_vec = mars_f[3:6] - sol[1]

        print("Delta V's:")
        print(dv1_vec)
        print(dv2_vec)
        print('\nTotal Delta V = %f km/s' % (mag3(dv1_vec) + mag3(dv2_vec)))
        print('Time-of-flight = %f days\n' % tof12)

        # Propagate transfer
        gm = c.u_sun_km3s2
        r_vec, v_vec = state_0[:3], state_0[3:]
        tf = 50000.
        tol = 1e-8
        n_steps = 201
        du = c.a_mars_km
        tu = (du ** 3 / gm) ** 0.5
        state_scales = np.array([du, du, du, du / tu, du / tu, du / tu])
        step_type = 0
        failed = False

        # Propagate initial spacecraft orbit
        tof0 = period_from_inertial(state_0, gm, max_time_sec=tf)
        traj1 = tbp.prop(list(state_0 / state_scales), [0., -tof0 / tu], [], 6, 2, 0, tol, tol,
                         tof0 / n_steps / tu - 1e-10, step_type, 3)
        traj1 = (np.array(traj1)[:, 1:] * state_scales).T

        # Propagate transfer orbit
        v_transfer_vec = state_0[3:6] + dv1_vec
        state_transfer = np.hstack((r_vec, v_transfer_vec))
        traj2 = tbp.prop(list(state_transfer / state_scales), [0., tof12 * c.day_to_sec / tu], [], 6, 2, 0, tol, tol,
                         tof12 * c.day_to_sec / n_steps / tu - 1e-10, step_type, 3)
        traj2 = (np.array(traj2)[:, 1:] * state_scales).T

        # Compute error between final propagated position and target position
        mars_f_dif = mag3(mars_f[:3] - traj2[:3, -1])
        print('mars_f_dif [km] = %f' % mars_f_dif)
        print('mars_f_dif [AU] = %f' % (mars_f_dif / c.au_to_km))
        print('mars_f_dif [r_soi_mars] = %f' % (mars_f_dif / c.r_soi_mars))

        # Compute final spacecraft orbit
        v3_vec = traj2[3:, -1] + dv2_vec
        state3 = np.hstack((traj2[:3, -1], v3_vec))
        tof23 = period_from_inertial(state3, gm, max_time_sec=tf)
        traj3 = tbp.prop(list(state3 / state_scales), [0., tof23 / tu], [], 6, 2, 0, tol, tol,
                         tof23 / n_steps / tu - 1e-10, step_type, 3)
        traj3 = (np.array(traj3)[:, 1:] * state_scales).T

        # Propagate Mars orbit
        mars_0 = c.ephem(['a', 'e', 'i', 'w', 'O', 'M'], ['mars'], np.array([t0 * c.sec_to_day * c.day_to_jc]))
        mars_0[2] = 0.
        mars_0 = keplerian_to_inertial_3d(mars_0, gm, 'mean')
        tof34 = period_from_inertial(mars_0, gm, max_time_sec=tf)
        traj4 = tbp.prop(list(mars_0 / state_scales), [0., tof34 / tu], [], 6, 2, 0, tol, tol,
                         tof34 / n_steps / tu - 1e-10, step_type, 3)
        traj4 = (np.array(traj4)[:, 1:] * state_scales).T

        # Plot orbits
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.plot(traj1[0], traj1[1], label='S/C init orbit')
        plt.scatter(mars_0[0], mars_0[1], s=40, facecolors='none', edgecolors='r', zorder=9, label='Mars init')
        plt.scatter(traj1[0, 0], traj1[1, 0], zorder=9, label='S/C at t=0')
        plt.plot(traj2[0], traj2[1], label='Transfer')
        plt.scatter(traj2[0, -1], traj2[1, -1], s=40, facecolors='r', edgecolors='r', zorder=9, label='S/C final')
        plt.scatter(mars_f[0], mars_f[1], s=40, facecolors='r', edgecolors='g', zorder=9, label='Mars final')
        plt.plot(traj3[0], traj3[1], label='S/C final orbit')
        plt.plot(traj4[0], traj4[1], label='Mars orbit')
        ax.axis('equal')
        x, y = traj1[0, -1], traj1[1, -1]
        plt.quiver(x, y, dv1_vec[0], dv1_vec[1], zorder=8)
        x, y = traj2[0, -1], traj2[1, -1]
        plt.quiver(x, y, dv2_vec[0], dv2_vec[1], zorder=8)
        plt.legend(loc='center')
        plt.title('Transfer = %.1f days' % tof12)
        # plt.show()

    test12 = False  # Plot a sample trajectory to test tolerances
    if test12:
        state_0 = np.array([100000, 0, 0, 0, 2, 0], float)
        gm = c.u_earth_km3s2
        tf = 50000.
        n_steps = 1000
        rtol_test = 1e-8
        atol_test = 1e-8
        rtol_truth = atol_truth = 1e-16
        du = mag3(state_0[:3])
        tu = (du ** 3 / gm) ** 0.5
        state_scales = np.array([du, du, du, du / tu, du / tu, du / tu])
        step_type = 1
        n_revolutions = 100

        tof0 = period_from_inertial(state_0, gm, max_time_sec=tf) * n_revolutions
        traj_test = tbp.prop(list(state_0 / state_scales), [0., tof0 / tu], [], 6, 2, 0, rtol_test, atol_test, 1.,
                             step_type, 3)
        traj_test = (np.array(traj_test)[:, 1:] * state_scales).T

        traj_truth = tbp.prop(list(state_0 / state_scales), [0., tof0 / tu], [], 6, 2, 0, rtol_truth, atol_truth, 1.,
                              step_type, 3)
        traj_truth = (np.array(traj_truth)[:, 1:] * state_scales).T

        pos_err = mag3(traj_test[:3, -1] - traj_truth[:3, -1])
        vel_err = mag3(traj_test[3:, -1] - traj_truth[3:, -1])
        print('Position error at end = %.3e km' % pos_err)
        print('Velocity error at end = %.3e km/s' % vel_err)

        # Plot transfer
        import matplotlib.pyplot as plt
        fig12, ax = plt.subplots()
        circ = plt.Circle((0, 0), c.r_earth_km, color='blue')
        ax.add_artist(circ)
        plt.plot(traj_test[0], traj_test[1], label='Test')
        plt.plot(traj_truth[0], traj_truth[1], label='Truth')
        plt.scatter(traj_truth[0, 0], traj_truth[1, 0], zorder=9, label='S/C at t=0')
        ax.axis('equal')
        plt.legend()
        plt.title('Transfer = %.1f days' % (tof0 / 86400))
        plt.show()
