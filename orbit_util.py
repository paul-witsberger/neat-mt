import numpy as np
from math import gamma, cos, sin
from traj_config import gm, year_to_sec
from numba import njit
from copy import copy
from constants import ephem, sec_to_day, reference_date_jd1950, day_to_jc


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
    a, e, i, w, om, f = inertial_to_keplerian_3d(state, gm=gm)
    if e < 1:
        per = 2 * np.pi * np.sqrt((a ** 3) / gm)
        # per = np.min((per, max_time_sec))
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


# @njit
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


@njit
def inertial_to_keplerian_3d(state: np.ndarray, gm: float = gm) -> np.ndarray:
    """
    Convert a 3D state vector from inertial to Keplerian
    :param state:
    :param gm:
    :return:
    """
    r_vec, v_vec = state[:3], state[3:]
    tol = 1e-8
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    h_vec = cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    k_vec = np.array([0, 0, 1])
    n_vec = cross(k_vec, h_vec)
    n = np.linalg.norm(n_vec)

    # Eccentricity
    e_vec =  ((v ** 2 - gm / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / gm
    e = np.linalg.norm(e_vec)
    eps = v ** 2 / 2 - gm / r

    # Semi-major axis
    if e == 1:
        a = np.infty
        p = h ** 2 / gm
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
    if n == 0:
        # Special case - equatorial
        # w = np.arccos(np.dot(np.array([1, 0, 0]), e_vec))
        w = 0.
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
    elif (i < tol or np.abs(np.pi - i) < tol):
        # Special case: elliptical equatorial - true longitude of periapsis
        f = np.arccos(e_vec[0] / e)
        if e_vec[1] < 0:
            f = 2 * np.pi - f
    else:
        # General
        f = np.arccos(min(np.dot(e_vec, r_vec) / (e * r), 1.))
        if np.dot(r_vec, v_vec) < 0:
            f = 2 * np.pi - f

    return np.array([a, e, i, w, om, f])


@njit
def keplerian_to_perifocal_3d(state: np.ndarray, gm: float = gm, mean_or_true: str = 'true') -> np.ndarray:
    """
    Convert a 3D state vector from Keplerian to perifocal
    :param state:
    :param gm:
    :return:
    """
    if mean_or_true == 'true':
        a, e, i, w, om, f = state
    else:
        a, e, i, w, om, m = state
        f = np.array([mean_to_true_anomaly(mm, e) for mm in m], float)
    p = a * (1 - e ** 2)
    r_p = np.array([p * np.cos(f) / (1 + e * np.cos(f)), p * np.sin(f) / (1 + e * np.cos(f)), 0.])
    v_p = np.array([-np.sqrt(gm / p) * np.sin(f), np.sqrt(gm / p) * (e + np.cos(f)), 0.])
    return np.array((r_p, v_p)).ravel()


@njit
def keplerian_to_inertial_3d(state: np.ndarray, gm: float = gm, mean_or_true: str = 'true') -> np.ndarray:
    """
    Convert a 3D state vector from Keplerian to inertial
    :param state:
    :param gm:
    :return:
    """
    a, e, i, w, om, f = state
    state_peri = keplerian_to_perifocal_3d(state, gm=gm, mean_or_true=mean_or_true)
    r_p, v_p = state_peri[:3], state_peri[3:]

    # Perform 3-1-3 rotation element-wise
    dcm3_1 = np.zeros((1, 3, 3)) if len(a.shape) == 0 else np.zeros((len(a.shape), 3, 3))
    dcm1_2 = dcm3_1.copy()
    dcm3_3 = dcm3_1.copy()

    dcm3_1[:, 0, 0] = np.cos(om)
    dcm3_1[:, 0, 1] = -np.sin(om)
    dcm3_1[:, 1, 0] = np.sin(om)
    dcm3_1[:, 1, 1] = np.cos(om)
    dcm3_1[:, 2, 2] = 1.

    dcm1_2[:, 0, 0] = 1.
    dcm1_2[:, 1, 1] = np.cos(i)
    dcm1_2[:, 1, 2] = -np.sin(i)
    dcm1_2[:, 2, 1] = np.sin(i)
    dcm1_2[:, 2, 2] = np.cos(i)

    dcm3_3[:, 0, 0] = np.cos(w)
    dcm3_3[:, 0, 1] = -np.sin(w)
    dcm3_3[:, 1, 0] = np.sin(w)
    dcm3_3[:, 1, 1] = np.cos(w)
    dcm3_3[:, 2, 2] = 1.

    dcm = np.matmul(np.matmul(dcm3_1, dcm1_2), dcm3_3)
    r_i = np.matmul(dcm, r_p)
    v_i = np.matmul(dcm, v_p)
    return np.hstack((r_i, v_i))


@njit
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


@njit
def keplerian_to_inertial_2d(state: np.ndarray, gm: float = gm, mean_or_true: str = 'true') -> np.ndarray:
    """
    Convert a 2D state vector from Keplerian to inertial
    :param state:
    :param gm:
    :return:
    """
    # Convert to perifocal frame
    state_peri = keplerian_to_perifocal_2d(state, gm=gm, mean_or_true=mean_or_true)
    r_p, v_p = state_peri[:2], state_peri[2:]
    # Construct DCMs
    # if len(np.shape(a)) == 0:
    dcm = np.zeros((2, 2))
    dcm[0, 0] = np.cos(w)
    dcm[0, 1] = -np.sin(w)
    dcm[1, 0] = np.sin(w)
    dcm[1, 1] = np.cos(w)
    # else:
    #     dcm = np.zeros((len(np.shape(a)), 2, 2))
    #     dcm[:, 0, 0] = np.cos(w)
    #     dcm[:, 0, 1] = -np.sin(w)
    #     dcm[:, 1, 0] = np.sin(w)
    #     dcm[:, 1, 1] = np.cos(w)
    # Rotate radius and velocity vectors
    # r_i = np.matmul(dcm, r_p)
    # v_i = np.matmul(dcm, v_p)
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
    f = e * np.cos(w + om)
    g = e * np.sin(w + om)
    h = np.tan(i / 2) * np.cos(om)
    k = np.tan(i / 2) * np.sin(om)
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


@njit
def rotate_vnc_to_inertial_3d(vec: np.ndarray, state: np.ndarray) -> np.ndarray:
    """
    Rotates the current velocity vector to an inertial frame.
    :param vec:
    :param state:
    :return:
    """
    r_vec, v_vec = state[:3], state[3:6]    # radius and velocity vectors
    v_hat = v_vec / np.linalg.norm(v_vec)   # velocity unit vector
    h_vec = cross(r_vec, v_vec)             # angular momentum vector
    n_hat = h_vec / np.linalg.norm(h_vec)   # angular momentum unit vector; also, normal unit vector
    c_hat = cross(v_hat, n_hat)             # co-normal unit vector
    dcm = np.vstack((v_hat, n_hat, c_hat))  # direction cosine matrix
    return np.matmul(dcm, vec)


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


@njit
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

    h_vec = np.cross(r_vec, v_vec)
    h_mag = np.linalg.norm(h_vec)
    assert h_mag > 0
    h_hat = h_vec / h_mag

    t_hat = np.cross(h_hat, r_hat)
    gamma_mag = np.arccos(np.dot(t_hat, v_hat))
    gamma_sign = 1 if np.cross(v_hat, t_hat)[-1] > 0 else -1
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
    if r_vec.size == 2:
        a, e, w, f = inertial_to_keplerian_2d(np.hstack((r_vec, v_vec)))
    else:
        try:
            a, e, i, w, om, f = inertial_to_keplerian_3d(np.hstack((r_vec, v_vec)))
        except ZeroDivisionError as e:
            print(r_vec)
            print(v_vec)
            raise e
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
    if r_vec.size == 2:
        a, e, w, f = inertial_to_keplerian_2d(np.hstack((r_vec, v_vec)))
    else:
        a, e, i, w, om, f = inertial_to_keplerian_3d(np.hstack((r_vec, v_vec)))

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
    :param r_sc_vec:
    :param v_sc_vec:
    :param r_target_vec:
    :param v_target_vec:
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
    epsilon = v_mag_kms * v_mag_kms / 2 - gm / r_mag_km
    # Define target orbit
    v_final_kms = (gm / r_periapsis_km) ** 0.5
    # Check if current state is hyperbolic
    if epsilon > 0:
        v_mag_capture_kms = (2 * gm / r_mag_km) ** 0.5
        v_periapsis_kms = (2 * gm / r_periapsis_km) ** 0.5
        dv1_mag = v_mag_capture_kms - v_mag_kms
        gamma = gamma_from_r_v(r_vec, v_vec)
        if gamma > 0:
            dv1_mag -= v_mag_capture_kms
    else:
        dv1_mag = 0
        v_periapsis_kms = (2 * (epsilon + gm / r_periapsis_km)) ** 0.5
    # Compute velocity vector after the maneuvers
    dv1_vec = v_hat * dv1_mag
    v_transfer_vec = v_vec + dv1_vec
    # Compute delta v to get into target orbit at periapsis
    dv2_mag = v_final_kms - v_periapsis_kms
    # Compute delta v vector and final velocity vector
    true_anomaly = true_anomaly_from_r_v(r_vec, v_transfer_vec)
    v_final_vec = rotate_vector_2d(np.array([v_final_kms, 0, 0]), true_anomaly)
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
def shift_vector_origin(cb1_to_sc: np.ndarray, cb2_to_cb1: np.ndarray) -> np.ndarray:
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


def change_central_body(states: np.ndarray, times: np.ndarray, cur_cb: str, new_cb: str) -> np.ndarray:
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
    elements = ['a', 'e', 'w', 'M'] if n_dim == 2 else ['a', 'e', 'i', 'w', 'o', 'M']
    assert cur_cb == 'sun' or new_cb == 'sun'
    if cur_cb == 'sun':
        planets = [new_cb]
        flip = False
    else:
        planets = [cur_cb]
        flip = True
    times_jc = (times * sec_to_day + reference_date_jd1950) * day_to_jc
    non_sun_cb_states = ephem(elements, planets, times_jc)
    # Flip relative direction if the new body is 'sun'
    cb2_to_cb1 = non_sun_cb_states if not flip else -non_sun_cb_states
    # Convert to inertial coordinates
    if n_dim == 4:
        keplerian_to_inertial_2d(cb2_to_cb1, mean_or_true='mean')
    else:
        keplerian_to_inertial_3d(cb2_to_cb1, mean_or_true='mean')
    # Shift vectors
    new_states = shift_vector_origin(states, cb2_to_cb1)
    return new_states


@njit
def mean_to_true_anomaly(m: float, e: float, tol: float = 1e-8) -> float:
    # Assume small eccentricity
    ea_guess = m
    ea_next = ea_guess
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


if __name__ == "__main__":
    test1 = False
    if test1:
        r = np.random.rand(2) * 4e5 - 2e5
        v = np.random.rand(2) * 4 - 2
        gm = 42328.372
        print(min_dv_capture(r, v, gm))

    test2 = False
    if test2:
        r1 = np.array([10000, 5000, 0.])
        v1 = np.array([-3, -4, 0.])
        r2 = np.array([-150000000, 0, 0.])
        v2 = np.array([0, -30, 0.])
        r3 = change_central_body(r1, r2)
        v3 = change_central_body(v1, v2)
        print(r3)
        print(v3)

    test3 = True
    if test3:
        m = 135 * np.pi / 180
        e = 0.1
        ta = mean_to_true_anomaly(m, e)
        print(ta)
        # Quick and dirty approximation
        ta2 = m + (2 * e - 0.25 * e ** 3) * sin(m) + 1.25 * e ** 2 * sin(2 * m) + 13 / 12 * e ** 3 * sin(3 * m)
        print(ta2)