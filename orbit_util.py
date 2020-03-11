import numpy as np
from math import gamma
from traj_config import gm, year_to_sec
from numba import njit
from copy import copy


@njit
def hohmann_circ(a1, a2, gm=gm):
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
def hohmann_rp_ra(rp1, vp1, ra2, va2, gm=gm):
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
def coe4_from_rv(r_vec, v_vec, gm=gm):
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
def period_from_inertial(state, gm=gm, max_time_sec=10*year_to_sec):
    a, e, i, w, om, f = inertial_to_keplerian_3d(state, gm=gm)
    if e < 1:
        per = 2 * np.pi * np.sqrt((a ** 3) / gm)
        # per = np.min((per, max_time_sec))
    else:
        per = max_time_sec
    return per


@njit
def inertial_to_local(state, has_extra=True):
    s1i = state[:4]
    s2i = state[4:8]
    if has_extra:
        extra = state[8:]
    r1 = np.sqrt(s1i[0] * s1i[0] + s1i[1] * s1i[1])
    r2 = np.sqrt(s2i[0] * s2i[0] + s2i[1] * s2i[1])
    v1 = np.sqrt(s1i[2] * s1i[2] + s1i[3] * s1i[3])
    v2 = np.sqrt(s2i[2] * s2i[2] + s2i[3] * s2i[3])
    th1 = np.arctan2(s1i[1], s1i[0])
    th2 = np.arctan2(s2i[1], s2i[0])
    al1 = np.arctan2(s1i[3], s1i[2])
    al2 = np.arctan2(s2i[3], s2i[2])
    if has_extra:
        return np.array([r1, r2, v1, v2, th1, th2, al1, al2, *extra])
    else:
        return np.array([r1, r2, v1, v2, th1, th2, al1, al2])


@njit
def inertial_to_keplerian(state, gm=gm):
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
def inertial_to_local_2d(state):
    x, y, vx, vy = state
    r = np.sqrt(x * x + y * y)
    v = np.sqrt(vx * vx + vy * vy)
    th = np.arctan2(y, x)
    al = np.arctan2(vy, vx)
    return np.array([r, v, th, al])


@njit
def inertial_to_keplerian_2d_old(state, gm=gm):
    r, v, th, al = inertial_to_local_2d(state)
    eps = v ** 2 / 2 - gm / r
    a = - gm / (2 * eps)
    fpa = -fix_angle(al - th - np.pi / 2)
    h = r * v * np.cos(fpa)
    p = h ** 2 / gm
    e = np.sqrt(1 - np.min((p / a, 1.0)))
    if e < 1e-6:
        arg = np.sign(p / r - 1)
    else:
        arg = (p / r - 1) / e
        if np.abs(arg) > 1:
            arg = np.sign(arg)
    ta = np.arccos(arg) * np.sign(fpa+1e-8)
    w = fix_angle(th - ta)
    return np.array([a, e, w, ta])


@njit
def inertial_to_keplerian_2d(state, gm=gm):
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
        ta = np.arccos(arg) * np.sign(fpa+1e-8)
        w = fix_angle(th - ta)
    return np.array([a, e, w, ta])


@njit
def cross(left, right):
    x = ((left[1] * right[2]) - (left[2] * right[1]))
    y = ((left[2] * right[0]) - (left[0] * right[2]))
    z = ((left[0] * right[1]) - (left[1] * right[0]))
    return np.array([x, y, z])


@njit
def inertial_to_keplerian_3d(state, gm=gm):
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


# @njit
def keplerian_to_perifocal_3d(state, gm=gm):
    a, e, i, w, om, f = state
    p = a * (1 - e ** 2)
    r_p = np.hstack((p * np.cos(f) / (1 + e * np.cos(f)), p * np.sin(f) / (1 + e * np.cos(f)), 0.))
    v_p = np.hstack((-np.sqrt(gm / p) * np.sin(f), np.sqrt(gm / p) * (e + np.cos(f)), 0.))
    return np.hstack((r_p, v_p))


# @njit
def keplerian_to_inertial_3d(state, gm=gm):
    a, e, i, w, om, f = state
    state_peri = keplerian_to_perifocal_3d(state, gm=gm)
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
def keplerian_to_perifocal_2d(state, gm=gm):
    a, e, w, f = state
    p = a * (1 - e ** 2)
    r_p = np.hstack((np.array(p * np.cos(f) / (1 + e * np.cos(f))), np.array(p * np.sin(f) / (1 + e * np.cos(f)))))
    v_p = np.hstack((np.array(-np.sqrt(gm / p) * np.sin(f)), np.array(np.sqrt(gm / p) * (e + np.cos(f)))))
    return np.hstack((r_p, v_p))


@njit
def keplerian_to_inertial_2d(state, gm=gm):
    # Convert to perifocal frame
    a, e, w, f = state
    state_peri = keplerian_to_perifocal_2d(state, gm=gm)
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
def keplerian_to_mee_3d(state):
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
def mee_to_keplerian_3d(state):
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
def rotate_vnc_to_inertial_3d(vec, state):
    r_vec, v_vec = state[:3], state[3:6]    # radius and velocity vectors
    v_hat = v_vec / np.linalg.norm(v_vec)   # velocity unit vector
    h_vec = cross(r_vec, v_vec)          # angular momentum vector
    n_hat = h_vec / np.linalg.norm(h_vec)   # angular momentum unit vector; also, normal unit vector
    c_hat = cross(v_hat, n_hat)          # co-normal unit vector
    dcm = np.vstack((v_hat, n_hat, c_hat))  # direction cosine matrix
    return np.matmul(dcm, vec)


@njit
def fix_angle(angle, upper_bound=np.pi, lower_bound=-np.pi):
    # Check that bounds are properly defined
    assert upper_bound - lower_bound == 2 * np.pi
    assert not np.isnan(angle)
    while True:
        angle += 2 * np.pi if angle < lower_bound else 0.  # add 2pi if too negative
        angle -= 2 * np.pi if angle > upper_bound else 0.  # subtract 2pi if too positive
        if angle <= upper_bound and angle >= lower_bound:
            return angle


def min_energy_lambert(r0, r1, gm=gm):
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
def c2(psi):
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
def c3(psi):
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
def vallado(k, r0, r, tof, short, numiter, rtol):
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


def lambert_min_dv(k, r0, v0, rf, vf, short=True, do_print=False):
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
    return best_tof, dv1, dv2


def gamma_from_r_v(r_vec, v_vec):
    r_mag = np.linalg.norm(r_vec)
    r_hat = r_vec / r_mag

    v_mag = np.linalg.norm(v_vec)
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


def min_dv_capture(r, v, gm):
    # Make sure vectors are 3D
    if r.size == 2:
        r = np.append(r, 0)
    if v.size == 2:
        v = np.append(v, 0)
    r_mag = np.linalg.norm(r)
    r_hat = r / r_mag
    v_mag = np.linalg.norm(v)
    v_hat = v / v_mag

    # Calculate flight path angle
    gamma = gamma_from_r_v(r, v)

    ra_max = 4e5 # km
    rp = 3389.5 + 100 # km
    # Check how s/c is moving wrt to planet
    if gamma > 0: # ascending
        if r_mag < ra_max: # below max ra
            ra = ra_max
            v0 = v_mag
            a1 = (ra + rp) / 2
            v1 = (gm * (2 / r_mag - 1 / a1)) ** 0.5
            dv1 = v1 - v0

        else: # above max ra
            ra = r_mag
            v0 = v_mag
            a1 = (ra + rp) / 2
            v1 = (gm * (2 / r_mag - 1 / a1)) ** 0.5
            a0, e0, w0, f0 = inertial_to_keplerian_2d(np.hstack((r[:2], v[:2])))
            n0 = (gm / a0 ** 3) ** 0.5
            fdot0 = n0 * a0 * a0 / r / r * (1 - e0 ** 2) ** 0.5
            rdot0 = r_mag * fdot0 * e0 * np.sin(f0) / (1 + e0 * np.cos(f0))
            dv1 = np.linalg.norm([rdot0, 0, 0])
            dv1 = v1 - v0

    elif gamma < 0: # decending
        rp1, r1, v0 = rp, r_mag, v_mag
        eps0 = v0 ** 2 / 2 - gm / r1
        a0 = - gm / 2 / eps0
        e0_vec = ((v0 ** 2 - gm / r1) * r - np.dot(r, v) * v) / gm
        e0 = np.linalg.norm(e0_vec)
        rp0 = a0 * (1 - e0)
        vp0 = (gm * a0 * (1 - e0 ** 2)) ** 0.5 / rp0
        a1 = (rp1 * ((rp0 * vp0 / rp1 / v0) ** 2 - 1)) / (2 * (rp0 ** 2 * vp0 ** 2 / v0 ** 2 / rp1 / r1) - 1)
        e1 = 1 - rp1 / a1
        vp1 = (gm * a1 * (1 - e1 ** 2)) ** 0.5 / rp1
        v1 = (gm * (2 / r1 - 1 / a1)) ** 0.5

    else: # apse
        ra = r_mag

    v2 = (gm / rp1) ** 0.5
    dv2 = v2 - vp1

    # Calculate desired orbit
    r_desired_mag = 2000
    r_desired_hat = - r / np.linalg.norm(r)
    r_desired =  r_desired_hat * r_desired_mag
    h = np.cross(r, v)
    if h.size == 1:
        h = np.array([0, 0, h])
    h_hat = h / np.linalg.norm(h)
    v_desired_mag = np.sqrt(gm / r_desired_mag)
    v_desired_hat = np.cross(h_hat, r_desired_hat)
    v_desired = v_desired_hat * v_desired_mag
    # Compute transfer
    v0, v1, tof = min_energy_lambert(r, r_desired, gm=gm)
    dv1 = v0 - v
    dv2 = v_desired - v1
    return dv1, dv2, tof


if __name__ == "__main__":
    test_1 = False
    if test_1:
        state_k = np.array([150e6, 0.5, 2, 2, 2, 2])
        state_m = keplerian_to_mee_3d(state_k)
        state_k2 = mee_to_keplerian_3d(state_m)
        state_i = keplerian_to_inertial_3d(state_k).ravel()
        state_k3 = inertial_to_keplerian_3d(state_i)
        state_m2 = keplerian_to_mee_3d(state_k3)
        print(state_k)
        print(state_m)
        print(state_k2)
        print(state_i)
        print(state_k3)
        print(np.allclose(state_m2, state_m))

    test_2 = True
    if test_2:
        r = np.array([100000, 0, 0])
        v = np.array([1, 4, 0])
        gamma = gamma_from_r_v(r, v)
        print(gamma)
