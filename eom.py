from constants import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from traj_config import gm


def eom2BP(t: float, x: np.ndarray, u: float=gm) -> np.ndarray:
    x_dot = np.zeros(x.shape)
    r = np.sqrt(np.sum(np.square(x[0:2])))
    x_dot[0:2] = x[2:]
    x_dot[2:] = -u / r**3 * x[0:2]
    return x_dot


def eom2BPThrust(t: float, y: np.ndarray, yf: np.ndarray, m_dry: float, u: float=gm, thrust_fcn=None, Isp_s: float=2000, T_max_kN: float=0.001) -> np.ndarray:
    # Define parameters
    y_dot = np.zeros(y.shape)
    r = np.sqrt(np.sum(np.square(y[0:2])))
    # Calculate thrust vector
    if y[4] > m_dry:
        thrust_vec = thrust_fcn(np.hstack((y[:4], yf, y[4]/m_dry))) * T_max_kN
    else:
        thrust_vec = np.array([0, 0])
    thrust_mag = np.sqrt(thrust_vec.dot(thrust_vec))
    # Derivatives
    y_dot[0:2] = y[2:4]
    y_dot[2:4] = -u / r**3 * y[0:2] + thrust_vec / y[4]
    y_dot[4] = - thrust_mag * 1000 / g0_ms2 / Isp_s
    return y_dot


def eom2BP_scaled(t: float, x_sc: np.ndarray, eta_r: np.ndarray, eta_v: np.ndarray) -> np.ndarray:
    x_sc_dot = np.zeros(x_sc.shape)
    r_sc = np.sqrt(np.sum(np.square(x_sc[0:3])))
    x_sc_dot[0:3] = x_sc[3:] * eta_r
    x_sc_dot[3:] = x_sc[1:3] / r_sc**3 * eta_v
    return x_sc_dot


def eomMEE(L: float, x: np.ndarray, T: float, I_sp: float, m0: float) -> np.ndarray:
    # Pull out MEE from vector
    p, f, g, h, k, [], m = x

    # Define spacecraft acceleration vectors
    if m <= 0.5*m0:
        T = 0

    Delta_T = T / m * u_earth_km3s2
    Delta_g = 0

    # Sum accelerations
    Delta = Delta_g + Delta_T
    Delta_r, Delta_theta, Delta_h = Delta

    # Define auxiliary terms
    q = 1 + f * np.cos(L) + g * np.sin(L)
    s2 = 1 + h**2 + k**2

    # Define time derivatives of MEE
    dpdt = np.sqrt(p / u_earth_km3s2) * 2 * p / q * Delta_theta
    dfdt = np.sqrt(p / u_earth_km3s2) * (np.sin(L) * Delta_r + 1 / q * ((q+1) + np.cos(L) + f) * Delta_theta - g / q * (h * np.sin(L) - k * np.cos(L)) * Delta_h)
    dgdt = np.sqrt(p / u_earth_km3s2) * (-np.cos(L) * Delta_r + 1 / q * ((q+1) * np.sin(L) + g) * Delta_theta + f / q * (h * np.sin(L) - k * np.cos(L)) * Delta_h)
    dhdt = np.sqrt(p / u_earth_km3s2) * s2 * np.cos(L) / 2 / q * Delta_h
    dkdt = np.sqrt(p / u_earth_km3s2) * s2 * np.sin(L) / 2 / q * Delta_h
    dLdt = np.sqrt(p / u_earth_km3s2) * (h * np.sin(L) - k * np.cos(L)) * Delta_h + np.sqrt(u_earth_km3s2 * p) * (q / p)**2
    dmdt = - T / g0_ms2 / I_sp

    # Invert dLdt to get derivatives with respect to L instead of t
    dtdL = 1 / dLdt

    # L derivatives of MEE
    xdot = dtdL * np.array([dpdt, dfdt, dgdt, dhdt, dkdt, 1, dmdt])
    return xdot


def eom2BPThrustScaled(t: float, y: np.ndarray, yf: np.ndarray, m_dry: float, thrust_fcn, du: float, tu: float,
                       mu: float, fu: float, Isp_s: float=2000, T_max_kN: float=0.001) -> np.ndarray:
    # Define parameters
    y_dot = np.zeros(y.shape)
    r = np.sqrt(np.sum(np.square(y[0:2])))
    # Calculate thrust vector
    # if y[4] > m_dry:
    #     scales = np.array([du, du, du/tu, du/tu])
    #     thrust_vec = thrust_fcn(np.hstack((y[:4] * scales, yf, y[4] * mu / m_dry))) * T_max_kN / fu
    # else:
    #     thrust_vec = np.array([0, 0])
    thrust_vec = np.array([1, 1]) * T_max_kN / fu
    thrust_mag = np.sqrt(thrust_vec.dot(thrust_vec))
    # Derivatives
    y_dot[:2] = y[2:4]
    y_dot[2:4] = -1 / r**3 * y[0:2] + thrust_vec / y[4]
    y_dot[4] = - thrust_mag * 1000 / g0_ms2 / Isp_s
    return y_dot


# def eom3BP_scaled(t, x):


# def eom3BP_stm(t, x, u):


# def eomNPB(t, x):


if __name__ == "__main__":
    # Boundary conditions
    x0, y0 = 10000.0, 0.0
    vx0, vy0 = 0.0, np.sqrt(u_earth_km3s2 / x0)
    m0 = 200.0
    t0, tf = 0, 2*np.pi*np.sqrt(x0**3 / u_earth_km3s2)*10
    # Scale factors
    du = 6371.0
    tu = np.sqrt(du**3 / u_earth_km3s2)
    vu = du / tu
    mu = m0
    fu = mu * vu / tu
    # Scaled BCs
    x0s, y0s = x0 / du, y0 / du
    vx0s, vy0s = vx0 / vu, vy0 / vu
    m0s = m0 / mu
    t0s, tfs = t0 / tu, tf / tu
    yy0 = np.array([x0s, y0s, vx0s, vy0s, m0s])
    # Integrate
    yout = solve_ivp(lambda t, x: eom2BPThrustScaled(t, x, 0, 0, 0, 0, 0, 0, fu), (t0, tfs), yy0, rtol=1e-8, atol=1e-8)
    # Plot
    plt.plot(yout.y[0, :] * du, yout.y[1, :] * du)
    plt.axis('equal')
    plt.show()
    plt.plot(yout.t * tu / 3600, yout.y[-1, :] * mu)
    plt.show()
