import orbit_util as ou
import constants as c
import neatfast as neat
import pickle
import os
import traj_config as tc
from nnet import Neurocontroller
import boost_tbp
import missed_thrust
import numpy as np


_frames_conversions = {
    'kepmee3': ou.keplerian_to_mee_3d,
    'kepinr3': ou.keplerian_to_inertial_3d,
    'meekep3': ou.mee_to_keplerian_3d,
    'inrkep3': ou.inertial_to_keplerian_3d,
    'kepinr2': ou.keplerian_to_inertial_2d,
    'inrkep2': ou.inertial_to_keplerian_2d
}

_const_thrust_engine_params = {
    'thrust_max_n': 1.2 * 1e-3,
    'isp_s': 2780,
    'variable_power': False
}

_variable_thrust_engine_params = {
    'power_min_kw': 3.4,
    'power_max_kw': 12.5,
    'solar_array_m2': 1.,
    'power_reference': c.solar_constant * 1.,
    'thrust_power_coef': np.array([-363.67, 225.49, -21.475, 0.7943, 0], float) / 1000,
    'isp_power_coef': np.array([2274.5, -319.39, 61.817, -2.6802, 0], float),
    'variable_power': True
}
pm = _variable_thrust_engine_params['power_max_kw']
_variable_thrust_engine_params['thrust_power_kn'] = _variable_thrust_engine_params['thrust_power_coef'] \
                                                    * np.array([1, pm, pm ** 2, pm ** 3, pm ** 4]) * 1e-3
_variable_thrust_engine_params['isp_s'] = _variable_thrust_engine_params['isp_power_coef'] \
                                          * np.array([1, pm, pm ** 2, pm ** 3, pm ** 4])


class Spacecraft:
    def __init__(self, dim: float = 3, state: list = None, mass: float = None, engine_params: dict = None,
                 genome: neat.genome.DefaultGenome = None, config: neat.config.Config = None,
                 skip_controller: bool = False):
        assert state is None or (len(state) == 4 and dim == 2) or (len(state) == 6 and dim == 3)
        self._state = state if state is not None else [0.] * 6
        self._mass = mass if mass is not None else 0.
        self.engine_params = engine_params if engine_params is not None else _const_thrust_engine_params
        self.dry_mass = tc.m_dry
        self.controller = None
        if not skip_controller:
            self.set_controller(genome=genome, config=config)

    @property
    def state(self) -> list:
        return self._state

    @state.setter
    def state(self, value: list):
        self._state = value

    @property
    def mass(self) -> float:
        return self._mass

    @mass.setter
    def mass(self, value: float):
        self._mass = value

    def set_controller(self, genome: neat.genome.DefaultGenome = None, config: neat.config.Config = None,
                       fname: str = 'winner-feedforward'):
        """
        Create a NEAT network either from a given genome or from a file, and then assign the thrust function as the
        output of the network.
        :param genome:
        :param config:
        :param fname:
        :return:
        """
        if genome is None:
            with open(fname, 'rb') as f:
                genome = pickle.load(f)
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config_default')
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                 neat.DefaultStagnation, config_path)
        neat_net = neat.nn.FeedForwardNetwork.create(genome, config)
        nc = Neurocontroller.init_from_neat_net(neat_net, tc.scales_in, tc.scales_out)
        self.controller = nc.get_thrust_vec_neat


class Trajectory:
    def __init__(self, central_body: str = 'sun', frame: str = 'kep', dim: int = tc.n_dim,
                 genome: neat.genome.DefaultGenome = None, config: neat.config.Config = None,
                 save_full_traj: bool = False, skip_controller: bool = False):
        self.spacecraft = Spacecraft(dim=dim, genome=genome, config=config, skip_controller=skip_controller)
        self._dim = dim
        self._frame = frame
        self._central_body = central_body
        self.traj_config = tc
        self.flag = 0  # used for checking integration status
        self.save_full_traj = save_full_traj
        self.evaluated = False
        self.dv1 = np.zeros(3, float)
        self.dv2 = np.zeros(3, float)
        self.fitness = None
        self._init_integrator_opts()
        self._init_times_and_states()
        self._update_sizes()
        self._update_params(np.zeros(3, float))

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def frame(self) -> str:
        return self._frame

    @frame.setter
    def frame(self, value: str):
        self._frame = value

    @property
    def central_body(self) -> str:
        return self._central_body

    @central_body.setter
    def central_body(self, value: str):
        self.unscaled_states[:, :-1] = ou.change_central_body(self.unscaled_states[:, :-1], self.unscaled_times,
                                                              self.central_body, value)
        self.scaled_states[:, :-1] = self.unscaled_states[:, :-1] / self.state_scales[:-1]
        self._central_body = value

    # TODO update frame conversion formats - maybe make a wrapper that interprets the strings?
    def convert_frame_to(self, new_frame: str) -> None:
        """
        Change the state vector to a different frame.
        :param new_frame:
        :return:
        """
        if not new_frame == self.frame:
            self._state = _frames_conversions[self._frame + new_frame + str(self._dim)](self._state, self._central_body)
            self._frame = new_frame

    def _init_integrator_opts(self):
        """
        Helper function to set up integrator and related variables.
        :return:
        """
        # Define scales to nondimensionalize state for integration - distance unit, time unit, mass unit, force unit
        if self.dim == 2:
            self.state_scales = [tc.du, tc.du, tc.du / tc.tu, tc.du / tc.tu, tc.mu]
        else:
            self.state_scales = [tc.du, tc.du, tc.du, tc.du / tc.tu, tc.du / tc.tu, tc.du / tc.tu, tc.mu]
        self.integrator_opts = {'step_type': 0 if tc.fixed_step else 1,
                                'eom_type': 1 if tc.variable_power else 0}
        self.integrator = boost_tbp.TBP()

    def _init_times_and_states(self, spacing: str = 'linear'):
        """
        Creates arrays for time - both unscaled and scaled. Spacing between times can be linear or power law. Also
        create arrays for states - both unscaled (dimensional) and scaled (dimensionless). Includes position, velocity,
        and mass.
        :param spacing:
        :return:
        """
        if spacing == 'linear':
            t = np.linspace(tc.t0, tc.tf, tc.num_nodes)
        elif spacing == 'power':
            t = np.power(np.linspace(0, 1, tc.num_nodes), 3 / 2) * (tc.tf - tc.t0) + tc.t0
        else:
            raise(RuntimeError, 'Invalid spacing type for time vector')
        self.extra_nodes = int(tc.do_terminal_lambert_arc) * 2
        for _ in range(self.extra_nodes):
            t = np.append(t, t[-1])
        self.unscaled_times = t
        self.scaled_times = t / tc.tu
        self.unscaled_states = np.empty((tc.num_nodes + self.extra_nodes, 2 * self.dim + 1), float)
        self.scaled_states = np.empty_like(self.unscaled_states, float)
        self.full_traj = np.empty(((tc.num_nodes - 1) * tc.n_steps + tc.n_terminal_steps + 1,
                                   2 * self.dim + 2), float)
        # TODO consider scrapping unscaled_ and scaled_ times and states, and only use full_traj_scaled - maybe at end
        #  compute full_traj_unscaled?

    def _propagate_main_leg(self, y0: np.ndarray, yf: np.ndarray, include_missed_thrust: bool) -> None:
        """
        Propagates the trajectory of a spacecraft starting from the initial body along a powered flight towards a
        target body.
        :param y0:
        :param yf:
        :param include_missed_thrust:
        :return:
        """
        # Get the indices of the legs that will experience missed thrust
        if include_missed_thrust:
            miss_ind = missed_thrust.calculate_missed_thrust_events(self.unscaled_times)
        else:
            miss_ind = np.array([]).astype(int)

        self.unscaled_states[0, :6] = y0
        self.unscaled_states[0, 6] = tc.m0
        self.scaled_states[0, :] = self.unscaled_states[0] / self.state_scales

        # Main loop - check if integration should continue, get values needed for the integrator that change with each
        #             step, integrate one time step, and save results
        for i in range(tc.num_nodes - 1):
            # Check if orbital energy is within reasonable bounds - terminate integration if not
            r = ou.mag3(self.unscaled_states[i, :3])
            eps = (ou.mag3(self.unscaled_states[i, 3:6]) ** 2 / 2 - tc.gm / r)
            if (eps > tc.max_energy or eps < tc.min_energy) and not self.save_full_traj:
                self.flag = 1
                break

            # Fixed step integrator step size
            step_size = (self.scaled_times[i + 1] - self.scaled_times[i]) / tc.n_steps

            # Get ratio of remaining propellant mass and elapsed time ratio
            mass_ratio = (self.unscaled_states[i, -1] - self.spacecraft.dry_mass) / \
                         (self.unscaled_states[0, -1] - self.spacecraft.dry_mass)
            time_ratio = self.scaled_times[i] / self.scaled_times[-1]

            # Check if i is supposed to miss thrust for this segment
            if i in miss_ind or self.unscaled_states[i, -1] <= self.spacecraft.dry_mass + 0.01:
                # Missed-thrust event is occurring, or no propellant remaining
                thrust = np.array([0, 0, 0], float)
            else:
                # Query NN to get next thrust vector
                # TODO check if scale is same for constant vs variable power
                thrust = self.spacecraft.controller(np.hstack((self.unscaled_states[i, tc.ind_dim], yf[tc.ind_dim[:-1]],
                                                               mass_ratio, time_ratio))) \
                         * self.spacecraft.engine_params['thrust_max_n'] / tc.fu

            self._update_params(thrust)

            # propagate from the current state until the next time step
            traj = self.integrator.prop(self.scaled_states[i].tolist(), self.scaled_times[i:i+2].tolist(), self.params,
                                        self.sizes['states'], self.sizes['times'], self.sizes['param'], tc.rtol,
                                        tc.atol, step_size, self.integrator_opts['step_type'],
                                        self.integrator_opts['eom_type'])

            # save full trajectory including intermediate states
            self.full_traj[i * tc.n_steps:(i+1) * tc.n_steps] = traj[1:]  # TODO check mass consumption

            # save final state of current leg
            self.scaled_states[i + 1] = np.array(traj[-1])[1:]
            self.unscaled_states[i + 1] = self.scaled_states[i + 1] * self.state_scales

    def _post_transfer_capture(self, yf: np.ndarray) -> None:
        # Check if final position is "close" to target position - if not, compute a Lambert arc to match target state
        pos_error = ou.mag3((self.unscaled_states[-self.extra_nodes - 1, :3] - yf[:3]) / self.state_scales[:3])
        if tc.do_terminal_lambert_arc:
            # Compute maneuvers required to capture into a target orbit
            if pos_error > tc.position_tol:  # very far away
                self.dv1, self.dv2, tof = ou.lambert_min_dv(tc.gm, self.unscaled_states[-self.extra_nodes - 1], yf)
                change_frame = False
            else:
                # # TODO fix min_dv_capture - I'm pretty sure the second impulse is being calculated incorrectly
                # self.dv1, self.dv2, tof = ou.min_dv_capture(self.unscaled_states[-self.extra_nodes - 1], yf,
                #                                             c.u_mars_km3s2, c.r_mars_km + tc.capture_periapsis_alt_km)
                state_relative = self.unscaled_states[-self.extra_nodes - 1] - yf
                rp_target = c.r_mars_km + tc.capture_periapsis_alt_km
                per_target = tc.capture_period_day * c.day_to_sec
                maneuvers = ou.capture(state_relative, rp_target, per_target, tc.gm, c.r_soi_mars,
                                       tc.capture_low_not_high, tc.capture_current_not_optimal)
                change_frame = True

            # Compute delta v magnitudes
            dv1_mag = ou.mag3(self.dv1)
            dv2_mag = ou.mag3(self.dv2)

            # Add time of flight to the final leg
            self.scaled_times[-1] = self.scaled_times[-2] + tof / tc.tu
            self.unscaled_times[-1] = self.unscaled_times[-2] + tof

            # Add first delta v
            self.unscaled_states[-2, :6] = self.unscaled_states[-3, :6] + np.hstack(([0., 0., 0.], self.dv1))
            self.scaled_states[-2, :6] = self.unscaled_states[-2, :6] / self.state_scales[:6]
            if change_frame:
                self.central_body = 'mars'
            # TODO instead of changing everything to mars frame, I could just convert the state that begins the
            #  integration from heliocentric to mars-centric

            # Compute mass after first maneuver
            m_penultimate = self.unscaled_states[-3, -1] / np.exp(dv1_mag * 1000 / c.g0_ms2 / tc.isp_chemical)
            self.unscaled_states[-2, -1] = m_penultimate
            self.scaled_states[-2, -1] = m_penultimate / tc.mu

            # Set up integration of Lambert arc
            eom_type = 3  # 2BP only
            state_size = self.dim * 2
            step_size = (self.scaled_times[-1] - self.scaled_times[-2]) / tc.n_terminal_steps
            params, param_size = [], 0

            # Integrate Lambert arc
            traj = self.integrator.prop(self.scaled_states[-2].tolist(), self.scaled_times[-2:].tolist(), params,
                                        state_size, self.sizes['times'], param_size, tc.rtol, tc.atol, step_size,
                                        self.integrator_opts['step_type'], eom_type)

            # Include mass in the state history
            last_leg = np.hstack((np.array(traj[1:]), m_penultimate * np.ones((len(traj) - 1, 1))))

            # Add last leg to the trajectory history
            self.full_traj[-tc.n_terminal_steps:] = last_leg

            # Save final state of current leg
            self.scaled_states[-1] = self.full_traj[-1, 1:]
            self.unscaled_states[-1] = self.full_traj[-1, 1:] * self.state_scales

            # Compute second required delta V to get on to target orbit
            self.unscaled_states[-1, 3:6] += self.dv2
            self.scaled_states[-1, 3:6] = self.unscaled_states[-1, 3:6] / self.state_scales[3:6]

            # Compute mass after maneuver
            m_final = m_penultimate / np.exp(dv2_mag * 1000 / c.g0_ms2 / tc.isp_chemical)

            # Update final mass
            self.unscaled_states[-1, -1] = m_final
            self.scaled_states[-1, -1] = m_final / tc.mu
            self.full_traj[-1, -1] = m_final

            if change_frame:
                self.central_body = 'sun'

        else:
            # No maneuver
            pass

        # Dimensionalize states
        self.full_traj[:, 1:] *= self.state_scales

    def _update_params(self, thrust: [list, np.ndarray]) -> None:
        # create list of parameters to pass to integrator
        if self.spacecraft.engine_params['variable_power']:
            self.params = [*thrust,
                           self.spacecraft.dry_mass / tc.mu,
                           self.spacecraft.engine_params['power_reference'],
                           self.spacecraft.engine_params['power_min'],
                           self.spacecraft.engine_params['power_max'],
                           *self.spacecraft.engine_params['thrust_power_coef'],
                           *self.spacecraft.engine_params['isp_power_coef']]
        else:
            self.params = [c.g0_ms2 * self.spacecraft.engine_params['isp_s'] / tc.du * tc.tu,
                           self.spacecraft.dry_mass / tc.mu,
                           *thrust]

    def _update_sizes(self) -> None:
        """
        # Define lengths of input vectors to the integrator
        :return:
        """
        self.sizes = {'states': 2 * self.dim + 1,
                      'times': 2,
                      'param': 17 if self.spacecraft.engine_params['variable_power'] else 5}

    def evaluate(self, genome: neat.genome.DefaultGenome = None, config: neat.config.Config = None) -> float:
        if genome is not None:
            self.spacecraft.set_controller(genome, config)
        self.fitness = np.ones(tc.num_cases) * np.inf
        for i in range(tc.num_cases):
            y0, yf = missed_thrust.compute_bcs()
            self.propagate(y0, yf)

            # Check if integration was terminated early
            if self.flag == 1:
                self.fitness[i] = tc.big_penalty
                continue

            # Get final state before capture maneuver
            yf_actual = self.unscaled_states[-self.extra_nodes - 1, tc.ind_dim]
            yf_target = yf[tc.ind_dim[:-1]]

            # Calculate propellant mass ratio and final time ratio
            m_ratio = (self.unscaled_states[0, -1] - self.unscaled_states[-1, -1]) / self.unscaled_states[0, -1]
            t_ratio = self.unscaled_times[-1] / self.unscaled_times[-self.extra_nodes - 1]

            # Get fitness
            self.fitness[i], dr, dv = missed_thrust.traj_fit_func(yf_actual, yf_target, y0, m_ratio, t_ratio)

        # Calculate scalar fitness using one of the following methods
        rdo = False
        rbdo = False
        if tc.num_cases > 1:
            if rdo:
                # Robust Design Optimization
                alpha = 1.  # weight to favor mean vs std
                f_mean = np.mean(self.fitness)
                f_std = np.std(self.fitness)
                f = alpha * f_mean + (1 - alpha) * f_std

            # TODO Reliability-Based Design Optimization
            elif rbdo:
                f_mean = np.mean(self.fitness)
                f_constraint_violation = number_cases_outside_bounds / tc.num_cases
                alpha = 100.
                f = f_mean + alpha * f_constraint_violation

            # Mean
            else:
                f = np.mean(self.fitness)
        else:
            f = self.fitness[0]

        return -f

    def propagate(self, y0: np.ndarray, yf: np.ndarray, include_missed_thrust: bool = tc.missed_thrust_allowed) -> None:
        """
        Calls the internal functions with each segment of the trajectory. Start with the main leg, and optionally add a
        post-transfer capture maneuver.
        :param y0:
        :param yf:
        :param include_missed_thrust:
        :return:
        """
        self._propagate_main_leg(y0, yf, include_missed_thrust)
        self._post_transfer_capture(yf)
