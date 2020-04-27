import orbit_util as ou
import constants as c
import neatfast as neat
import pickle
import os
import traj_config as tc
from nnet import Neurocontroller
import boost_tbp
import big_idea
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
    'thrust_n': 1.2,
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
                 genome: neat.genome.DefaultGenome = None):
        assert state is None or (len(state) == 4 and dim == 2) or (len(state) == 6 and dim == 3)
        self._state = state if state is not None else [0.] * 6
        self._mass = mass if mass is not None else 0.
        self.engine_params = engine_params if engine_params is not None else _const_thrust_engine_params
        self.dry_mass = tc.m_dry
        self.set_controller(genome=genome)

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

    def set_controller(self, genome: neat.genome.DefaultGenome = None, fname: str = 'winner_feedforward'):
        """
        Create a NEAT network either from a given genome or from a file, and then assign the thrust function as the
        output of the network.
        :param genome:
        :param fname:
        :return:
        """
        if genome is not None:
            net = genome
        else:
            with open(fname, 'rb') as f:
                net = pickle.load(f)
        config_path = os.path.join(os.path.dirname(__file__), 'config_feedforward')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                             neat.DefaultStagnation, config_path)
        neat_net = neat.nn.FeedForwardNetwork.create(net, config)
        nc = Neurocontroller.init_from_neat_net(neat_net, tc.scales_in, tc.scales_out)
        self.controller = nc.get_thrust_vec_neat


class Trajectory:
    def __init__(self, central_body: str = 'sun', frame: str = 'kep', dim: int = tc.n_dim,
                 genome: neat.genome.DefaultGenome = None):
        self.spacecraft = Spacecraft(dim=dim, state=init_sc_state, genome=genome)
        self._initialize_integrator_opts()
        self._init_times()
        self.unscaled_states = init_sc_state if init_sc_state is not None else []
        self.scaled_states = []
        self._dim = dim
        self._frame = frame
        self._central_body = central_body
        self.traj_config = tc
        self._update_sizes()
        self._update_params(np.zeros(3, float))

    def _initialize_integrator_opts(self):
        """
        Helper function to set up integrator and related variables.
        :return:
        """
        # Define scales to nondimensionalize state for integration
        du = max(tc.a0_max, tc.af_max)
        tu = (du ** 3 / tc.gm) ** 0.5
        mu = tc.m0
        self.fu = mu * du / tu / tu
        if self.dim == 2:
            self.state_scales = [du, du, du / tu, du / tu, mu]
        else:
            self.state_scales = [du, du, du, du / tu, du / tu, du / tu, mu]
        self.du, self.tu, self.mu = du, tu, mu
        self.integrator_opts = {'step_type': 0 if tc.fixed_step else 1,
                                'eom_type': 1 if tc.variable_power else 0}
        self.integrator = boost_tbp.TBP()

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
        self._central_body = value

    def _init_times(self, spacing: str = 'linear'):
        if spacing == 'linear':
            t = np.linspace(tc.t0, tc.tf, tc.num_nodes)
        elif spacing == 'power':
            t = np.power(np.linspace(0, 1, tc.num_nodes), 3 / 2) * (tc.tf - tc.t0) + tc.t0
        else:
            raise(RuntimeError, 'Invalid spacing type for time vector')
        t = np.append(t, t[-1])
        self.unscaled_times = t
        self.scaled_times = t / self.tu

    # TODO - Implement propagate function
    #   1. Initialize state, time, mass, and full state matrices
    #   2. Main trajectory integration
    #   3. Post-transfer capture manuever
    def propagate(self, y0, yf, include_missed_thrust: bool = tc.missed_thrust_allowed,
                  save_full_traj: bool = False):
        self._propagate_main_leg(y0, yf, include_missed_thrust, save_full_traj)
        self._post_transfer_capture(yf)

    def _propagate_main_leg(self, y0, yf, include_missed_thrust: bool, save_full_traj: bool):
        # Get the indices of the legs that will experience missed thrust
        if tc.missed_thrust_allowed and include_missed_thrust:
            miss_ind = big_idea.calculate_missed_thrust_events(self.times * self.tu)
        else:
            miss_ind = np.array([]).astype(int)

        full_traj = []
        for i in range(len(self.times) - 2):
            # Check if orbital energy is within reasonable bounds - terminate integration if not
            r = np.linalg.norm(y[i, :3])
            eps = (np.linalg.norm(y[i, 3:6]) ** 2 / 2 - tc.gm / r)
            if (eps > tc.max_energy or eps < tc.min_energy) and not save_full_traj:
                return np.array(0), 0, 0, 0, 0

            # Fixed step integrator step size
            step_size = (self.times[i + 1] - self.times[i]) / tc.n_steps

            # Get ratio of remaining propellant mass and elapsed time ratio
            mass_ratio = (y[i, -1] - self.spacecraft.dry_mass) / (y[0, -1] - self.spacecraft.dry_mass)
            time_ratio = self.times[i] / self.times[-2]

            # Check if i is supposed to miss thrust for this segment
            if i in miss_ind or y[i, -1] <= self.spacecraft.dry_mass + 0.01:
                # Missed-thrust event, or no propellant remaining
                thrust = np.array([0, 0, 0], float)
            else:
                # query NN to get next thrust vector
                thrust = self.spacecraft.controller(np.hstack((y[i, tc.ind_dim], yf[tc.ind_dim[:-1]], mass_ratio,
                                                               time_ratio)), float) \
                         * self.spacecraft.engine_params['thrust_max_n'] / self.fu

            self._update_params(thrust)

            # propagate from the current state until the next time step
            traj = self.integrator.prop(self.scaled_states[i].aslist(), self.times[i:i+2].aslist(), self.params,
                                        self.sizes['states'], self.sizes['times'], self.sizes['param'], tc.rtol,
                                        tc.atol, step_size, self.integrator_opts['step_type'],
                                        self.integrator_opts['eom_type'])

            # save full trajectory including intermediate states
            full_traj.extend(traj[1:])

            # save final state of current leg
            y[i + 1] = np.array(traj[-1])[1:] * self.integrator_opts['scales']

    def _post_transfer_capture(self):
        pass

    def _update_params(self, thrust):
        # create list of parameters to pass to integrator
        if self.spacecraft.engine_params['variable_power']:
            self.params = [*thrust,
                           self.spacecraft.dry_mass / self.mu,
                           self.spacecraft.engine_params['power_reference'],
                           self.spacecraft.engine_params['power_min'],
                           self.spacecraft.engine_params['power_max'],
                           *self.spacecraft.engine_params['thrust_power_coef'],
                           *self.spacecraft.engine_params['isp_power_coef']]
        else:
            self.params = [*thrust,
                           self.spacecraft.dry_mass / self.mu,
                           c.g0_ms2 * self.spacecraft.engine_params['isp_s'] / self.du * self.tu]

    def _update_sizes(self):
        """
        # Define lengths of input vectors to the integrator
        :return:
        """
        self.sizes = {'states': 2 * self.dim,
                      'times': 2,
                      'param': 17 if self.spacecraft.engine_params['variable_power'] else 5}

    # TODO update frame conversion formats - maybe make a wrapper that interprets the strings?
    def convert_frame_to(self, new_frame):
        if not new_frame == self.frame:
            self._state = _frames_conversions[self._frame + new_frame + str(self._dim)](self._state, self._central_body)
            self._frame = new_frame

    def evaluate(self, save_full_traj: bool = False):
        f = np.ones(tc.num_cases) * np.inf
        for i in range(tc.num_cases):
            y0, yf = big_idea.make_new_bcs()
            self.propagate(y0, yf, save_full_traj)
            raise (NotImplementedError)
