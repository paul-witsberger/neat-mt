import orbit_util as ou
import constants as c
import neatfast as neat
import pickle
import os
import traj_config
from nnet import Neurocontroller

_frames_conversions = {
    'kepmee3': ou.keplerian_to_mee_3d,
    'kepinr3': ou.keplerian_to_inertial_3d,
    'meekep3': ou.mee_to_keplerian_3d,
    'inrkep3': ou.inertial_to_keplerian_3d,
    'kepinr2': ou.keplerian_to_inertial_2d,
    'inrkep2': ou.inertial_to_keplerian_2d
}

_default_engine_params = {
    'isp_s': 2780,
    'thrust_n': 1.2,
    'variable_power': False
}

class Spacecraft:
    def __init__(self, dim: float = 3, state: list = None, mass: float = None, engine_params: dict = None,
                 genome: neat.genome.DefaultGenome = None):
        assert state is None or (len(state) == 4 and dim == 2) or (len(state) == 6 and dim == 3)
        self._state = state if state is not None else [0.] * 6
        self._mass = mass if mass is not None else 0.
        self.engine_parameters = engine_params if engine_params is not None else {'isp': 2780, 'thrust': 1.2}
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

    def set_controller(self, genome=None, fname: str = 'winner_feedforward'):
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
        nc = Neurocontroller.init_from_neat_net(neat_net, traj_config.scales_in, traj_config.scales_out)
        self.controller = nc.get_thrust_vec_neat


class Trajectory:
    def __init__(self, central_body: str = 'sun', frame: str = 'kep', dim: float = 3, time: float = 0.,
                 init_sc_state: list = None, genome: neat.genome.DefaultGenome = None):
        self.spacecraft = Spacecraft(dim=dim, state=init_sc_state, genome=genome)
        self.time = time
        self._dim = 3
        self._frame = frame
        self._central_body = central_body

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

    def convert_frame_to(self, new_frame):
        if not new_frame == self.frame:
            self._state = _frames_conversions[self._frame + new_frame + str(self._dim)](self._state, self._central_body)
            self._frame = new_frame

    # TODO implement propagate function
    def propagate(self, integration_time: list, include_missed_thrust: bool = True):
        # Define lengths of vectors for C++
        state_size, time_size = len(y0), 2
        param_size = 17 if variable_power else 5

        # Create placeholder matrix for trajectory - these are dimensionalized states
        y = np.zeros((len(ti), state_size))

        # Assign initial condition
        y[0] = y0

        # Create 2-body integrator object
        tbp = boost_tbp.TBP()

        # Define flag for missed thrust
        if missed_thrust_allowed:
            miss_ind = calculate_missed_thrust_events(ti * tu)
        else:
            miss_ind = np.array([]).astype(int)

        # Scales for nondim integration
        scales = np.array([du, du, du, du / tu, du / tu, du / tu, mu])

        # Main loop
        # Choose type of integrator and equations of motion
        if fixed_step:
            integrator_type = 0  # 0 fixed step
        else:
            integrator_type = 1  # adaptive step
        eom_type = 0  # 0 constant power, 1 variable power, 2 state transition matrix
        full_traj = []
