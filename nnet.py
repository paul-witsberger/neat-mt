import numpy as np
import orbit_util as ou
from traj_config import n_dim, input_frame, input_indices, scales_in, scales_out, gm, out_node_scales,\
    angle_choices, use_multiple_angle_nodes
from numba import jit


@jit(nopython=True, cache=False)
def scale_inputs(inputs: np.ndarray, scales_in: np.ndarray = scales_in) -> np.ndarray:
    return inputs / scales_in


@jit(nopython=True, cache=False)
def scale_outputs(outputs: np.ndarray, old_scales: np.ndarray = out_node_scales, new_scales: np.ndarray = scales_out,
                  nd: int = n_dim):
    so = old_scales
    sn = new_scales
    outputs_scaled = (outputs[0] - so[:nd, 0]) * (sn[:nd, 1] - sn[:nd, 0]) / (so[:nd, 1] - so[:nd, 0]) + sn[:nd, 0]
    return outputs_scaled


class Neurocontroller:

    def __init__(self, scales_in, scales_out, n_in=13, n_hid=10, n_out=2, neat_net=None, init_from_neat_net=False):
        if init_from_neat_net:
            assert neat_net is not None
            self.net = neat_net
            self.scales_in = scales_in
            self.scales_out = scales_out
            self.was_init_from_neat_net = True
        else:
            # Determine number of layers and initialize W and b
            if type(n_hid) == tuple:
                self.W = [[] for _ in range(len(n_hid))]
                self.b = [[] for _ in range(len(n_hid))]
                self.n_layers = len(n_hid) + 1
            else:
                self.W = [[], []]
                self.b = [[], []]
                self.n_layers = 2
            self.n_in = n_in
            self.n_hid = n_hid
            self.n_out = n_out

            # Initialize weights and biases with random normal values
            mu, sigma = 0, 0.1
            if self.n_layers > 2:
                for i in range(self.n_layers):
                    if i == 0:
                        self.W[i] = np.random.normal(mu, sigma, (n_hid[i], n_in))
                        self.b[i] = np.random.normal(mu, sigma, (n_hid[i], 1))
                    elif i == self.n_layers-1:
                        self.W[i] = np.random.normal(mu, sigma, (n_out, n_hid[i-1]))
                        self.b[i] = np.random.normal(mu, sigma, (n_out, 1))
                    else:
                        self.W[i] = np.random.normal(mu, sigma, (n_hid[i], n_hid[i-1]))
                        self.b[i] = np.random.normal(mu, sigma, (n_hid[i], 1))
            else:
                self.W[0] = np.random.normal(mu, sigma, (n_hid, n_in))
                self.b[0] = np.random.normal(mu, sigma, (n_hid, 1))
                self.W[1] = np.random.normal(mu, sigma, (n_out, n_hid))
                self.b[1] = np.random.normal(mu, sigma, (n_out, 1))

            assert(scales_in.shape[0] == n_in)
            assert(scales_out.shape[0] == n_out)
            self.scales_in = scales_in
            self.scales_out = scales_out
            self.was_init_from_neat_net = False

    @staticmethod
    def activation(W: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.tanh(np.sum((np.matmul(W, x), b), axis=0))

    @staticmethod
    def scaleInputs(inputs: np.ndarray) -> np.ndarray:
        inputs_scaled = inputs / scales_in
        return inputs_scaled

    @staticmethod
    def scaleOutputs(outputs: np.ndarray) -> np.ndarray:
        so = out_node_scales  # old scales
        sn = scales_out       # new scales
        outputs_scaled = np.array([(out - so[i, 0]) / (so[i, 1] - so[i, 0]) * (sn[i, 1] - sn[i, 0]) + sn[i, 0]
                                   for i, out in enumerate(outputs[0])])
        return outputs_scaled

    def getThrustVec(self, state: np.ndarray) -> np.ndarray:
        state_coe = ou.inertial_to_keplerian(state)
        state_scaled = self.scaleInputs(state_coe)
        out = self.forwardPass(state_scaled)
        angle, throttle = self.scaleOutputs(out)
        thrust = np.hstack((np.cos(angle), np.sin(angle))) * throttle
        angle_v = np.arctan2(state[3], state[2])
        dcm = np.array([[np.cos(angle_v), -np.sin(angle_v)], [np.sin(angle_v), np.cos(angle_v)]])
        thrust = np.matmul(dcm, thrust)
        return thrust

    @staticmethod
    def get_angle(angle_choice: np.ndarray) -> float:
        return angle_choices[angle_choice]

    def get_thrust_vec_neat(self, state: np.ndarray) -> np.ndarray:
        _state = np.empty_like(state)
        # Convert the state to the desired frame with the appropriate number of dimensions (starts in inertial frame)
        if n_dim == 2:
            # if input_frame == 'kep':  # 2D Keplerian
            curr = ou.inertial_to_keplerian_2d(state[:n_dim * 2], gm=gm)
            targ = ou.inertial_to_keplerian_2d(state[n_dim * 2:n_dim * 4], gm=gm)
        else:
            assert input_frame == 'kep' or input_frame == 'mee'
            if input_frame == 'kep':  # 3D Keplerian
                # TODO is there a way to avoid converting inertial to keplerian each step?
                curr = ou.inertial_to_keplerian_3d(state[:n_dim * 2], gm=gm)
                targ = ou.inertial_to_keplerian_3d(state[n_dim * 2:n_dim * 4], gm=gm)
            else:  # 3D MEE  # TODO this is wrong - ine2kep first, then kep2mee
                curr = ou.inertial_to_keplerian_3d(ou.keplerian_to_mee_3d(state[:n_dim * 2]), gm=gm)
                targ = ou.inertial_to_keplerian_3d(ou.keplerian_to_mee_3d(state[n_dim * 2:n_dim * 4]), gm=gm)
        _state[:n_dim * 2] = curr
        _state[n_dim * 2:n_dim * 4] = targ
        _state[-2:] = state[-2:]

        # Scale inputs before passing them to the network
        # state_scaled = self.scaleInputs(_state)
        state_scaled = scale_inputs(_state)

        # Choose the desired state elements to pass to the network
        if input_indices is not None:
            state_scaled = state_scaled[input_indices]

        # Get network activations
        out = np.array(self.net.activate(state_scaled)).reshape((1, -1))

        # Convert outputs from angle and throttle to thrust vector
        if n_dim == 2:
            if use_multiple_angle_nodes:  # categorical classification
                angle_choice = np.argmax(out[0, :-1])
                alpha = self.get_angle(angle_choice)
                throttle = out[0, -1]
            else:  # regression
                # alpha, throttle = self.scaleOutputs(out)
                alpha, throttle = scale_outputs(out)
            # thrust = np.hstack((np.cos(alpha), np.sin(alpha), 0)) * throttle
            thrust = np.empty(2, np.float64)
            # thrust[0] = np.cos(alpha) * throttle  # velocity
            # thrust[1] = np.sin(alpha) * throttle  # anti-co-normal
            # thrust[2] = 0.                        # normal
            thrust[0] = np.cos(alpha) * throttle    # velocity
            thrust[1] = np.sin(alpha) * throttle   # co-normal
        else:
            alpha, beta, throttle = self.scaleOutputs(out)
            # TODO use this info to properly apply thrust
            thrust = np.hstack((np.cos(alpha) * np.cos(beta), np.sin(alpha) * np.cos(beta), np.sin(beta)))
            # See 532 Notes, Page JS_3Dex 2-3 for reference on VNC frame

        return thrust

    def forwardPass(self, state: np.ndarray) -> np.ndarray:
        a = [[] for _ in range(self.n_layers)]
        for layer in range(self.n_layers):
            if layer == 0:
                x_in = np.array([state]).transpose()
            else:
                x_in = a[layer-1]
            a[layer] = self.activation(self.W[layer], self.b[layer], x_in)
        return a[-1].transpose()

    def setWeights(self, Wb_new: np.ndarray):
        W_new = [[] for _ in range(self.n_layers)]
        b_new = [[] for _ in range(self.n_layers)]
        assert(self.n_layers == 2)  # only supports 2 layers currently
        W_new[0] = np.reshape(Wb_new[:(self.n_in*self.n_hid)], (self.n_hid, self.n_in))
        b_new[0] = np.reshape(Wb_new[(self.n_in*self.n_hid):(self.n_in*self.n_hid+self.n_hid)], (self.n_hid, 1))
        W_new[-1] = np.reshape(Wb_new[-(self.n_hid*self.n_out+self.n_out):-self.n_out], (self.n_out, self.n_hid))
        b_new[-1] = np.reshape(Wb_new[-self.n_out:], (self.n_out, 1))
        for i, w in enumerate(self.W):
            self.W[i] = W_new[i]
            self.b[i] = b_new[i]

    @staticmethod
    def init_from_neat_net(net, scales_in, scales_out):
        return Neurocontroller(scales_in, scales_out, 0, 0, 0, net, True)
