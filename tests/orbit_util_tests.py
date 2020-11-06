import unittest
from orbit_util import np, keplerian_to_mee_3d, mee_to_keplerian_3d, keplerian_to_inertial_3d, inertial_to_keplerian_3d,\
    gamma_from_r_v


class MyTestCase(unittest.TestCase):
    def test_3d_frame_conversions(self):
        state_k = np.array([150e6, 0.5, 2, 2, 2, 2])
        state_m = keplerian_to_mee_3d(state_k)
        state_k2 = mee_to_keplerian_3d(state_m)
        state_i = keplerian_to_inertial_3d(state_k).ravel()
        state_k3 = inertial_to_keplerian_3d(state_i)
        state_m2 = keplerian_to_mee_3d(state_k3)
        self.assertTrue(np.allclose(state_m2, state_m))

    def test_gamma_from_r_v(self):
        r = np.array([100000, 0, 0])
        v = np.array([1, 4, 0])
        gamma = gamma_from_r_v(r, v)
        self.assertEqual(gamma, 0.24497866312686423)


if __name__ == '__main__':
    unittest.main()
