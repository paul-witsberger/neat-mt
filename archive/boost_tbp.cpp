#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <algorithm>
#include <cmath>
#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <limits>

#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/math/special_functions/sign.hpp>

using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::python;

typedef std::vector<double> state_type;
typedef std::vector<double> Vec;
typedef std::vector<Vec> Mat;


// Define some helper functions
Vec operator*(const Mat &a, const Vec &x) {
	int i, j;
	int m = a.size();
	int n = x.size();
	Vec prod(m);

	for (i = 0; i < m; i++) {
		prod[i] = 0.;
		for (j = 0; j < n; j++) {
			prod[i] += a[i][j] * x[j];
		}
	}
	return prod;
}


Mat operator*(const Mat &a, const Mat &b) {
	int row, col;
	int n_rows = a.size();
	int n_cols = b.size();
	Mat prod(n_rows, vector<double>(b.at(0).size()));

	// Loop through and add each component
	for (row = 0; row < n_rows; ++row) {
		for (col = 0; col < n_cols; ++col) {
			for (size_t inner = 0; inner < b.size(); ++inner) {
				prod.at(row).at(col) += a.at(row).at(inner) * b.at(inner).at(col);
			}
		}
	}
	return prod;
}


Vec operator/(const Vec &a, const double b) {
	int i;
	int n_rows = a.size();
	Vec out;
	for (i = 0; i < n_rows; ++i) {
		out.push_back(a[i] / b);
	}
	return out;
}


void print(Vec &vec) {
	int i;
	int n_rows = vec.size();
	for (i = 0; i < n_rows; ++i) {
		cout << vec[i] << " \n"[i == (n_rows - 1)];
	}
}


void print(Mat &mat) {
	int i, j;
	int n_rows = mat.size();
	int n_cols = mat[0].size();
	for (i = 0; i < n_rows; ++i) {
		for (j = 0; j < n_cols; ++j) {
			cout << mat[i][j] << " \n"[j == (n_cols - 1)];
		}
	}
}


double mag3(const Vec &vec) {
	return pow(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2], 0.5);
}


Vec cross(const Vec &left, const Vec &right) {
	Vec product;
	double x, y, z;
	x = ((left[1] * right[2]) - (left[2] * right[1]));
	y = ((left[2] * right[0]) - (left[0] * right[2]));
	z = ((left[0] * right[1]) - (left[1] * right[0]));
	product = { x, y, z };
	return product;
}


Mat transpose(const Mat &mat) {
	Mat trans(mat);
	int i, j;
	for (i = 0; i < mat.size(); ++i) {
		for (j = 0; j < mat.at(i).size(); ++j) {
			trans[j][i] = mat[i][j];
		}
	}
	return trans;
}


Vec rotateVNCtoInertial3D(const Vec &vec, const Vec &state) {
	int i;
	Vec r_vec, v_vec, h_vec, v_hat, n_hat, c_hat, product;
	Mat dcm, dcm_t;
	// Split state into position and velocity vectors
	r_vec.assign(state.begin(), state.begin() + 3);
	v_vec.assign(state.begin() + 3, state.begin() + 6);
	// Calculate V, N, and C unit vectors
	v_hat = v_vec / mag3(v_vec);
	h_vec = cross(r_vec, v_vec);
	n_hat = h_vec / mag3(h_vec);
	c_hat = cross(v_hat, n_hat);
	// Construct DCM
	dcm.push_back(c_hat);
	dcm.push_back(v_hat);
	dcm.push_back(n_hat);
	dcm_t = transpose(dcm);
	// Apply rotation
	product = dcm_t * vec;
	return product;
}


// 2BP Equations of Motion
class eom2BPScaled {

public:
	eom2BPScaled(std::vector<double> param){
		
	}

	void operator()(const state_type &x, state_type &dxdt, const double /* t */) {
		// calculate auxiliary variables
		double r = sqrt((x[0] * x[0]) + (x[1] * x[1]) + (x[2] * x[2]));
		double r3 = r * r * r;
		
		// EOMs (3 velocity, 3 accel w/ gravity only)
		dxdt[0] = x[3];
		dxdt[1] = x[4];
		dxdt[2] = x[5];
		dxdt[3] = -1 / r3 * x[0];
		dxdt[4] = -1 / r3 * x[1];
		dxdt[5] = -1 / r3 * x[2];
	}
};

class eom2BPScaled_variable_power {
	double g0, gm, mdry, power_reference, power_min, power_max;
	std::vector<double> thrust_body, thrust_coef, isp_coef;

public:
	eom2BPScaled_variable_power(std::vector<double> param) {
		g0 = 9.80665;
		mdry = param[0];
		thrust_body.assign({ param[1], param[2] , param[3] });
		power_reference = param[4];
		power_min = param[5];
		power_max = param[6];
		thrust_coef.assign({ param[7], param[8], param[9], param[10], param[11] });
		isp_coef.assign({ param[12], param[13], param[14], param[15], param[16] });
	}

	double thrust_fcn(double power) {
		return (thrust_coef[4] * pow(power, 4) + thrust_coef[3] * pow(power, 3) + thrust_coef[2] * pow(power, 2) + thrust_coef[1] * power + thrust_coef[0]);
	}

	double isp_fcn(double power) {
		return (isp_coef[4] * pow(power, 4) + isp_coef[3] * pow(power, 3) + isp_coef[2] * pow(power, 2) + isp_coef[1] * power + isp_coef[0]);
	}

	void operator()(const state_type &x, state_type &dxdt, const double /* t */) {
		// calculate auxiliary variables
		double r, r3, thrust, ve;
		r = sqrt((x[0] * x[0]) + (x[1] * x[1]) + (x[2] * x[2]));
		r3 = r * r * r;
		thrust = sqrt((thrust_body[0] * thrust_body[0]) + (thrust_body[1] * thrust_body[1]) + (thrust_body[2] * thrust_body[2]));
		std::vector<double> thrust_inertial;

		// check if mass has dropped below minimum allowable value or thrust is zero
		if (x[6] <= mdry || thrust <= 0.0) {
			thrust_inertial.assign({ 0.0, 0.0, 0.0 });
			thrust = 0.0;
			ve = 1000.0;
		}

		// otherwise, calculate available power and adjust thrust accordingly
		else {
			// check if availble power is below minimum usable threshold
			double au_to_km = 149597870.7;
			double power_available = power_reference * pow((au_to_km / r), 2);
			if (power_available < power_min) {
				thrust_inertial.assign({ 0.0, 0.0, 0.0 });
				thrust = 0.0;
				ve = 1000.0;
			}

			// otherwise, see if thrust vector needs to be updated
			else {
				double rotation_angle, isp, eta, thrust_produced, power_used;
				// rotate thrust from spacecraft frame into inertial frame
				rotation_angle = atan2(x[4], x[3]);
				thrust_inertial.assign({ cos(rotation_angle) * thrust_body[0] - sin(rotation_angle) * thrust_body[1],
									 sin(rotation_angle) * thrust_body[0] + cos(rotation_angle) * thrust_body[1],
									 0.0});

				power_used = std::min(power_available, power_max);
				thrust_produced = thrust_fcn(power_used);
				isp = isp_fcn(power_used);
				ve = g0 * isp;
				if (thrust_produced < thrust) {
					// calculate new Isp based on available power
					eta = thrust_produced / thrust;
					thrust_inertial[0] *= eta;
					thrust_inertial[1] *= eta;
					thrust_inertial[2] *= eta;
				}
			}
		}

		// EOMs (3 velocity, 3 accel w/ grav and thrust, 1 mass)
		dxdt[0] = x[3];
		dxdt[1] = x[4];
		dxdt[2] = x[5];
		dxdt[3] = -1 / r3 * x[0] + thrust_inertial[0] / x[6];
		dxdt[4] = -1 / r3 * x[1] + thrust_inertial[1] / x[6];
		dxdt[5] = -1 / r3 * x[2] + thrust_inertial[2] / x[6];
		dxdt[6] = -thrust * 1000.0 / ve;
	}
};

class eom2BPScaled_constant_power {
	double gm, mdry, ve, thrust_mag;
	std::vector<double> thrust_body;

public:
	eom2BPScaled_constant_power(std::vector<double> param) {
		ve = param[0];
		mdry = param[1];
		thrust_body.assign({ param[2], param[3] , param[4] });
		thrust_mag = sqrt((thrust_body[0] * thrust_body[0]) + (thrust_body[1] * thrust_body[1]) + (thrust_body[2] * thrust_body[2]));
	}

	void operator()(const state_type &x, state_type &dxdt, const double /* t */) {
		double r, r3;
		std::vector<double> thrust_inertial;

		// calculate radius
		r = sqrt((x[0] * x[0]) + (x[1] * x[1]) + (x[2] * x[2]));
		r3 = r * r * r;

		// check if mass has dropped below minimum allowable value or thrust is zero
		if (x[6] <= mdry || thrust_mag <= std::numeric_limits<double>::min()) {
			thrust_inertial.assign({ 0.0, 0.0, 0.0 });
		}

		// otherwise, rotate thrust vector to inertial frame
		else {
			// double rotation_angle = atan2(x[4], x[3]);
			// thrust_inertial.assign({ cos(rotation_angle) * thrust_body[0] - sin(rotation_angle) * thrust_body[1],
								 	 // sin(rotation_angle) * thrust_body[0] + cos(rotation_angle) * thrust_body[1],
								 	 // 0.0});

			// TODO need to properly compute psi, theta, and phi to convert from VNC to inertial
			// double psi, theta, phi;
			// psi = 0.0;
			// theta = 0.0;
			// phi = 0.0;
			// thrust_inertial.assign({ (cos(psi) * cos(phi) - sin(psi) * sin(phi) * cos(theta)) * thrust_body[0] + (cos(psi) * sin(phi) + sin(psi) * cos(theta) * cos(phi)) * thrust_body[1] + (sin(psi) * sin(theta)) * thrust_body[2] },
			// 	{ -(sin(psi) * cos(phi) - cos(psi) * sin(phi) * cos(theta)) * thrust_body[0] - (sin(psi) * sin(phi) - cos(psi) * cos(theta) * cos(phi)) * thrust_body[1] + (cos(psi) * sin(theta)) * thrust_body[2] },
			// 	{ (sin(theta) * sin(phi)) * thrust_body[0] - (sin(theta) * cos(phi)) * thrust_body[1] + (cos(theta)) * thrust_body[2] });
			thrust_inertial = rotateVNCtoInertial3D(thrust_body, x);
		}

		// EOMs (3 velocity, 3 accel w/ grav and thrust, 1 mass)
		dxdt[0] = x[3];
		dxdt[1] = x[4];
		dxdt[2] = x[5];
		dxdt[3] = -1.0 / r3 * x[0] + thrust_inertial[0] / x[6];
		dxdt[4] = -1.0 / r3 * x[1] + thrust_inertial[1] / x[6];
		dxdt[5] = -1.0 / r3 * x[2] + thrust_inertial[2] / x[6];
		dxdt[6] = -thrust_mag * 1000.0 / ve;
		if (isnan(dxdt[6])) {
			cout << "dmdt is nan" << endl;
			cout << "thrust_mag = " << thrust_mag << endl;
			cout << "ve = " << ve << endl;
			cout << "m = " << x[6] << endl;
		}
	}
};

class eom2BPScaled_stm {

public:
	eom2BPScaled_stm(std::vector<double> param) {
		
	}

	void operator()(const state_type &x, state_type &dxdt, const double /* t */) {
		double r, r3, r5, A[6][6], A21[3][3], stm[6][6], dstmdt[6][6];
		int counter;

		// calculate radius
		r = sqrt((x[0] * x[0]) + (x[1] * x[1]) + (x[2] * x[2]));
		r3 = r * r * r;
		r5 = r3 * r * r;

		// EOMs (3 velocity, 3 accel w/ grav and thrust)
		double xx, yy, zz, vx, vy, vz;
		xx = x[0]; yy = x[1]; zz = x[2]; vx = x[3]; vy = x[4]; vz = x[5];
		dxdt[0] = vx;
		dxdt[1] = vy;
		dxdt[2] = vz;
		dxdt[3] = -1.0 / r3 * xx;
		dxdt[4] = -1.0 / r3 * yy;
		dxdt[5] = -1.0 / r3 * zz;

		// State Transition Matrix
		counter = 6;
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 6; j++) {
				stm[i][j] = x[counter];
			}
		}
		A21[0][0] = -1.0 / r3 + 3 * xx * xx / r5;
		A21[0][1] = 3 * xx * yy / r5;
		A21[0][2] = 3 * xx * zz / r5;
		A21[1][0] = 3 * yy * xx / r5;
		A21[1][1] = -1.0 / r3 + 3 * yy * yy / r5;
		A21[1][2] = 3 * yy * zz / r5;
		A21[2][0] = 3 * zz * xx / r5;
		A21[2][1] = 3 * zz * yy / r5;
		A21[2][2] = -1.0 / r3 + 3 * zz * zz / r5;
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 6; j++) {
				A[i][j] = 0.0;
			}
		}
		for (int i = 3; i < 6; i++) {
			for (int j = 0; j < 3; j++) {
				A[i][j] = A21[i-3][j];
			}
		}
		A[0][3] = 1.0;
		A[1][4] = 1.0;
		A[2][5] = 1.0;

		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 6; j++) {
				dstmdt[i][j] = 0.0;
			}
		}

		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 6; j++) {
				for (int k = 0; k < 6; k++) {
					dstmdt[i][j] = A[i][j] * stm[k][j];
				}
			}
		}

		counter = 6;
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 6; j++) {
				dxdt[counter] = dstmdt[i][j];
				counter += 1;
			}
		}
	}
};

class myexception : public exception {
	virtual const char* what() const throw()
	{
		return "gotcha - custom error in C++";
	}
};

// Used to pull states out of the integration
struct getStateAndTime {
	std::vector<state_type> &m_states;
	std::vector<double> &m_times;

	getStateAndTime(std::vector<state_type> &states, std::vector<double> &times) : m_states(states), m_times(times) {}

	void operator()(const state_type &x, double t) {
		// Check if energy is too high or orbit is too small
		/*double r = sqrt((x[0] * x[0]) + (x[1] * x[1]) + (x[2] * x[2]));
		if (r < 10000.0) {
			myexception ex;
			throw ex;
		}
		double v = sqrt((x[3] * x[3]) + (x[4] * x[4]) + (x[5] * x[5]));
		double eps = (v * v) / 2 - 398600.0 / r;
		if (eps > 2.0) {
			myexception ex;
			throw ex;
		}*/

		// Store new values
		m_states.push_back(x);
		m_times.push_back(t);
	}
};

struct TBP {
	// Put function into format python can understand and the propagate
	boost::python::list propPy(boost::python::list &ic, boost::python::list &ti, boost::python::list &p, int state_dim, int t_dim,
							   int p_dim, double rtol, double atol, double step_size, int integrator_type, int eom_type) {

		typedef std::vector<double> state_type;
		std::vector<state_type> statesAndTimes;

		state_type IC(state_dim, 0);
		state_type t(t_dim, 0);
		std::vector<double> param(p_dim, 0);

		// Transform inputs
		for (int i = 0; i < len(ic); ++i) {
			IC[i] = boost::python::extract<double>(ic[i]);
		}

		for (int i = 0; i < len(ti); ++i) {
			t[i] = boost::python::extract<double>(ti[i]);
		}

		for (int i = 0; i < len(p); ++i) {
			param[i] = boost::python::extract<double>(p[i]);
		}

		// Propagate
		statesAndTimes = prop(IC, t, param, state_dim, t_dim, p_dim, rtol, atol, step_size, integrator_type, eom_type);

		// Create python list from data to return
		return toTwoDimPythonList(statesAndTimes);
	}

	// Propagation function
	std::vector<vector<double >> prop(vector<double> ic, vector<double> t, vector<double> param, int state_dim, int t_dim,
									  int p_dim, double rtol, double atol, double step_size, int integrator_type, int eom_type) {
		using namespace std;
		using namespace boost::numeric::odeint;

		typedef std::vector<double> state_type;
		std::vector<state_type> statesAndTimes;

		// Set vectors intermediate steps during integration
		std::vector<double> tOut;
		std::vector<state_type> statesOut;

		// Determine step size (forward or backward) and set initial step size
		double h = t[1] > t[0] ? step_size : -step_size;

		// Set integrator type -> Currently set at rk78
		double relTol = rtol;
		double absTol = atol;
		typedef runge_kutta_fehlberg78<state_type> rk78;
		auto stepper = make_controlled<rk78>(absTol, relTol);
		
		// Create eom to integrate
		if (eom_type == 0) {
			eom2BPScaled_constant_power eom(param);
			if (integrator_type == 0) {
				size_t steps = integrate_const(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			}
			else if (integrator_type == 1) {
				size_t steps = integrate_adaptive(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			}
		} else if (eom_type == 1) {
			eom2BPScaled_variable_power eom(param);
			if (integrator_type == 0) {
				size_t steps = integrate_const(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			}
			else if (integrator_type == 1) {
				size_t steps = integrate_adaptive(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			}
		}
		else if (eom_type == 2) {
			eom2BPScaled_stm eom(param);
			if (integrator_type == 0) {
				size_t steps = integrate_const(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			}
			else if (integrator_type == 1) {
				size_t steps = integrate_adaptive(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			}
		}
		else if (eom_type == 3) {
			eom2BPScaled eom(param);
			if (integrator_type == 0) {
				size_t steps = integrate_const(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			}
			else if (integrator_type == 1) {
				size_t steps = integrate_adaptive(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			}
		}

		// Insert IC into list of state vectors
		statesAndTimes.resize(statesOut.size());

		for (int i = 0; i < statesAndTimes.size(); i++) {
			statesAndTimes[i].resize(ic.size() + 1);
			for (int j = 0; j < statesAndTimes[i].size(); j++) {
				if (j == 0) {
					statesAndTimes[i][j] = tOut[i];
				}
				else {
					statesAndTimes[i][j] = statesOut[i][j - 1];
				}
			}
		}
		
		return statesAndTimes;
	}

	template<class T>
	boost::python::list toPythonList(std::vector<T> vector) {
		typename std::vector<T>::iterator iter;
		boost::python::list list;
		for (iter = vector.begin(); iter != vector.end(); ++iter) {
			list.append(*iter);
		}
		return list;
	}

	template<class T>
	boost::python::list toTwoDimPythonList(std::vector<std::vector<T> > vector) {
		typename std::vector<std::vector<T> >::iterator iter;

		boost::python::list list;
		for (iter = vector.begin(); iter != vector.end(); ++iter) {
			list.append(toPythonList(*iter));
		}
		return list;
	}
};

BOOST_PYTHON_MODULE(boost_tbp) {
	class_<TBP>("TBP")
		.def("prop", &TBP::propPy);
};

int main() {
	Vec vec_vnc = {1, 0, 0};
	Vec state = {0, 1e8, 0, -29, 0, 0};
	Vec vec_i = rotateVNCtoInertial3D(vec_vnc, state);

	cout << "VNC:\n";
	print(vec_vnc);
	cout << "Inertial:\n";
	print(vec_i);

	cout << endl;
	return 0;
}
