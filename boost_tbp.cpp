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

// 2BP Equations of Motion
class eom2BP {
	double gm, ve, mdry;
	std::vector<double> thrust;

public:
	eom2BP(std::vector<double> param){
		gm = param[0];
		ve = param[1];
		mdry = param[2];
		thrust.assign({ param[3], param[4] , param[5] });
	}

	void operator()(const state_type &x, state_type &dxdt, const double /* t */) {
		// calculate auxiliary variables
		double r = sqrt((x[0] * x[0]) + (x[1] * x[1]) + (x[2] * x[2]));
		double r3 = r * r * r;
		double tmag = sqrt((thrust[0] * thrust[0]) + (thrust[1] * thrust[1]) + (thrust[2] * thrust[2]));
		// check if mass has dropped below minimum allowable value
		if (x[6] <= mdry) {
			thrust.assign({ 0.0, 0.0, 0.0 });
			tmag = 0.0;
		}
		// EOMs (3 velocity, 3 accel w/ grav and thrust, 1 mass)
		dxdt[0] = x[3];
		dxdt[1] = x[4];
		dxdt[2] = x[5];
		dxdt[3] = -gm / r3 * x[0] + thrust[0] / x[6];
		dxdt[4] = -gm / r3 * x[1] + thrust[1] / x[6];
		dxdt[5] = -gm / r3 * x[2] + thrust[2] / x[6];
		dxdt[6] = -tmag * 1000.0 / ve;
	}
};

class eom2BPScaled_variable_power {
	double g0, gm, mdry, power_reference, power_min, power_max;
	std::vector<double> thrust_body, thrust_coef, isp_coef;

public:
	eom2BPScaled_variable_power(std::vector<double> param) {
		g0 = 9.80665;
		gm = param[0];
		mdry = param[1];
		thrust_body.assign({ param[2], param[3] , param[4] });
		power_reference = param[5];
		power_min = param[6];
		power_max = param[7];
		thrust_coef.assign({ param[8], param[9], param[10], param[11], param[12] });
		isp_coef.assign({ param[13], param[14], param[15], param[16], param[17] });
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
		double g0 = 9.80665;
		gm = param[0];
		ve = param[1] * g0;
		mdry = param[2];
		thrust_body.assign({ param[3], param[4] , param[5] });
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
			double rotation_angle = atan2(x[4], x[3]);
			thrust_inertial.assign({ cos(rotation_angle) * thrust_body[0] - sin(rotation_angle) * thrust_body[1],
								 	 sin(rotation_angle) * thrust_body[0] + cos(rotation_angle) * thrust_body[1],
								 	 0.0});
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
	boost::python::list propPy(boost::python::list &ic, boost::python::list &ti, boost::python::list &p,
								int state_dim, int t_dim, int p_dim, double tol, double step_size, int power_type) {

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
		statesAndTimes = prop(IC, t, param, state_dim, t_dim, p_dim, tol, step_size, power_type);

		// Create python list from data to return
		return toTwoDimPythonList(statesAndTimes);
	}

	// Propagation function
	std::vector<vector<double >> prop(vector<double> ic, vector<double> t, vector<double> param, int state_dim, int t_dim,
										int p_dim, double tol, double step_size, int power_type) {
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
		double relTol = tol;
		double absTol = tol;
		typedef runge_kutta_fehlberg78<state_type> rk78;
		auto stepper = make_controlled<rk78>(absTol, relTol);
		
		// Create eom to integrate
		if (power_type == 0) {
			eom2BPScaled_constant_power eom(param);
			size_t steps = integrate_const(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
		} else if (power_type == 1) {
			eom2BPScaled_variable_power eom(param);
			size_t steps = integrate_const(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
		}
		//size_t steps = integrate_adaptive(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));		

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
	cout << "main" << endl;
}