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

#include "astIOD.h"

using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::python;

typedef std::vector<double> stateType;
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


double mag2(const Vec &vec) {
	return pow(vec[0] * vec[0] + vec[1] * vec[1], 0.5);
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


Vec rotateVNCtoInertial2D(const Vec &vec, const Vec &state) {
	Vec product;
	Mat dcm, dcm_t;
	// Calculate angle of velocity vector
	double thetaV = std::atan2(state[3], state[2]);
	// Construct DCM
	dcm.push_back({ cos(thetaV), -sin(thetaV)});
	dcm.push_back({ sin(thetaV),  cos(thetaV)});
	// dcm_t = transpose(dcm);
	// Apply rotation
	product = dcm * vec;
	return product;
}


Vec rotateVNCtoInertial3D(const Vec &vec, const Vec &state) {
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


class eom2BPScaled3D {

public:
	eom2BPScaled3D(std::vector<double> param){
		
	}

	void operator()(const stateType &x, stateType &dxdt, const double /* t */) {
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


class eom2BPScaledVariablePower3D {
	double g0, gm, mDry, powerReference, powerMin, powerMax;
	std::vector<double> thrustBody, thrustCoef, ispCoef;

public:
	eom2BPScaledVariablePower3D(std::vector<double> param) {
		g0 = 9.80665;
		thrustBody.assign({param[0], param[1], param[2]});
		mDry = param[3];
		powerReference = param[4];
		powerMin = param[5];
		powerMax = param[6];
		thrustCoef.assign({param[7], param[8], param[9], param[10], param[11]});
		ispCoef.assign({param[12], param[13], param[14], param[15], param[16]});
	}

	double computeThrust(double power) {
		return (thrustCoef[4] * pow(power, 4) + thrustCoef[3] * pow(power, 3) + thrustCoef[2] * pow(power, 2) + thrustCoef[1] * power + thrustCoef[0]);
	}

	double computeIsp(double power) {
		return (ispCoef[4] * pow(power, 4) + ispCoef[3] * pow(power, 3) + ispCoef[2] * pow(power, 2) + ispCoef[1] * power + ispCoef[0]);
	}

	void operator()(const stateType &x, stateType &dxdt, const double /* t */) {
		// calculate auxiliary variables
		double r, r3, thrustMag, ve;
		std::vector<double> thrustInertial;

		r = sqrt((x[0] * x[0]) + (x[1] * x[1]) + (x[2] * x[2]));
		r3 = r * r * r;
		thrustMag = sqrt((thrustBody[0] * thrustBody[0]) + (thrustBody[1] * thrustBody[1]) + (thrustBody[2] * thrustBody[2]));

		// check if mass has dropped below minimum allowable value or thrust is zero
		if (x[6] <= mDry || thrustMag <= 0.0) {
			thrustInertial.assign({0.0, 0.0, 0.0});
			thrustMag = 0.0;
			ve = 1000.0;
		}

		// otherwise, calculate available power and adjust thrust accordingly
		else {
			// check if availble power is below minimum usable threshold
			double auToKm = 149597870.7;
			double powerAvailable = powerReference * pow((auToKm / r), 2);
			if (powerAvailable < powerMin) {
				thrustInertial.assign({0.0, 0.0, 0.0});
				thrustMag = 0.0;
				ve = 1000.0;
			}

			// otherwise, see if thrust vector needs to be updated
			else {
				double rotationAngle, isp, eta, thrustProduced, powerUsed;
				// rotate thrust from spacecraft frame into inertial frame
				rotationAngle = atan2(x[4], x[3]);
				thrustInertial.assign({cos(rotationAngle) * thrustBody[0] - sin(rotationAngle) * thrustBody[1],
									   sin(rotationAngle) * thrustBody[0] + cos(rotationAngle) * thrustBody[1],
									   0.0});

				powerUsed = std::min(powerAvailable, powerMax);
				thrustProduced = computeThrust(powerUsed);
				isp = computeIsp(powerUsed);
				ve = g0 * isp;
				if (thrustProduced < thrustMag) {
					// calculate new Isp based on available power
					eta = thrustProduced / thrustMag;
					thrustInertial[0] *= eta;
					thrustInertial[1] *= eta;
					thrustInertial[2] *= eta;
				}
			}
		}

		// EOMs (3 velocity, 3 accel w/ grav and thrust, 1 mass)
		dxdt[0] = x[3];
		dxdt[1] = x[4];
		dxdt[2] = x[5];
		dxdt[3] = -1 / r3 * x[0] + thrustInertial[0] / x[6];
		dxdt[4] = -1 / r3 * x[1] + thrustInertial[1] / x[6];
		dxdt[5] = -1 / r3 * x[2] + thrustInertial[2] / x[6];
		dxdt[6] = -thrustMag * 1000.0 / ve;
	}
};


class eom2BPScaledConstantPower3D {
	double gm, mDry, ve, thrustMag;
	std::vector<double> thrustBody;

public:
	eom2BPScaledConstantPower3D(std::vector<double> param) {
		thrustBody.assign({param[0], param[1] , param[2]});
		ve = param[3];
		mDry = param[4];
		thrustMag = sqrt((thrustBody[0] * thrustBody[0]) + (thrustBody[1] * thrustBody[1]) + (thrustBody[2] * thrustBody[2]));
	}

	void operator()(const stateType &x, stateType &dxdt, const double /* t */) {
		double r, r3;
		std::vector<double> thrustInertial;

		// calculate radius
		r = sqrt((x[0] * x[0]) + (x[1] * x[1]) + (x[2] * x[2]));
		r3 = r * r * r;

		// check if mass has dropped below minimum allowable value or thrust is zero
		if (x[6] <= mDry || thrustMag <= std::numeric_limits<double>::min()) {
			thrustInertial.assign({0.0, 0.0, 0.0});
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
			thrustInertial = rotateVNCtoInertial3D(thrustBody, x);
		}

		// EOMs (3 velocity, 3 accel w/ grav and thrust, 1 mass)
		dxdt[0] = x[3];
		dxdt[1] = x[4];
		dxdt[2] = x[5];
		dxdt[3] = -1.0 / r3 * x[0] + thrustInertial[0] / x[6];
		dxdt[4] = -1.0 / r3 * x[1] + thrustInertial[1] / x[6];
		dxdt[5] = -1.0 / r3 * x[2] + thrustInertial[2] / x[6];
		dxdt[6] = -thrustMag * 1000.0 / ve;
		if (isnan(dxdt[6])) {
			cout << "dmdt is nan" << endl;
			cout << "thrust_mag = " << thrustMag << endl;
			cout << "ve = " << ve << endl;
			cout << "m = " << x[6] << endl;
		}
	}
};


class eom2BPScaledSTM3D {

public:
	eom2BPScaledSTM3D(std::vector<double> param) {
		
	}

	void operator()(const stateType &x, stateType &dxdt, const double /* t */) {
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


class eom2BPScaled2D {

public:
	eom2BPScaled2D(std::vector<double> param){
		
	}

	void operator()(const stateType &x, stateType &dxdt, const double /* t */) {
		// calculate auxiliary variables
		double r = sqrt((x[0] * x[0]) + (x[1] * x[1]));
		double r3 = r * r * r;
		
		// EOMs (3 velocity, 3 accel w/ gravity only)
		dxdt[0] = x[2];
		dxdt[1] = x[3];
		dxdt[2] = -1 / r3 * x[0];
		dxdt[3] = -1 / r3 * x[1];
	}
};


class eom2BPScaledVariablePower2D {
	double g0, mDry, powerReference, powerMin, powerMax;
	std::vector<double> thrustBody, thrustCoef, ispCoef;

public:
	eom2BPScaledVariablePower2D(std::vector<double> param) {
		g0 = 9.80665;
		thrustBody.assign({param[0], param[1]});
		mDry = param[2];
		powerReference = param[3];
		powerMin = param[4];
		powerMax = param[5];
		thrustCoef.assign({param[6], param[7], param[8], param[9], param[10]});
		ispCoef.assign({param[11], param[12], param[13], param[14], param[15]});
	}

	double computeThrust(double power) {
		return (thrustCoef[4] * pow(power, 4) + thrustCoef[3] * pow(power, 3) + thrustCoef[2] * pow(power, 2) + thrustCoef[1] * power + thrustCoef[0]);
	}

	double computeIsp(double power) {
		return (ispCoef[4] * pow(power, 4) + ispCoef[3] * pow(power, 3) + ispCoef[2] * pow(power, 2) + ispCoef[1] * power + ispCoef[0]);
	}

	void operator()(const stateType &x, stateType &dxdt, const double /* t */) {
		// calculate auxiliary variables
		double r, r3, thrustMag, ve;
		std::vector<double> thrustInertial;

		r = sqrt((x[0] * x[0]) + (x[1] * x[1]));
		r3 = r * r * r;
		thrustMag = sqrt((thrustBody[0] * thrustBody[0]) + (thrustBody[1] * thrustBody[1]));
		
		// check if mass has dropped below minimum allowable value or thrust is zero
		if (x[4] <= mDry || thrustMag <= 0.0) {
			thrustInertial.assign({0.0, 0.0});
			thrustMag = 0.0;
			ve = 1000.0;
		}

		// otherwise, calculate available power and adjust thrust accordingly
		else {
			// check if availble power is below minimum usable threshold
			double auToKm = 149597870.7;
			double powerAvailable = powerReference * pow((auToKm / r), 2);
			if (powerAvailable < powerMin) {
				thrustInertial.assign({0.0, 0.0});
				thrustMag = 0.0;
				ve = 1000.0;
			}

			// otherwise, see if thrust vector needs to be updated
			else {
				double rotationAngle, isp, eta, thrustProduced, powerUsed;
				// rotate thrust from spacecraft frame into inertial frame
				rotationAngle = atan2(x[3], x[2]);
				thrustInertial.assign({cos(rotationAngle) * thrustBody[0] - sin(rotationAngle) * thrustBody[1],
									   sin(rotationAngle) * thrustBody[0] + cos(rotationAngle) * thrustBody[1]});

				powerUsed = std::min(powerAvailable, powerMax);
				thrustProduced = computeThrust(powerUsed);
				isp = computeIsp(powerUsed);
				ve = g0 * isp;
				if (thrustProduced < thrustMag) {
					// calculate new Isp based on available power
					eta = thrustProduced / thrustMag;
					thrustInertial[0] *= eta;
					thrustInertial[1] *= eta;
				}
			}
		}

		// EOMs (3 velocity, 3 accel w/ grav and thrust, 1 mass)
		dxdt[0] = x[2];
		dxdt[1] = x[3];
		dxdt[2] = -1 / r3 * x[0] + thrustInertial[0] / x[4];
		dxdt[3] = -1 / r3 * x[1] + thrustInertial[1] / x[4];
		dxdt[4] = -thrustMag * 1000.0 / ve;
	}
};


class eom2BPScaledConstantPower2D {
	double gm, mDry, ve, thrustMag;
	std::vector<double> thrustBody;

public:
	eom2BPScaledConstantPower2D(std::vector<double> param) {
		thrustBody.assign({param[0], param[1]});
		ve = param[2];
		mDry = param[3];
		thrustMag = sqrt((thrustBody[0] * thrustBody[0]) + (thrustBody[1] * thrustBody[1]));
	}

	void operator()(const stateType &x, stateType &dxdt, const double /* t */) {
		double r, r3;
		std::vector<double> thrustInertial;

		// calculate radius
		r = sqrt((x[0] * x[0]) + (x[1] * x[1]));
		r3 = r * r * r;

		// check if mass has dropped below minimum allowable value or thrust is zero
		if (x[4] <= mDry || thrustMag <= std::numeric_limits<double>::min())
			thrustInertial.assign({0.0, 0.0});
		// otherwise, rotate thrust vector to inertial frame
		else
			thrustInertial = rotateVNCtoInertial2D(thrustBody, x);

		// EOMs (3 velocity, 3 accel w/ grav and thrust, 1 mass)
		dxdt[0] = x[2];
		dxdt[1] = x[3];
		dxdt[2] = -1.0 / r3 * x[0] + thrustInertial[0] / x[4];
		dxdt[3] = -1.0 / r3 * x[1] + thrustInertial[1] / x[4];
		dxdt[4] = -thrustMag * 1000.0 / ve;
		if (isnan(dxdt[4])) {
			cout << "dmdt is nan" << endl;
			cout << "thrust_mag = " << thrustMag << endl;
			cout << "ve = " << ve << endl;
			cout << "m = " << x[4] << endl;
		}
	}
};


struct getStateAndTime {
	std::vector<stateType> &m_states;
	std::vector<double> &m_times;

	getStateAndTime(std::vector<stateType> &states, std::vector<double> &times) : m_states(states), m_times(times) {}

	void operator()(const stateType &x, double t) {
		// Store new values
		m_states.push_back(x);
		m_times.push_back(t);
	}
};


struct TBP {
	// Put function into format python can understand and the propagate
	boost::python::list propPy(boost::python::list &ic, boost::python::list &ti, boost::python::list &p, int stateDim, int tDim,
							   int pDim, double rTol, double aTol, double stepSize, int integratorType, int eomType, int nDim) {

		typedef std::vector<double> stateType;
		std::vector<stateType> statesAndTimes;

		stateType IC(stateDim, 0);
		stateType t(tDim, 0);
		std::vector<double> param(pDim, 0);

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
		statesAndTimes = prop(IC, t, param, rTol, aTol, stepSize, integratorType, eomType, nDim);

		// Create python list from data to return
		return toTwoDimPythonList(statesAndTimes);
	}

	// Propagation function
	std::vector<vector<double >> prop(vector<double> ic, vector<double> t, vector<double> param, double rTol, double aTol,
									  double stepSize, int integratorType, int eomType, int nDim) {
		using namespace std;
		using namespace boost::numeric::odeint;

		typedef std::vector<double> stateType;
		std::vector<stateType> statesAndTimes;

		// Set vectors intermediate steps during integration
		std::vector<double> tOut;
		std::vector<stateType> statesOut;

		// Determine step size (forward or backward) and set initial step size
		double h = t[1] > t[0] ? stepSize : -stepSize;

		// Set integrator type -> Currently set at rk78
		double relTol = rTol;
		double absTol = aTol;
		typedef runge_kutta_fehlberg78<stateType> rk78;
		auto stepper = make_controlled<rk78>(absTol, relTol);
		
		// Create eom to integrate
		if ((eomType == 0) && (nDim == 3)) {
			eom2BPScaledConstantPower3D eom(param);
			if (integratorType == 0) {
				size_t steps = integrate_const(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			} else if (integratorType == 1) {
				size_t steps = integrate_adaptive(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			}
		} else if ((eomType == 1) && (nDim == 3)) {
			eom2BPScaledVariablePower3D eom(param);
			if (integratorType == 0) {
				size_t steps = integrate_const(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			} else if (integratorType == 1) {
				size_t steps = integrate_adaptive(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			}
		} else if ((eomType == 2) && (nDim == 3)) {
			eom2BPScaledSTM3D eom(param);
			if (integratorType == 0) {
				size_t steps = integrate_const(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			} else if (integratorType == 1) {
				size_t steps = integrate_adaptive(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			}
		} else if ((eomType == 3) && (nDim == 3)) {
			eom2BPScaled3D eom(param);
			if (integratorType == 0) {
				size_t steps = integrate_const(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			} else if (integratorType == 1) {
				size_t steps = integrate_adaptive(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			}
		} else if ((eomType == 0) && (nDim == 2)) {
			eom2BPScaledConstantPower2D eom(param);
			if (integratorType == 0) {
				size_t steps = integrate_const(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			} else if (integratorType == 1) {
				size_t steps = integrate_adaptive(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			}
		} else if ((eomType == 1) && (nDim == 2)) {
			eom2BPScaledVariablePower2D eom(param);
			if (integratorType == 0) {
				size_t steps = integrate_const(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			} else if (integratorType == 1) {
				size_t steps = integrate_adaptive(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			}
		} else if ((eomType == 2) && (nDim == 2)) {
			// not defined
		} else if ((eomType == 3) && (nDim == 2)) {
			eom2BPScaled2D eom(param);
			if (integratorType == 0) {
				size_t steps = integrate_const(stepper, eom, ic, t[0], t[1], h, getStateAndTime(statesOut, tOut));
			} else if (integratorType == 1) {
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


struct maneuvers {
	boost::python::list lambert(double gm, boost::python::list &r1, boost::python::list &r2, double dtsec, bool direct)
	{
		double _r1[3], _r2[3], v1Battin[3], v1[3], v2[3];
		int error, nrev = 0;
		char dm, df = 'd';
		dm = (direct ? 's' : 'l');
		std::string fname = "test_lambertbattin.txt";
		bool printToFile = false;
		FILE *outfile;

		for (int i = 0; i < 3; ++i) {
			_r1[i] = boost::python::extract<double>(r1[i]);
			_r2[i] = boost::python::extract<double>(r2[i]);
		}

		// lambertbattin
		if (printToFile)
			outfile = fopen(fname.c_str(), "w");
		astIOD::lambertbattin(_r1, _r2, gm, v1Battin, dm, df, nrev, dtsec, v1, v2, error, outfile, printToFile);
		if (printToFile)
			fclose(outfile);

		boost::python::list listV1, listV2;

		for (int i = 0; i < 3; ++i) {
            listV1.append(v1[i]);
            listV2.append(v2[i]);
        }

		boost::python::list outList;
		outList.append(listV1);
		outList.append(listV2);
		outList.append(error);

		return outList;
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
	class_<TBP>("TBP").def("prop", &TBP::propPy);
	class_<maneuvers>("maneuvers").def("lambert", &maneuvers::lambert);
};

