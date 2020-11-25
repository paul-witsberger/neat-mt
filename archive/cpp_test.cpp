#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <math.h>
using namespace std;


typedef vector<double> Vec;
typedef vector<Vec> Mat;


Vec operator*(const Mat &a, const Vec &x) {
	int i, j;
	int m = a.size();
	int n = x.size();
	Vec prod(m);

	for(i = 0; i < m; i++) {
		prod[i] = 0.;
		for(j = 0; j < n; j++) {
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
	for(row = 0; row < n_rows; ++row) {
		for(col = 0; col < n_cols; ++col) {
			for(size_t inner = 0; inner < b.size(); ++inner) {
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
	for(i = 0; i < n_rows; ++i) {
		out.push_back(a[i] / b);
	}
	return out;
}


void print(Vec &vec) {
	int i;
	int n_rows = vec.size();
	for(i = 0; i < n_rows; ++i) {
		cout << vec[i] << " \n"[i==(n_rows-1)];
	}
}


void print(Mat &mat) {
	int i, j;
	int n_rows = mat.size();
	int n_cols = mat[0].size();
	for(i = 0; i < n_rows; ++i) {
		for(j = 0; j < n_cols; ++j) {
			cout << mat[i][j] << " \n"[j==(n_cols-1)];
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
	product = {x, y, z};
	return product;
}


Mat transpose(const Mat &mat) {
	Mat trans(mat);
	int i, j;
	for(i = 0; i < mat.size(); ++i) {
		for(j = 0; j < mat.at(i).size(); ++j) {
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
	dcm.push_back(v_hat);
	dcm.push_back(n_hat);
	dcm.push_back(c_hat);
	dcm_t = transpose(dcm);
	// Apply rotation
	product = dcm_t * vec;
	return product;
}


Vec euler313(const Vec &vec, double psi, double theta, double phi) {
	Mat dcm = {	{ (cos(psi) * cos(phi) - sin(psi) * sin(phi) * cos(theta)), - (sin(psi) * cos(phi) + cos(psi) * sin(phi) * cos(theta)), (sin(theta) * sin(phi)) },
				{ (cos(psi) * sin(phi) + sin(psi) * cos(theta) * cos(phi)), - (sin(psi) * sin(phi) - cos(psi) * cos(theta) * cos(phi)), (sin(theta) * cos(phi)) },
				{                                  (sin(psi) * sin(theta)), -                                  (cos(psi) * sin(theta)),            (cos(theta)) } };
	Vec prod = dcm * vec;
	return prod;
}


int main() {
	int n_rows = 3;
	int n_cols = 3;
	double psi, theta, phi;
	double pi = 2 * acos(0.0);
	double threshold = 1e-8;

	Mat a = { {-2., 4., 1.}, {2., 3., -9.}, {3., -1., 8.} };
	Vec d = {1., 1., 1.};

	psi = pi / 1;
	theta = psi;
	phi = psi;

	Vec rotated = euler313(d, psi, theta, phi);
	for(int i = 0; i < rotated.size(); ++i) {
		if (fabs(rotated[i]) < threshold) {
			rotated[i] = 0.;
		}
	}

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
