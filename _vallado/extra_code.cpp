
// #include "C:/Users/pawit/Documents/codes/src/vallado - Copy/ast2Body.h"
// #include "C:/Users/pawit/Documents/codes/src/vallado - Copy/astIOD.h"

// void test_lambert()
// {
// 	double r1[3], r2[3], dtsec, v1[3], v2[3];
// 	char dm, df;
// 	int nrev, error;
// 	r1[0] = 10000.0;
// 	r1[1] = 0.0;
// 	r1[2] = 0.0;
// 	r2[0] = 0.0;
// 	r2[1] = 11000.0;
// 	r2[2] = 0.0;
// 	dtsec = 2000;  // transfer time in sec
// 	dm = 's';  // short or long; 'l' for long, else short
// 	df = 'd';  // only used if nrev > 0; 'd' for lower initial guess, else higher initial guess
// 	nrev = 0;
// 	astIOD::lambertuniv(r1, r2, dm, df, nrev, dtsec, v1, v2, error);
// 	// Outputs: 
// 	//   > v1: double[3]
// 	//   > v2: double[3]
// 	//   > error:  1 = not converged; 2 = y negative; 3 = 180 deg transfer
// 	std::cout << "v1 | " << "v2" << std::endl;
// 	for (int i = 0; i < 3; ++i) {
// 		std::cout << v1[i] << " | " << v2[i] << std::endl;
// 	}
// 	std::cout << "error = " << error << std::endl;
// }

	// double r[3] = {10000, 0, 0}, v[3] = {0, 7, 0};
	// double p, a, ecc, incl, raan, argp, nu, m, eccanom, arglat, truelon, lonper;
	// const double mu = 398600.0;
	// ast2Body::rv2coe(r, v, mu, p, a, ecc, incl, raan, argp, nu, m, eccanom, arglat, truelon, lonper);
	// std::cout << "From ast2Body: p = " << p << std::endl;

	// double r1[3] = {10000, 0, 0}, r2[3] = {0, 11000, 0}, dtsec, v1[3], v2[3];
	// char dm = 's', df = 'd';
	// int nrev = 0, error;
	// astIOD::lambertuniv(r1, r2, dm, df, nrev, dtsec, v1, v2, error);
	// std::cout << "From astIOD: v1 = [" << v1[0] << ", " << v1[1] << ", " << v1[2] << "]\n";