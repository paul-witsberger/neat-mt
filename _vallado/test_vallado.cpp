#include "astMath.h"
#include "astTime.h"
#include "ast2Body.h"
#include "astIOD.h"
#include <iostream>
#include <chrono>

int main(int argc, char *argv[]) {
	// Test astMath
	std::cout << "From astMath: pi = " << astMath::round(4.3) << std::endl;

	// Test astTime
	std::cout << "From astTime: month = " << astTime::getmon((char*)"Jan") << std::endl;

	// Test ast2Body
	double r[3] = {10000, 0, 0}, v[3] = {0, 7, 0};
	double p, a, ecc, incl, raan, argp, nu, m, eccanom, arglat, truelon, lonper;
	const double mu = 398600.0;
	ast2Body::rv2coe(r, v, mu, p, a, ecc, incl, raan, argp, nu, m, eccanom, arglat, truelon, lonper);
	std::cout << "From ast2Body: p = " << p << std::endl;

	// Test astIOD - lambertuniv and lambertbattin
	double r1[3] = {10000, 0, 0}, r2[3] = {0, 11000, 12000}, dtsec = 1000, v1Battin[3] = {0, 0, 0};
	double v1[3], v2[3], v1t[3], v2t[3];
	int numIter = 40, numTests = (int)1;
	char dm = 's';
	bool printToFile = true;
	if (argc > 1)
		dtsec = atof(argv[1]);
		if (argc > 2)
			numIter = atoi(argv[2]);
			if (argc > 3)
				dm = *argv[3];
				if (argc > 4)
					printToFile = (bool)atoi(argv[4]);
					if (argc > 5)
						numTests = atoi(argv[5]);
						if ((numTests > 1) && printToFile) {
							printf("WARNING: printing output to file adds significant runtime!\n");
						}
	char df = 'd';  // 'd' for direct or 'r' for retrograde (...I think)
	int nrev = 0, error;
	FILE *outfile;

	// lambertuniv
	outfile = fopen("test_lambertuniv.txt", "w");
	auto start_univ = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < numTests; ++i) {
		astIOD::lambertuniv(r1, r2, dm, df, nrev, dtsec, v1, v2, error, outfile, printToFile);
	}
	auto stop_univ = std::chrono::high_resolution_clock::now();
	fclose(outfile);
	printf("\nFrom lambertuniv:\n");
	printf("\tv1 = [%f, %f, %f]\n", v1[0], v1[1], v1[2]);
	printf("\tv2 = [%f, %f, %f]\n", v2[0], v2[1], v2[2]);
	printf("\terror code = %d\n", error);

	// lambertbattin
	outfile = fopen("test_lambertbattin.txt", "w");
	auto start_battin = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < numTests; ++i) {
		astIOD::lambertbattin(r1, r2, v1Battin, dm, df, nrev, dtsec, v1t, v2t, error, outfile, printToFile);
	}
	auto stop_battin = std::chrono::high_resolution_clock::now();
	fclose(outfile);
	printf("\nFrom lambertbattin:\n");
	printf("\tv1 = [%f, %f, %f]\n", v1t[0], v1t[1], v1t[2]);
	printf("\tv2 = [%f, %f, %f]\n", v2t[0], v2t[1], v2t[2]);
	printf("\terror code = %d\n", error);

	auto time_univ = std::chrono::duration_cast<std::chrono::microseconds>(stop_univ - start_univ);
	auto time_battin = std::chrono::duration_cast<std::chrono::microseconds>(stop_battin - start_battin);
	printf("\nElapsed time:\n\tlambertuniv:\t");
	std::cout << time_univ.count() << " us\n";
	printf("\tlambertbattin:\t");
	std::cout << time_battin.count() << " us\n";
}

/*
1. To compile astIOD.cpp: g++ -c astIOD.cpp
2. To create libvallado.lib: ar rs libvallado.lib *.o
3. To compile example2.cpp: g++ -o example2_out_test example2.cpp -L. -llibvallado
4. To run: example2_out [dtsec numIter dm printToFile numTests]
*/
