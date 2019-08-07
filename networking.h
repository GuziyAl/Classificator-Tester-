#ifndef NETWORKING_H
#define NETWORKING_H

#include <cmath>
#include <fstream>
#include <string>

using namespace std;

#define E 2.7182818284590452354

double activationFunction(double net);

double crtNumber(char *num, int num_length);

void readInputData (ifstream *fin, double *input_signals, string *target);

#endif // NETWORKING_H
