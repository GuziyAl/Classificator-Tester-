#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

void showVector (double *vector, int neurons);
void showMatrix(double **matrix,int strings, int columns);

void normalization (double *input_vector);

double *matrixAndVectorMultiplication(double **left,
                                      int neurons,
                                      int weights,
                                      double *right);

double **crtLayer (int neurons, int weights);
void deleteLayer(double **layer, int neurons);

#endif // MATRIX_OPERATIONS_H
