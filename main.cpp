#include <iostream>
#include <string>
#include "networking.h"
#include "matrix_operations.h"

using namespace std;

int main()
{
    cout << ">>> Iris classifacator started <<<" << endl;

    double hidden_out [4];
    double **hidden = crtLayer(4,4);

    hidden [0][0] = -0.26068;
    hidden [0][1] = -7.33755;
    hidden [0][2] = 8.95943;
    hidden [0][3] = 3.43337;

    hidden [1][0] = 4.71534;
    hidden [1][1] = 4.15975;
    hidden [1][2] = -6.50998;
    hidden [1][3] = -3.1242;

    hidden [2][0] = 2.56886;
    hidden [2][1] = 5.62617;
    hidden [2][2] = -10.147;
    hidden [2][3] = -5.93718;

    hidden [3][0] = -14.793;
    hidden [3][1] = -11.6343;
    hidden [3][2] = 20.6151;
    hidden [3][3] = 12.3229;

    double outer_out [3];
    double **outer = crtLayer(3,4);

    outer [0][0] = -8.39012;
    outer [0][1] = 1.09938;
    outer [0][2] = 7.27913;
    outer [0][3] = -7.27241;

    outer [1][0] = 9.46498;
    outer [1][1] = 2.34741;
    outer [1][2] = -11.1553;
    outer [1][3] = -21.7562;

    outer [2][0] = -1.4036;
    outer [2][1] = -11.012;
    outer [2][2] = -8.77385;
    outer [2][3] = 17.6884;

    int num_of_right = 0;
    int set_found = 0;
    int vers_found = 0;
    int virg_found = 0;

    ifstream fin ("C:\\Learning\\new_learn.txt");
    for (int i = 0; i < 60; ++i) {

        double input_signals [4];
        string target;
        readInputData(&fin, input_signals, &target);

        normalization(input_signals);

        double *hidden_net = matrixAndVectorMultiplication(hidden, 4, 4, input_signals);
        for (int i = 0; i < 4; ++i) {
            hidden_out[i] = activationFunction(hidden_net[i]);
        }
        delete [] hidden_net;

        double *outer_net = matrixAndVectorMultiplication(outer, 3, 4, hidden_out);
        for (int i = 0; i < 3; ++i) {
            outer_out[i] = activationFunction(outer_net[i]);
        }
        delete [] outer_net;

        string class_of_iris;
        if (outer_out[0] > outer_out[1] &&
            outer_out[0] > outer_out[2]) {
            class_of_iris = "Iris-setosa";
        }
        else if (outer_out[1] > outer_out[0] &&
                 outer_out[1] > outer_out[2]) {
                 class_of_iris = "Iris-versicolor";
        }
        else if (outer_out[2] > outer_out[1] &&
                 outer_out[2] > outer_out[0]) {
                 class_of_iris = "Iris-virginica";
        }

        if (target == class_of_iris) {
            num_of_right++;
            if (class_of_iris == "Iris-setosa") {
                set_found++;
            }
            else if (class_of_iris == "Iris-versicolor") {
                vers_found++;
            }
            else if (class_of_iris == "Iris-virginica") {
                virg_found++;
            }
        }
    }
    fin.close();

    cout << "Accuracy: " << (num_of_right / 60.0) << endl;
    cout << "Setosa class recall: " << (set_found / 20.0) << endl;
    cout << "Versicolor class recall: " << (vers_found / 20.0) << endl;
    cout << "Virginica class recall: " << (virg_found / 20.0) << endl;

    return 0;
}
