#include "networking.h"

double activationFunction(double net)
{
    double tmp_1 = pow(E,(-1 * net));
    double tmp_2 = 1.0 + tmp_1;
    double tmp = 1.0 / tmp_2;
    return tmp;
}

double crtNumber(char *num, int num_length)
{
        double number = 0;
        int point_pos;
        for (point_pos = 0;point_pos < num_length; ++point_pos) {
            if (num[point_pos] == '.') {
                break;
            }
        }

        // До запятой
        for (int i = 0;i < point_pos; ++i) {
            number += ((int)num[i] - '0') * pow(10,point_pos - i - 1);
        }

        // После запятой
        for (int i = point_pos + 1;i < num_length; ++i) {
            number += ((int)num[i] - '0') * pow(10 ,-1 * (i - 1));
        }

        return number;
}

void readInputData (ifstream *fin, double *input_signals, string *target) {
    for (int i = 0; i < 5; ++i) {
        if (i == 4) {
            *fin >> *target;
        }
        else {
            string buffer;
            *fin >> buffer;
            buffer.erase(3);
            char tmp [3];
            for (int j = 0; j < 3; ++j) {
                tmp [j] = buffer [j];
            }
            input_signals[i] = crtNumber(tmp, 3);
        }
    }
}


