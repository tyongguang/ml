/**
 * @file main.cpp
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/05/27 16:48:31
 * @brief 
 *  
 **/
#include "util.h"
#include "soft_max.h"
#include <iostream>
using namespace std;
void test_softmax();
int main()
{
    test_softmax();
    return 0;
}

void test_softmax() {
    double grad_rate = 0.1;
    int num_iter = 400;
    const int size_in = 3, size_out = 8, num_train = 7;
    double train_X[num_train][size_in] = {
        //{0, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 1, 0},
        {1, 1, 1}
    };
    double train_Y[num_train][size_out] = {
        //{1, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 1},
    };
    const char* train_data_file = "train.data";

    SoftMax smc_a(size_in, size_out), smc_b(size_in, size_out);
    smc_a.train(&train_X[0][0], &train_Y[0][0], num_train, grad_rate, num_iter);
    double test_X[2][size_in] = {
        {1, 0, 1},
        {0, 0, 1}
    };
    smc_a.predict(&test_X[0][0], 2);
    smc_a.dump_train_data(train_data_file);

    smc_b.load_train_data(train_data_file, 0);
    smc_b.predict(&train_X[0][0], num_train);
}
