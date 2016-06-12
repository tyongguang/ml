/**
 * @file main.cpp
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/05/27 16:48:31
 * @brief 
 *  
 **/
#include "util.h"
#include "neural_network.h"
#include <iostream>
using namespace std;
void test_softmax();
void test_softmax_xor();
void test_mlp();
void test_xor();
void test_load1();
void test_load2();
int main()
{
    //test_softmax();
    //test_mlp();
    cout << "mlp result:" << endl;
    test_xor();
    cout << "softmax result:" << endl;
    test_softmax_xor();
    //test_load1();
    //test_load2();
    //list_folder_files("digits_txt");
    //txt2pb("digits_txt/9_3.txt", "9_3.data", 32, 32);
    //folder_txt2pb("digits_txt", "digits_pb", 32, 32);
    return 0;
}

void test_load1() {
    double a[32][32] = {0};
    load_matrix("digits_txt/9_3.txt", &a[0][0], 32, 32);
    print_data(&a[0][0], 32, 32);
    dump_matrix("a.data", &a[0][0], 32, 32);
    load_matrix("a.data", &a[0][0], 32, 32, 0);
    print_data(&a[0][0], 32, 32);
}
void test_load2() {
    double** data = 0;
    NEW_POINTERM(data, double, 32, 32);
    set_value(0.0, data, 32, 32);
    load_matrix("digits_txt/9_3.txt", data, 32, 32);
    print_data(data, 32, 32);
    dump_matrix("data.data", data, 32, 32);
    load_matrix("data.data", data, 32, 32, 0);
    print_data(data, 32, 32);
    FREE_POINTERM(data, 32);
}
void test_xor() {
    double grad_rate = 0.1;
    int num_iter = 10000;
    const int size_in = 2, size_out = 2, num_train = 4;
    double train_X[num_train][size_in] = {
        {0, 0},
        {1, 0},
        {0, 1},
        {1, 1},
    };
    double train_Y[num_train][size_out] = {
        {1, 0},
        {0, 1},
        {0, 1},
        {1, 0},
    };
    const char* train_data_file = "train_xor.data";

    int struct_hidden[] = {2, 2};
    NeuralNetwork nn_a(size_in, size_out, &struct_hidden[0]);
    nn_a.train(&train_X[0][0], &train_Y[0][0], num_train, grad_rate, num_iter);
    nn_a.predict(&train_X[0][0], num_train);
}
void test_softmax_xor() {
    double grad_rate = 0.1;
    int num_iter = 10000;
    const int size_in = 2, size_out = 2, num_train = 4;
    double train_X[num_train][size_in] = {
        {0, 0},
        {1, 0},
        {0, 1},
        {1, 1},
    };
    double train_Y[num_train][size_out] = {
        {1, 0},
        {0, 1},
        {0, 1},
        {1, 0},
    };
    const char* train_data_file = "train_sfm_xor.data";

    SoftMax lr(size_in, size_out);
    lr.train(&train_X[0][0], &train_Y[0][0], num_train, grad_rate, num_iter);
    lr.predict(&train_X[0][0], num_train);
}
void test_mlp() {
    double grad_rate = 0.1;
    int num_iter = 4000;
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
    const char* train_data_file = "train_mlp.data";

    int struct_hidden[] = {5, 5};
    NeuralNetwork nn_a(size_in, size_out, &struct_hidden[0]), nn_b(size_in, size_out, &struct_hidden[0]);
    nn_a.train(&train_X[0][0], &train_Y[0][0], num_train, grad_rate, num_iter);
    double test_X[2][size_in] = {
        {1, 0, 1},
        {0, 0, 1}
    };
    nn_a.predict(&test_X[0][0], 2);
    nn_a.predict(&train_X[0][0], num_train);
    nn_a.dump_train_data(train_data_file);

    nn_b.load_train_data(train_data_file, 0);
    nn_b.predict(&train_X[0][0], num_train);
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
    const char* train_data_file = "train_sfm.data";

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
