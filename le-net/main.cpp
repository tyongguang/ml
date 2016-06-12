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
void test_matrix();
void test_cnn_layer();
void test_cnn();

int main()
{
    test_softmax();
    //test_mlp();
    //cout << "mlp result:" << endl;
    //test_xor();
    //cout << "softmax result:" << endl;
    //test_softmax_xor();
    //test_load1();
    //test_load2();
    //list_folder_files("digits_txt");
    //txt2pb("digits_txt/9_3.txt", "9_3.data", 32, 32);
    //folder_txt2pb("digits_txt", "digits_pb", 32, 32);
    //test_matrix();
    //test_cnn_layer();
    test_cnn();
    return 0;
}

void test_cnn() {
    cout << "====test_cnn====" << endl;
    Shape4d shape_in = shape_2d(32, 32, 1, 1);
    Shape4d shape_conv = shape_2d(5, 5, 20, 1);
    Shape4d shape_pool = shape_2d(2, 2, 1, 1);
    Shape4d shape_kernel[2];
    shape_kernel[0] = shape_2d(5, 5, 20, 1);
    shape_kernel[1] = shape_2d(2, 2, 1, 1);
    Cnn cnn = Cnn(shape_in, &shape_kernel[0], 1, 10);
    int num_in = 4;
    int size = 32;
    int size_a = num_in * size;
    cout << cnn._cnn_layers[0]->_conv_layer->_size_in << endl;
    cout << cnn._cnn_layers[0]->_conv_layer->_size_w << endl;
    cout << cnn._cnn_layers[0]->_conv_layer->_size_out << endl;
    cout << cnn._cnn_layers[0]->_pool_layer->_size_in << endl;
    cout << cnn._cnn_layers[0]->_pool_layer->_size_out << endl;
    double a[40][1024] = {0};
    load_digits(&a[0][0], 1, 32, 32, 20);
    load_digits(&a[0][0] + 20480, 9, 32, 32, 20);
    double b[40][10] = {
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
    };
    cnn.train(&a[0][0], &b[0][0], 40, 0.1, 20);
    double c[32][32] = {0};
    load_matrix("digits_txt/1_43.txt", &c[0][0], 32, 32);
    cnn.predict(&c[0][0], 1);
    double d[32][32] = {0};
    load_matrix("digits_txt/9_67.txt", &d[0][0], 32, 32);
    cnn.predict(&d[0][0], 1);
}
void test_cnn_layer() {
    cout << "====test_cnn_layer====" << endl;
    Shape4d shape_in = shape_2d(32, 32, 1, 1);
    Shape4d shape_conv = shape_2d(5, 5, 20, 1);
    Shape4d shape_pool = shape_2d(2, 2, 1, 1);
    Shape4d shape_kernel[2];
    shape_kernel[0] = shape_2d(5, 5, 20, 1);
    shape_kernel[1] = shape_2d(2, 2, 1, 1);
    CnnLayer cnn_layer = CnnLayer(shape_in, &shape_kernel[0]);
    cout << cnn_layer._conv_layer->_size_in << endl;
    cout << cnn_layer._conv_layer->_size_w << endl;
    cout << cnn_layer._conv_layer->_size_out << endl;
    shape_print(cnn_layer._conv_layer->_shape_image);
    shape_print(cnn_layer._conv_layer->_shape_kernel);
    shape_print(cnn_layer._conv_layer->_shape_out);
    cout << cnn_layer._pool_layer->_size_in << endl;
    cout << cnn_layer._pool_layer->_size_out << endl;
    shape_print(cnn_layer._pool_layer->_shape_image);
    shape_print(cnn_layer._pool_layer->_shape_kernel);
    shape_print(cnn_layer._pool_layer->_shape_out);
    double a[32][32] = {0};
    load_matrix("digits_txt/9_3.txt", &a[0][0], 32, 32);
    cnn_layer.forward_propagation(&a[0][0]);
    //print_data(cnn_layer._pool_layer->_W, 20, 14 * 14);
    //print_data(cnn_layer._conv_layer->_output_data, 28 * 28);
    print_data(cnn_layer._conv_layer->_output_data + 28 * 28, 28 * 28);
    print_data(cnn_layer._conv_layer->_output_data + 2 * 28 * 28, 28 * 28);
    print_data(cnn_layer._pool_layer->_output_data, 14 * 14);
    print_data(cnn_layer._pool_layer->_output_data + 14 * 14, 14 * 14);
    print_data(cnn_layer._pool_layer->_output_data + 2 * 14 * 14, 14 * 14);
}
void test_matrix() {
    double a[32][32] = {0};
    double b[36][36] = {0};
    double c[16][16] = {0};
    double d[28][28] = {0};
    double e[5][5] = {
        {1, 0, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 0, 1, 0},
        {0, 0, 0, 0, 1},
    };
    int f[16][16] = {0};
    load_matrix("digits_txt/9_3.txt", &a[0][0], 32, 32);
    print_data(&a[0][0], 32, 32);
    conv2d(&a[0][0], &e[0][0], &d[0][0], shape_2d(32, 32, 1, 1), shape_2d(5, 5, 1, 1));
    print_data(&d[0][0], 28, 28);
    Shape4d shape = padding(&a[0][0], &b[0][0], shape_2d(32, 32, 1, 1), shape_2d(5, 5, 1, 1));
    print_data(&b[0][0], 36, 36);
    conv2d(&b[0][0], &e[0][0], &a[0][0], shape_2d(36, 36, 1, 1), shape_2d(5, 5, 1, 1));
    print_data(&a[0][0], 32, 32);
    //pool_max(&a[0][0], &c[0][0], shape_2d(32, 32, 1, 1), shape_2d(2, 2, 1, 1));
    //print_data(&c[0][0], 16, 16);
    //pool_max_inverse(&c[0][0], &a[0][0], shape_2d(16, 16, 1, 1), shape_2d(2, 2, 1, 1));
    //print_data(&a[0][0], 32, 32);
    //pool_max(&a[0][0], &c[0][0], shape_2d(32, 32, 1, 1), shape_2d(2, 2, 1, 1));
    //print_data(&c[0][0], 16, 16);
    //rot180(&a[0][0], shape_2d(32, 32, 1, 1));
    //print_data(&a[0][0], 32, 32);
    //pool_max(&a[0][0], &c[0][0], &f[0][0], shape_2d(32, 32, 1, 1), shape_2d(2, 2, 1, 1));
    //print_data(&c[0][0], 16, 16);
    //print_data(&f[0][0], 16, 16);
    //print_data(&a[0][0], 32, 32);
    //pool_max_inverse(&c[0][0], &f[0][0], &a[0][0], shape_2d(16, 16, 1, 1), shape_2d(2, 2, 1, 1));
    //print_data(&a[0][0], 32, 32);
    double g[5][5] = {
        {2,2,2,2,2},
        {2,2,2,2,2},
        {2,2,2,2,2},
        {2,2,2,2,2},
        {2,2,2,2,2},
    };
    double h[5][5] = {
         {3,3,3,3,3},
         {3,3,3,3,3},
         {3,3,3,3,3},
         {3,3,3,3,3},
         {3,3,3,3,3},
    };
    add(&g[0][0], &h[0][0], 25);
    print_data(&g[0][0], 5, 5);
    double sum_g = sum(&g[0][0], 25);
    cout << sum_g << endl;
    copy(&g[0][0], &h[0][0], 25);
    print_data(&h[0][0], 5, 5);
    double** data = 0;
    double** tmp = 0;
    NEW_POINTERM(data, double, 32, 32);
    NEW_POINTERM(tmp, double, 32, 32);
    set_value(0.0, data, 32, 32);
    set_value(3, tmp, 32, 32);
    load_matrix("digits_txt/9_3.txt", data, 32, 32);
    print_data(data, 32, 32);
    copy(tmp, data, 32, 32);
    print_data(data, 32, 32);
    mat_func(&h[0][0], 25, sec);
    print_data(&h[0][0], 5, 5);
    double aa[8][8] = {
         {3,3,3,3,3,3,3,3},
         {3,3,3,3,3,3,3,3},
         {3,3,3,3,3,3,3,3},
         {3,3,3,3,3,3,3,3},
         {3,3,3,3,3,3,3,3},
         {3,3,3,3,3,3,3,3},
         {3,3,3,3,3,3,3,3},
         {3,3,3,3,3,3,3,3},
    };
    double ab[4][4] = {0};
    sum(&aa[0][0], &ab[0][0], 64, 16);
    print_data(&ab[0][0], 4, 4);
    Shape4d shape_t = shape_2d(5, 6, 3, 4);
    cout << size2_shape(shape_t) << endl;
    cout << size3_shape(shape_t) << endl;
    cout << size3c_shape(shape_t) << endl;
    cout << size_shape(shape_t) << endl;
    crt_data(&aa[0][0], 64);
    print_data(&aa[0][0], 8, 8);
    double ac[8][8] = {0};
    reshape(&aa[0][0], &ac[0][0], shape_2d(2, 8, 2, 2));
    cout << "===============" << endl;
    print_data(&ac[0][0], 8, 8);
    reshape(&aa[0][0], &ac[0][0], shape_2d(2, 2, 2, 8));
    cout << "===============" << endl;
    print_data(&ac[0][0], 8, 8);
    double** data_a = 0;
    double** tmp_a = 0;
    NEW_POINTERM(data_a, double, 2, 32);
    NEW_POINTERM(tmp_a, double, 8, 8);
    crt_data(data_a, 2, 32);
    print_data(data_a, 2, 32);
    reshape(data_a, tmp_a, shape_2d(2, 2, 2, 8));
    print_data(tmp_a, 8, 8);
    reshape_a(data_a, tmp_a, shape_2d(2, 2, 2, 8));
    print_data(tmp_a, 8, 8);
    rot180(data_a, tmp_a, shape_2d(2, 2, 2, 8));
    print_data(tmp_a, 8, 8);
    cout << "===============" << endl;
    print_data(&aa[0][0], 8, 8);
    cout << "===============" << endl;
    rot180(&aa[0][0], &ac[0][0], shape_2d(8, 8, 1, 1));
    print_data(&ac[0][0], 8, 8);
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
    MLP nn_a(size_in, size_out, &struct_hidden[0]);
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
    MLP nn_a(size_in, size_out, &struct_hidden[0]), nn_b(size_in, size_out, &struct_hidden[0]);
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
