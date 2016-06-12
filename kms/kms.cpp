/**
 * @file kms.cpp
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/06/05 14:57:03
 * @brief 
 *  
 **/

#include "kms.h"
#include "util.h"

Kms::Kms(int size_in, int size_out) : _size_in(size_in), _size_out(size_out) {
    NEW_POINTERV(_K, double, size_in * size_out);
    NEW_POINTERV(_num_K, int, size_out);
    NEW_POINTERV(_output_data, double, size_out);
    double a = 1.0 / size_in;
    for(int i = 0; i < size_in * size_out; ++i) 
        _K[i] = uniform(-a, a);
}
Kms::~Kms() {
    FREE_POINTERV(_K);
    FREE_POINTERV(_num_K);
    FREE_POINTERV(_output_data);
}
void Kms::init_K(double* x) {
    copy(x, _K, _size_out * _size_in);
}
void Kms::train(double* x, int num_in, int iterator) {
    for(int i = 0; i < iterator; ++i) {
        for(int j = 0; j < num_in; ++j) {
            predict(x + j * _size_in);
        }
    }
}
int Kms::predict(double* x) {
    distance(_K, x, _output_data, _size_in, _size_out);
    int label = arg_min(_output_data, _size_out);
    print_data(_output_data, _size_out);
    print_data(_num_K, _size_out);
    means(_K + label * _size_in, x, _size_in, _num_K[label]);
    _num_K[label] += 1;
    return label;
}
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
