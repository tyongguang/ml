/**
 * @file soft_max.cpp
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/05/27 16:48:31
 * @brief 
 *  
 **/
#include "util.h"
#include <cmath>
#include <ctime>
#include <iostream>
#include "soft_max.h"

using namespace std;

/// n*1 = n*m*m*1 + n*1
// size_y    size_x
// init Y = WX + B + E
SoftMax::SoftMax(int size_x, int size_y) {
    _size_in = size_x;
    _size_out = size_y;

    NEW_POINTERM(_W, double, _size_in, _size_out);
    NEW_POINTERV(_B, double, _size_out);
    NEW_POINTERV(_delta, double, _size_out);
    NEW_POINTERV(_output_data, double, _size_out);

    double a = 1.0 / _size_in;
    for(int i = 0; i < _size_out; ++i) {
        for(int j = 0; j < _size_in; ++j)
            _W[i][j] = uniform(-a, a);
        _B[i] = uniform(-a, a);
    }
}
SoftMax::~SoftMax() {
    FREE_POINTERM(_W, _size_out);
    FREE_POINTERV(_B);
    FREE_POINTERV(_output_data);
    FREE_POINTERV(_delta);
}
void SoftMax::forward_propagation(double* input_data) {
    for(int i = 0; i < _size_out; ++i) {
        _output_data[i] = 0.0;
        for(int j = 0; j < _size_in; ++j)
            _output_data[i] += _W[i][j] * input_data[j];
        _output_data[i] += _B[i];
    }
    softmax(_output_data, _size_out);
}
void SoftMax::back_propagation(double* input_data, double* label, int num_train, double grad_rate) {
    for(int i = 0; i < _size_out; ++i) {
        _delta[i] = label[i] - _output_data[i] ;
        for(int j = 0; j < _size_in; ++j) {
            _W[i][j] += grad_rate * _delta[i] * input_data[j] / num_train;
        }
        _B[i] += grad_rate * _delta[i] / num_train;
    }
}
void SoftMax::train(double* x, double* y, int num_train, double grad_rate) {
    forward_propagation(x);
    back_propagation(x, y, num_train, grad_rate);
}
void SoftMax::train(double* x, double* y, int num_train, double grad_rate, int num_iter) {
    for(int i = 0; i < num_iter; ++i)
        for(int j = 0; j < num_train; ++j)
            train(x + j * _size_in, y + j * _size_out, num_train, grad_rate);
}
int SoftMax::predict(double* x) {
    forward_propagation(x);
    return arg_max(_output_data, _size_out);
}
int SoftMax::predict(double* x, int num_test) {
    for(int i = 0; i < num_test; ++i) {
        print_data(x + _size_in * i, _size_in);
        printf("predict class: %d\n", predict(x + _size_in * i));
    }
}
double SoftMax::cal_error(double **ppdtest, double* pdlabel, int ibatch) {
    double error = 0.0, dmax = 0;
    int imax = -1, ierrNum = 0;
    for (int i = 0; i < ibatch; ++i) {
        imax = predict(ppdtest[i]);
        if (imax != pdlabel[i])
            ++ierrNum;
    }
    error = (double)ierrNum / ibatch;
    return error;
}
void SoftMax::print() {
    print_data(_W, _size_out, _size_in);
    print_data(_B, _size_out);
    print_data(_output_data, _size_out);
}
void SoftMax::dump_train_data(const char* file_name) {
    dump_matrix(file_name, _W, _size_out, _size_in);
    dump_vec(file_name, _B, _size_out);
}
long SoftMax::load_train_data(const char* file_name, long offset) {
    long offset_b = load_matrix(file_name, _W, _size_out, _size_in, offset);
    return load_vec(file_name, _B, _size_out, offset_b);
}
