/**
 * @file knn.cpp
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/06/05 14:57:03
 * @brief 
 *  
 **/
#include "knn.h"
#include <float.h>
#include "util.h"

Knn::~Knn() {
    FREE_POINTERV(_output_data);
    FREE_POINTERV(_sort_k);
    FREE_POINTERV(_idx_k);
    FREE_POINTERV(_label_tmp);
}
void Knn::train(double* x, int* y, int num_x) {
    _K = x;
    _label = y;
    _num_in = num_x;
    NEW_POINTERV(_output_data, double, num_x);
    NEW_POINTERV(_label_tmp, int, num_x);
    NEW_POINTERV(_sort_k, double, _k);
    NEW_POINTERV(_idx_k, int, _k);
}
int Knn::predict(double* x) {
    set_value(DBL_MAX, _sort_k, _k);
    set_value(0, _idx_k, _k);
    set_value(0, _label_tmp, _num_in);
    distance(_K, x, _output_data, _size_in, _num_in);
    sort_k(_output_data, _label, _sort_k, _idx_k, _num_in, _k);
    print_data(_idx_k, _k);
    return max_repeat(_idx_k, _label_tmp, _k, _num_in);
}
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
