/**
 * @file decision_tree.cpp
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/06/07 11:31:50
 * @brief 
 *  
 **/
#include "decision_tree.h"
#include <math.h>
#include "util.h"

DecisionTree::DecisionTree(int size_in, int size_out) : _size_in(size_in), _size_out(size_out) {
    _h_e = (int)pow(2.0, (double)_size_in);
    NEW_POINTERV(_e, int, _h_e);
    set_value(-1, _e, _h_e);
}
DecisionTree::~DecisionTree() {
    FREE_POINTERV(_e);
}
void DecisionTree::train(double* x, double* y, int num_in) {
    _num_in  = num_in;
    double* x_t = 0;
    int* pos = 0;
    NEW_POINTERV(x_t, double, _size_in * num_in);
    NEW_POINTERV(pos, int, num_in);
    init_data(pos, num_in);
    reshape(x, x_t, num_in, _size_in);
    train(x_t, y, pos, num_in, 0, 0);
    print_data(_e, _h_e);
    FREE_POINTERV(pos);
    FREE_POINTERV(x_t);
}
void DecisionTree::train(double* x, double* y, int* pos, int num_in, int h, int it) {
    int idx_tree = 2 * h + it;
    int idx = gain(x, pos, y, _e, num_in, _size_in, _size_out, 0, idx_tree);
    _e[idx_tree] = idx;
    if(1000 <= idx) {
        return;
    }
    int* v = 0;
    int* t = 0;
    NEW_POINTERV(v, int, _size_out);
    NEW_POINTERV(t, int, num_in);
    split(x + idx * _num_in, pos, y, num_in, t, v, _size_out);
    for(int i = 0; i < _size_out; ++i) {
        int idx_t = 0;
        if(0 != i) 
            idx_t = v[i - 1];
        train(x, y, t + idx_t, v[i], idx_tree, i + 1);
    }
    FREE_POINTERV(v);
}
int DecisionTree::predict(double* x) {
    if(-1 == _e[0])
        return INT_MIN;
    int i = 0;
    while(i < _h_e) {
        i = 2 * i + (int)x[_e[i]] + 1;
        if(1000 <= _e[i])
            return _e[i] - 1000;
    }
    return INT_MIN;
}
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
