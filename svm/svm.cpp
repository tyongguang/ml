/**
 * @file svm.cpp
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/05/27 18:28:19
 * @brief 
 *  
 **/
#include "svm.h"
#include "util.h"

int SVM::predict(double* input_data) {
    double output = 0.0;
    for(int i = 0; i < _num_in; i++)
        if(_alphas[i] > 0)
            output += _alphas[i] * _label[i] * rbf(_input_data + i * _size_in, input_data, _size_in, _delta);
    output -= _smo._b;
    return output > 0 ? 1 : -1;
}
void SVM::train(double* input_data, int* label, int iterator) {
    _input_data = input_data;
    _label = label;
    _smo._label = _label;
    rbf(_input_data, _K, _size_in, _num_in, _delta);
    _smo.process(iterator);
}
double SVM::error_rate() {
    return _smo.error_rate();
}
SVM::SVM(svm_t paras) {
    _num_in = paras._num_in;
    _size_in = paras._size_in;
    _delta = paras._delta; 
    _smo = SMO(paras._C, paras._eps, paras._T);
    NEW_POINTERV(_alphas, double, _num_in);
    NEW_POINTERV(_error, double, _num_in);
    NEW_POINTERV(_K, double, _num_in * _num_in);
    _smo._size_in = _size_in;
    _smo._num_in = _num_in;
    _smo._K = _K;
    _smo._alphas = _alphas;
    _smo._error = _error;
}
SVM::~SVM() {
    FREE_POINTERV(_alphas);
    FREE_POINTERV(_error);
    FREE_POINTERV(_K);
}
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
