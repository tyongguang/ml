/**
 * @file svm.h
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/05/27 16:48:31
 * @brief 
 *  
 **/

#ifndef  __SVM_H_
#define  __SVM_H_
#include "smo.h"

struct svm_t{
    int _num_in;
    int _size_in;
    double _C;
    double _T;
    double _eps;
    double _delta;
};
class SVM{
public:
    SVM(svm_t);
    ~SVM();
    void train(double* input_data, int* label, int iterator);
    double error_rate();
    int predict(double* input_data);
private:
    SMO _smo;
    int _num_in;
    int _size_in;
    double _delta;
    int* _label;
    double* _input_data;
    double* _alphas;
    double* _error;
    double* _K;
 
};
#endif  //__SVM_H_
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
