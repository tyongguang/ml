/**
 * @file kms.h
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/06/05 14:56:21
 * @brief 
 *  
 **/

#ifndef  __KMS_H_
#define  __KMS_H_

class Kms {
public:
    Kms(int size_in, int size_out);
    ~Kms();
    void init_K(double* x);
    void train(double* x, int num_in, int iterator);
    int predict(double* x);
    int _size_in;
    int _size_out;
    double* _output_data;
public:
    double* _K;
    int* _num_K;
};
#endif  //__KMS_H_
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
