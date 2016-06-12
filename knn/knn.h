/**
 * @file knn.h
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/06/05 14:56:21
 * @brief 
 *  
 **/

#ifndef  __KNN_H_
#define  __KNN_H_

class Knn {
public:
    Knn(int size_in, int size_out, int k) : _size_in(size_in), _size_out(size_out), _k(k) {};
    ~Knn();
    void train(double* x, int* y, int num_x);
    int predict(double* x);
    int _k;
    int _size_in;
    int _size_out;
    int _num_in;
    double* _output_data;
private:
    double* _K;
    int* _label;
    double* _sort_k;
    int* _label_tmp;
    int* _idx_k;
};
#endif  //__KNN_H_
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
