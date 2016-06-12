/**
 * @file neural_network.h
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/05/27 16:48:31
 * @brief 
 *  
 **/
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include "neural_base.h"

class SoftMax : public NeuralBase {
public:
    SoftMax(int size_x, int size_y) : NeuralBase(size_x, size_y) {};
    void forward_propagation(double* input_data);
    void back_propagation(double* input_data, double* label, int num_train, double grad_rate);
};
class HiddenLayer : public NeuralBase {
public:
    HiddenLayer(int size_x, int size_y) : NeuralBase(size_x, size_y) {};
    void forward_propagation(double* input_data);
    void back_propagation(double* input_data, double* delta_next, double** W_next, int size_out_next, double grad_rate);
    void back_propagation(double* input_data, NeuralBase* layer_next, double grad_rate);
    void back_propagation(NeuralBase* layer_pre, NeuralBase* layer_next, double grad_rate);
};
class NeuralNetwork : public NeuralBase {
public:
    NeuralNetwork(int size_x, int size_y, int* struct_hidden);
    virtual ~NeuralNetwork();
    using NeuralBase::train;
    void train(double* x, double* y, int num_train, double grad_rate);
    using NeuralBase::predict;
    int predict(double* x); 
    void dump_train_data(const char* file_name);
    long load_train_data(const char* file_name, long offset);
private:
    int _num_hidden;
    HiddenLayer** _hidden_layers;
    SoftMax* _softmax_layer;
};
#endif
