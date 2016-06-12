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
#include "util.h"

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
class MLP : public NeuralBase {
public:
    MLP(int size_x, int size_y, int* struct_hidden);
    virtual ~MLP();
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
class ConvLayer : public NeuralBase {
public:
    ConvLayer(Shape4d shape_img, Shape4d shape_kernel);
    ~ConvLayer();
    void forward_propagation(double* input_data);
    void back_propagation(double* input_data, double* delta_next, double** W_next, Shape4d shape_delta_next, Shape4d shape_W_next);
public:
    Shape4d _shape_image;
    Shape4d _shape_kernel;
    Shape4d _shape_out;
    int _size_w;
};
class PoolLayer : public NeuralBase {
public:
    PoolLayer(Shape4d shape_img, Shape4d shape_kernel);
    ~PoolLayer();
    void forward_propagation(double* input_data);
    void back_propagation(double* delta_next, double** W_next, int size_out_next);
    void back_propagation(double* delta_next, double** W_next, Shape4d shape_delta_next, Shape4d shape_W_next);
public:
    Shape4d _shape_image;
    Shape4d _shape_kernel;
    Shape4d _shape_out;
};
class CnnLayer : public NeuralBase {
public:
    CnnLayer(Shape4d shape_img, Shape4d* shape_kernel);
    ~CnnLayer();
    void forward_propagation(double* input_data);
    void back_propagation(double* input_data, double* delta_next, double** W_next, Shape4d shape_delta_next, Shape4d shape_W_next);
    void back_propagation(double* input_data, CnnLayer* cnn_next);
    void back_propagation(double* input_data, double* delta_next, double** W_next, int size_out_next);
    void back_propagation(double* input_data, NeuralBase* nn_next);
public:
    double* _output_data;
    ConvLayer* _conv_layer;
    PoolLayer* _pool_layer;
};
class Cnn : public NeuralBase {
public:
    Cnn(Shape4d shape_img, Shape4d* shape_struct, int num_cnn, int size_out);
    virtual ~Cnn();
    using NeuralBase::train;
    void train(double* x, double* y, int num_train, double grad_rate);
    using NeuralBase::predict;
    int predict(double* x); 
    void dump_train_data(const char* file_name) {};
    long load_train_data(const char* file_name, long offset) {};
public:
    int _num_cnn;
    CnnLayer** _cnn_layers;
    NeuralBase* _nn_layer;
};
#endif
