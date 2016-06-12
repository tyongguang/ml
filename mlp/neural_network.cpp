/**
 * @file neural_network.cpp
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/05/27 16:48:31
 * @brief 
 *  
 **/
#include "util.h"
#include "neural_network.h"

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
void HiddenLayer::forward_propagation(double* input_data) { 
    for(int i = 0; i < _size_out; ++i) {                              
        _output_data[i] = 0.0;                                     
        for(int j = 0; j < _size_in; ++j) 
            _output_data[i] += _W[i][j] * input_data[j];
        _output_data[i] += _B[i];
        _output_data[i] = sigmoid(_output_data[i]);
    }
}
void HiddenLayer::back_propagation(double* input_data, double* delta_next, double** W_next, int size_out_next, double grad_rate) {
    double* sigma;
    NEW_POINTERV(sigma, double, _size_out);
    set_value(0.0, sigma, _size_out);

    for(int i = 0; i < size_out_next; ++i)
        for(int j = 0; j < _size_out; ++j)
            sigma[j] += W_next[i][j] * delta_next[i];

    //计算得到本层的残差delta
    for(int i = 0; i < _size_out; ++i)
        _delta[i] = sigma[i] * _output_data[i] * (1 - _output_data[i]);

    //调整本层的权值w
    for(int i = 0; i < _size_out; ++i) {
        for(int j = 0; j < _size_in; ++j)
            _W[i][j] += grad_rate * _delta[i] * input_data[j];
        _B[i] += grad_rate * _delta[i];
    }
    FREE_POINTERV(sigma);
}
void HiddenLayer::back_propagation(double* input_data, NeuralBase* layer_next, double grad_rate) {
    back_propagation(input_data, layer_next->_delta, layer_next->_W, layer_next->_size_out, grad_rate);
}
void HiddenLayer::back_propagation(NeuralBase* layer_pre, NeuralBase* layer_next, double grad_rate) {
    back_propagation(layer_pre->_output_data, layer_next->_delta, layer_next->_W, layer_next->_size_out, grad_rate);
}
NeuralNetwork::NeuralNetwork(int size_x, int size_y, int* struct_hidden) : NeuralBase(size_x, size_y) {
    _num_hidden = sizeof(struct_hidden) / sizeof(int);
    _hidden_layers = new HiddenLayer* [_num_hidden];
    _hidden_layers[0] = new HiddenLayer(_size_in, struct_hidden[0]);
    for(int i = 1; i < _num_hidden; ++i)
        _hidden_layers[i] = new HiddenLayer(struct_hidden[i-1], struct_hidden[i]);
    _softmax_layer = new SoftMax(struct_hidden[_num_hidden-1], _size_out);
}
NeuralNetwork::~NeuralNetwork() {
    FREE_POINTERV(_hidden_layers);
    FREE_POINTER(_softmax_layer);
}
void NeuralNetwork::train(double* x, double* y, int num_train, double grad_rate) {
    if(0 >= _num_hidden) {
        _softmax_layer->train(x, y, num_train, grad_rate);
        return;
    }
    // x -> hidden -> softmax
    forward_propagation(x, _hidden_layers[0]);
    for(int i = 1; i < _num_hidden; ++i)
        forward_propagation(_hidden_layers[i-1], _hidden_layers[i]);
    forward_propagation(_hidden_layers[_num_hidden-1], _softmax_layer);

    // y -> softmax -> hidden -> x
    _softmax_layer->back_propagation(_hidden_layers[_num_hidden-1]->_output_data, y, num_train, grad_rate);
    if(1 == _num_hidden) {
        back_propagation(x, _hidden_layers[0], _softmax_layer, grad_rate); 
        return;
    }
    back_propagation(_hidden_layers[_num_hidden-2], _hidden_layers[_num_hidden-1], _softmax_layer, grad_rate); 
    for(int i = _num_hidden - 2; i > 0; --i)
        back_propagation(_hidden_layers[i-1], _hidden_layers[i], _hidden_layers[i+1], grad_rate);
    back_propagation(x, _hidden_layers[0], _hidden_layers[1], grad_rate);
}
int NeuralNetwork::predict(double* x) {
    if(0 >= _num_hidden) {
        _softmax_layer->predict(x);
        return -1;
    }
    _hidden_layers[0]->forward_propagation(x);
    for(int i = 1; i < _num_hidden; ++i)
        _hidden_layers[i]->forward_propagation(_hidden_layers[i-1]->_output_data);
    return _softmax_layer->predict(_hidden_layers[_num_hidden-1]->_output_data);
}
void NeuralNetwork::dump_train_data(const char* file_name) {
    for(int i = 0; i < _num_hidden; ++i)
        _hidden_layers[i]->dump_train_data(file_name);
    _softmax_layer->dump_train_data(file_name);
}
long NeuralNetwork::load_train_data(const char* file_name, long offset) {
    long off = offset, read_size;
    for(int i = 0; i < _num_hidden; ++i) {
        read_size = _hidden_layers[i]->load_train_data(file_name, off);
        CHECK_RET(read_size, -1);
        off += read_size;
    }
    read_size = _softmax_layer->load_train_data(file_name, off);
    CHECK_RET(read_size, -1);
    return off += read_size; 
}
