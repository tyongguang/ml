/**
 * @file neural_network.cpp
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/05/27 16:48:31
 * @brief 
 *  
 **/
#include "util.h"
#include <cmath>
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
MLP::MLP(int size_x, int size_y, int* struct_hidden) : NeuralBase(size_x, size_y) {
    _num_hidden = sizeof(struct_hidden) / sizeof(int);
    _hidden_layers = new HiddenLayer* [_num_hidden];
    _hidden_layers[0] = new HiddenLayer(_size_in, struct_hidden[0]);
    for(int i = 1; i < _num_hidden; ++i)
        _hidden_layers[i] = new HiddenLayer(struct_hidden[i-1], struct_hidden[i]);
    _softmax_layer = new SoftMax(struct_hidden[_num_hidden-1], _size_out);
}
MLP::~MLP() {
    FREE_POINTERV(_hidden_layers);
    FREE_POINTER(_softmax_layer);
}
void MLP::train(double* x, double* y, int num_train, double grad_rate) {
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
int MLP::predict(double* x) {
    if(0 >= _num_hidden) {
        _softmax_layer->predict(x);
        return -1;
    }
    _hidden_layers[0]->forward_propagation(x);
    for(int i = 1; i < _num_hidden; ++i)
        _hidden_layers[i]->forward_propagation(_hidden_layers[i-1]->_output_data);
    return _softmax_layer->predict(_hidden_layers[_num_hidden-1]->_output_data);
}
void MLP::dump_train_data(const char* file_name) {
    for(int i = 0; i < _num_hidden; ++i)
        _hidden_layers[i]->dump_train_data(file_name);
    _softmax_layer->dump_train_data(file_name);
}
long MLP::load_train_data(const char* file_name, long offset) {
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
ConvLayer::ConvLayer(Shape4d shape_img, Shape4d shape_kernel) {
    shape_print(shape_img);
    shape_print(shape_kernel);
    shape_copy(shape_img, _shape_image);
    shape_copy(shape_kernel, _shape_kernel);
    shape_print(_shape_image);
    shape_print(_shape_kernel);
    _shape_out = shape_2d(shape_img.m - shape_kernel.m + 1, 
                                 shape_img.n - shape_kernel.n + 1, 
                                 shape_img.num, 
                                 shape_kernel.num);
    _size_in = shape_kernel.num;
    _size_w = size3_shape(shape_kernel);
    _size_out = size_shape(_shape_out);
    cout << shape_kernel.num << " " << size3_shape(shape_kernel) << " " << size_shape(_shape_out) << endl;
    cout << _size_in << " " << _size_w << " " << _size_out << endl;

    NEW_POINTERM(_W, double, _size_in, _size_w);
    NEW_POINTERV(_B, double, _size_in);
    NEW_POINTERV(_delta, double, _size_out);
    NEW_POINTERV(_output_data, double, _size_out);

    double uniform_param = 1.0 / _size_w;
    for(int i = 0; i < _size_in; ++i)
        for(int j = 0; j < _size_w; ++j)
            _W[i][j] = uniform(-uniform_param, uniform_param);
    set_value(0.0, _B, _size_in);
}
ConvLayer::~ConvLayer() {
    FREE_POINTERM(_W, _size_in);
}
void ConvLayer::forward_propagation(double* input_data) {
    conv2d(input_data, _W, _output_data, _shape_image, _shape_kernel);
    //mat_func(_output_data, _size_out, tanh);
}
void ConvLayer::back_propagation(double* input_data, double* delta_next, double** W_next, Shape4d shape_delta_next, Shape4d shape_W_next) {
    pool_max_inverse(delta_next, W_next, _delta, shape_delta_next, shape_W_next);
    mat_func(_delta, _size_out, sec);

    double* input_data_resize = 0;
    double* delta_rot = 0;
    double** W_tmp = 0;
    NEW_POINTERV(input_data_resize, double, size_shape(_shape_image));
    NEW_POINTERV(delta_rot, double, _size_out);
    NEW_POINTERM(W_tmp, double,_shape_kernel.c, size3c_shape(_shape_kernel));

    reshape(input_data, input_data_resize, _shape_image);
    reshape(_delta, delta_rot, _shape_out);
    conv2d(input_data_resize, delta_rot, W_tmp, _shape_image, _shape_out);
    reshape_a(W_tmp, _W, reshape(_shape_kernel));

    FREE_POINTERV(input_data_resize);
    FREE_POINTERV(delta_rot);
    FREE_POINTERM(W_tmp, _shape_kernel.c);
}
PoolLayer::PoolLayer(Shape4d shape_img, Shape4d shape_kernel) {
    shape_copy(shape_img, _shape_image);
    shape_copy(shape_kernel, _shape_kernel);
    _shape_out = shape_2d(shape_img.m / shape_kernel.m, 
                                 shape_img.n / shape_kernel.n, 
                                 shape_img.num, 
                                 shape_img.c);
    _size_in = size2c_shape(shape_img);
    _size_out = size_shape(_shape_out);

    NEW_POINTERM(_W, double, _size_in, size2_shape(_shape_out));
    NEW_POINTERV(_B, double, _size_in);
    NEW_POINTERV(_delta, double, _size_out);
    NEW_POINTERV(_output_data, double, _size_out);

    set_value(0.0, _W, _size_in, size2_shape(_shape_out));
    double uniform_param = 1.0 / _size_in;
    for(int i = 0; i < _size_in; ++i)
        _B[i] = uniform(-uniform_param, uniform_param);
}
PoolLayer::~PoolLayer() {
    FREE_POINTERM(_W, _size_in);
}
void PoolLayer::forward_propagation(double* input_data) {
    pool_max(input_data, _output_data, _W, _shape_image, _shape_kernel, _B);
}
void PoolLayer::back_propagation(double* delta_next, double** W_next, int size_out_next) {
    double* sigma;
    NEW_POINTERV(sigma, double, _size_out);
    set_value(0.0, sigma, _size_out);

    for(int i = 0; i < size_out_next; ++i)
        for(int j = 0; j < _size_out; ++j)
            sigma[j] += W_next[i][j] * delta_next[i];

    for(int i = 0; i < _size_out; ++i)
        _delta[i] = sigma[i] * _output_data[i] * (1 - _output_data[i]);

    sum(_delta, _B, _size_out, _size_in);
}
void PoolLayer::back_propagation(double* delta_next, double** W_next, Shape4d shape_delta_next, Shape4d shape_W_next) {
    Shape4d shape_padding = shape_2d(shape_delta_next.m + 2 * (shape_W_next.m - 1), 
                                     shape_delta_next.n + 2 * (shape_W_next.n - 1), 
                                     shape_delta_next.num, 
                                     shape_delta_next.c);
    double* delta_next_padding = 0;
    NEW_POINTERV(delta_next_padding, double, size_shape(shape_padding));
    padding(delta_next, delta_next_padding, shape_delta_next, shape_W_next);
    double** W_next_rot = 0;
    NEW_POINTERM(W_next_rot, double, shape_W_next.num, size3_shape(shape_W_next));
    rot180(W_next, W_next_rot, shape_W_next);

    conv2d(delta_next_padding, W_next_rot, _delta, shape_padding, shape_W_next);
    //sum(_delta, _B, size_shape(shape_delta_next), _size_out);
    sum(_delta, _B, _size_out, _size_in);

    FREE_POINTERV(delta_next_padding);
    FREE_POINTERM(W_next_rot, shape_W_next.num);
}
CnnLayer::CnnLayer(Shape4d shape_img, Shape4d* shape_kernel) {
    _conv_layer = new ConvLayer(shape_img, shape_kernel[0]);
    _pool_layer = new PoolLayer(_conv_layer->_shape_out, shape_kernel[1]);
}
CnnLayer::~CnnLayer() {
    FREE_POINTER(_conv_layer);
    FREE_POINTER(_pool_layer);
    _W = 0;
    _B = 0;
    _output_data = 0;
    _delta = 0;
}
void CnnLayer::forward_propagation(double* input_data) {
    _conv_layer->forward_propagation(input_data);
    _pool_layer->forward_propagation(_conv_layer->_output_data);
    _output_data = _pool_layer->_output_data;
}
void CnnLayer::back_propagation(double* input_data, double* delta_next, double** W_next, Shape4d shape_delta_next, Shape4d shape_W_next) {
    _pool_layer->back_propagation(delta_next, W_next, shape_delta_next, shape_W_next);
    _conv_layer->back_propagation(input_data, _pool_layer->_delta, _pool_layer->_W, _pool_layer->_shape_out, _pool_layer->_shape_kernel);
}
void CnnLayer::back_propagation(double* input_data, CnnLayer* cnn_next) {
    back_propagation(input_data, cnn_next->_conv_layer->_delta, cnn_next->_conv_layer->_W, cnn_next->_conv_layer->_shape_out, cnn_next->_conv_layer->_shape_kernel);
}
void CnnLayer::back_propagation(double* input_data, double* delta_next, double** W_next, int size_out_next) {
    _pool_layer->back_propagation(delta_next, W_next, size_out_next);
    _conv_layer->back_propagation(input_data, _pool_layer->_delta, _pool_layer->_W, _pool_layer->_shape_out, _pool_layer->_shape_kernel);
}
void CnnLayer::back_propagation(double* input_data, NeuralBase* nn_next) {
    back_propagation(input_data, nn_next->_delta, nn_next->_W, nn_next->_size_out);
}
Cnn::Cnn(Shape4d shape_img, Shape4d* shape_struct, int num_cnn, int size_out) {
    _size_out = size_out;
    _size_in = size2_shape(shape_img);
    //_num_cnn = sizeof(shape_struct) / (2 * sizeof(Shape4d));
    _num_cnn = num_cnn;
    _cnn_layers = new CnnLayer* [_num_cnn];
    _cnn_layers[0] = new CnnLayer(shape_img, &shape_struct[0]);
    for(int i = 1; i < _num_cnn; ++i) {
        shape_img = shape_2d(shape_img.m - shape_struct[i].m,
                             shape_img.n - shape_struct[i].n,
                             shape_img.num,
                             shape_struct[2 * i].num);
        _cnn_layers[i] = new CnnLayer(shape_img, &shape_struct[i]);
    }
    _nn_layer = new SoftMax(size3_shape(shape_img), _size_out);
}
Cnn::~Cnn() {
    FREE_POINTERV(_cnn_layers);
    _W = 0;
    _B = 0;
    _output_data = 0;
    _delta = 0;
    FREE_POINTER(_nn_layer);
}
void Cnn::train(double* x, double* y, int num_train, double grad_rate) {
    _cnn_layers[0]->forward_propagation(x);
    for(int i = 1; i < _num_cnn; ++i)
        _cnn_layers[i]->forward_propagation(_cnn_layers[i - 1]->_output_data);
    _nn_layer->predict(_cnn_layers[_num_cnn - 1]->_output_data);
    _nn_layer->back_propagation(_cnn_layers[_num_cnn - 1]->_output_data, y, num_train, grad_rate);
    if(1 == _num_cnn) {
        _cnn_layers[0]->back_propagation(x, _nn_layer);
        return;
    }
    _cnn_layers[_num_cnn - 1]->back_propagation(_cnn_layers[_num_cnn - 2]->_output_data, _nn_layer);
    for(int i = _num_cnn - 2; i > 0; --i)
        _cnn_layers[i]->back_propagation(_cnn_layers[i - 1]->_output_data, _cnn_layers[i + 1]);
    _cnn_layers[0]->back_propagation(x, _cnn_layers[1]);
}
int Cnn::predict(double* x) {
    _cnn_layers[0]->forward_propagation(x);
    for(int i = 1; i < _num_cnn; ++i)
        _cnn_layers[i]->forward_propagation(_cnn_layers[i - 1]->_output_data);
    return _nn_layer->predict(_cnn_layers[_num_cnn - 1]->_output_data);
}
