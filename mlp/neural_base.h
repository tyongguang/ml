/**
 * @file neural_base.h
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/05/27 16:48:31
 * @brief 
 *  
 **/
#ifndef NEURAL_BASE_H
#define NEURAL_BASE_H

class NeuralBase {
public:
    NeuralBase();
    NeuralBase(int size_x, int size_y);
    virtual ~NeuralBase();

    virtual void forward_propagation(double* input_data){};
    virtual void forward_propagation(NeuralBase* layer_pre){
        forward_propagation(layer_pre->_output_data);
    };
    virtual void forward_propagation(double* input_data, NeuralBase* layer_cur){
        layer_cur->forward_propagation(input_data);
    };
    virtual void forward_propagation(NeuralBase* layer_pre, NeuralBase* layer_cur){
        layer_cur->forward_propagation(layer_pre->_output_data);
    };
    virtual void back_propagation(double* input_data, double* label, int num_train, double grad_rate){};
    virtual void back_propagation(double* input_data, NeuralBase* layer_next, double grad_rate){};
    virtual void back_propagation(double* input_data, NeuralBase* layer_cur, NeuralBase* layer_next, double grad_rate){
        layer_cur->back_propagation(input_data, layer_next, grad_rate);
    };
    virtual void back_propagation(NeuralBase* layer_pre, NeuralBase* layer_next, double grad_rate){};
    virtual void back_propagation(NeuralBase* layer_pre, NeuralBase* layer_cur, NeuralBase* layer_next, double grad_rate){
        layer_cur->back_propagation(layer_pre, layer_next, grad_rate);
    };
    virtual void train(double* x, double* y, int num_train, double grad_rate);
    virtual void train(double* x, double* y, int num_train, double grad_rate, int num_iter);
    virtual int predict(double* x);
    virtual int predict(double* x, int num_test);
    virtual double cal_error(double** ppdtest, double* pdlabel, int ibatch);

    virtual void print();
    virtual void dump_train_data(const char* file_name);
    virtual long load_train_data(const char* file_name, long offset);

    //本层前向传播的输出值，也是最终的预测值
    double* _output_data;
    //反向传播 差值
    double* _delta;
public:
    int _size_in;
    int _size_out;
    double** _W;
    double* _B;
};
#endif

