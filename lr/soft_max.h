/**
 * @file soft_max.h
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/05/27 16:48:31
 * @brief 
 *  
 **/
#ifndef SOFT_MAX_H
#define SOFT_MAX_H

class SoftMax {
public:
	SoftMax(int size_x, int size_y);
	~SoftMax();

	void forward_propagation(double* input_data);
	void back_propagation(double* input_data, double* label, int num_train, double grad_rate);
	void train(double* x, double* y, int num_train, double grad_rate);
	void train(double* x, double* y, int num_train, double grad_rate, int num_iter);
	int predict(double* x);
	int predict(double* x, int num_test);
	double cal_error(double** ppdtest, double* pdlabel, int ibatch);

	void print();
	void dump_train_data(const char* file_name);
	long load_train_data(const char* file_name, long offset);

	//本层前向传播的输出值，也是最终的预测值
	double* _output_data;
	//反向传播  差值
	double* _delta;
public:
	int _size_in;
	int _size_out;
	double** _W;
	double* _B;
};
#endif

