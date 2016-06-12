/**
 * @file decision_tree.h
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/06/07 11:31:27
 * @brief 
 *  
 **/

#ifndef  __DECISION_TREE_H_
#define  __DECISION_TREE_H_

class DecisionTree {
public:
    DecisionTree(int size_in, int size_out);
    ~DecisionTree();
    void train(double* x, double* y, int num_in);
    void train(double* x, double* y, int* pos, int num_in, int h, int it);
    int predict(double* x);
private:
    int _size_in;
    int _num_in;
    int _size_out;
    int _h_e;
    int* _e;
};
#endif  //__DECISION_TREE_H_
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
