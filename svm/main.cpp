/**
 * @file main.cpp
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/05/27 16:48:31
 * @brief 
 *  
 **/
#include <iostream>
#include "svm.h"
#include "util.h"
using namespace std;
void test_svm();
void test_multi();
void test_rbf();

int main(void) {
    test_multi();
    test_rbf();
    test_svm();
}

void test_rbf() {
    double a[2][3] = {
        {0, 1, 2},
        {2, 1, 0}
};
    double b[2][2] = {0};
    rbf(&a[0][0], &b[0][0], 3, 2, 1);
    print_data(&b[0][0], 2, 2);
}
void test_multi() {
    double a[2][3] = {
        {0, 1, 2},
        {2, 1, 0}
};
    double b[2][2] = {0};
    multi(&a[0][0], &b[0][0], 3, 2);
    print_data(&b[0][0], 2, 2);
}
void test_svm() {
    svm_t para;
    para._num_in = 270;
    para._size_in = 13;
    para._C = 1.0;
    para._T = 0.001;
    para._eps = 1.0E-12;
    para._delta = 2.0;

    SVM svm(para);
    double input_data[13 * 270] = {0};
    int label[270] = {0};
    load_heart_scale("heart_scale", &input_data[0], &label[0], 13);
    svm.train(&input_data[0], &label[0], 50000);
    cout << "ÕýÈ·ÂÊ: " << svm.error_rate() * 100 << "£¥" << endl;

    double test1[] = {-0.291667, 1, 1, -0.132075, -0.155251, -1, -1, -0.251908, 1, -0.419355, 0, 0.333333, 1};
    double test2[] = { -0.166667, 1, -0.333333, -0.320755, -0.360731, -1, -1, 0.526718, -1, -0.806452, -1, -1, -1,};
    cout<< svm.predict(test1) << "\t1" << endl;
    cout<< svm.predict(test2) << "\t-1" << endl;
}

