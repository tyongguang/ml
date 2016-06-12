/**
 * @file main.cpp
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/06/05 14:53:09
 * @brief 
 *  
 **/
#include "knn.h"
#include "util.h"
using namespace std;
void test_knn();

int main(void) {
    test_knn();
    return 1;
}

void test_knn() {
    Knn knn = Knn(32 * 32, 2, 5);
    double a[40][1024] = {0};
    load_digits(&a[0][0], 1, 32, 32, 20);
    load_digits(&a[20][0], 9, 32, 32, 20);
    int b[40] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    knn.train(&a[0][0], &b[0], 40);
    double c[32][32] = {0};
    load_matrix("digits_txt/9_47.txt", &c[0][0], 32, 32);
    cout << "predict class: " << knn.predict(&c[0][0]) << endl;
}
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
