/**
 * @file main.cpp
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/06/05 14:53:09
 * @brief 
 *  
 **/
#include "kms.h"
#include "util.h"
using namespace std;
void test_kms();

int main(void) {
    test_kms();
    return 1;
}

void test_kms() {
    Kms kms = Kms(32 * 32, 2);
    double k[2][1024] = {0};
    load_digits(&k[0][0], 1, 32, 32, 1);
    load_digits(&k[1][0], 9, 32, 32, 1);
    kms.init_K(&k[0][0]);
    double a[40][1024] = {0};
    load_digits(&a[0][0], 1, 32, 32, 20);
    load_digits(&a[20][0], 9, 32, 32, 20);
    kms.train(&a[0][0], 40, 100);
    double c[32][32] = {0};
    load_matrix("digits_txt/9_11.txt", &c[0][0], 32, 32);
    cout << "predict class: " << kms.predict(&c[0][0]) << endl;
}
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
