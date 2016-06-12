/**
 * @file main.cpp
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/06/06 14:59:31
 * @brief 
 *  
 **/
#include "util.h"
#include "decision_tree.h"
void test_f();
void test_dc();

int main(void) {
    test_f();
    test_dc();
    return 1;
}

void test_dc() {
    DecisionTree dc = DecisionTree(4, 2);
    double a[20] = {0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    double b[5] = {1, 0, 0, 0, 1};
    double c[4] = {0, 1, 0, 0};
    double d[4] = {1, 0, 1, 0};
    dc.train(&a[0], &b[0], 5);
    cout << "predict class: " << dc.predict(&c[0]) << endl;
    cout << "predict class: " << dc.predict(&d[0]) << endl;
}
void test_f() {
    double a[10] = {0, 0, 1, 1, 1, 1, 0, 1, 1, 1};
    int pos[5] = {0, 1, 2, 5, 6};
    double b[2] = {0};
    prob(&a[0], &b[0], 10, 2);
    print_data(&b[0], 2);
    prob(&a[0], &pos[0], &b[0], 5, 2);
    print_data(&b[0], 2);
    cout << "entropy: " << entropy(&a[0], 10, 2) << endl;
    cout << "entropy: " << entropy(&a[0], &pos[0], 5, 2) << endl;
    double c[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
    cout << "gain: " << gain(&a[0], &c[0], 10, 2) << endl;
    cout << "gain: " << gain(&a[0], &pos[0], &c[0], 5, 2) << endl;
    double d[20] = {0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    double f[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
    cout << "col: " << gain(&d[0], &c[0], 10, 2, 2) << endl;
    cout << "col: " << gain(&d[0], &pos[0], &c[0], 5, 2, 2) << endl;
    int exist[1] = {1};
    cout << "col: " << gain(&d[0], &pos[0], &c[0], &exist[0], 5, 2, 2, 1) << endl;
    int h[10] = {0};
    int g[2] = {0};
    split(&a[0], &c[0], 10, &h[0], &g[0], 2);
    print_data(&h[0], 10);
    print_data(&g[0], 2);
    double j[6] = {0, 1, 0, 1, 0, 1};
    double k[6] = {0};
    reshape(&j[0], &k[0], 2, 3);
    print_data(&k[0], 6);
}
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
