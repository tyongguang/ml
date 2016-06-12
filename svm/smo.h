/**
 * @file smo.h
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/05/27 16:48:31
 * @brief 
 *  
 **/

#ifndef  __SMO_H_
#define  __SMO_H_
class SMO {
public:
    SMO(){};
    SMO(double C, double eps, double T);
    ~SMO(){};
    bool optimize();
    int optimize(int i);
    bool optimize(int i, int j);
    void process(int interator);
    double error_rate();

    double* _alphas;
    int* _label;
    double* _error;
    double* _K;
    int _num_in;
    int _size_in;

    double _alpha1;
    double _alpha2;
    double _y1;
    double _y2;
    double _err_o;
    double _err_n;
    double _K11;
    double _K12;
    double _K22;
    double _a1;
    double _a2;
    double _b;
    double _t1;
    double _t2;
    double _C;
    double _T;
private:
    void LH();
    void eta() {_eta = _K11 + _K22 - 2 * _K12;}
    void s() {_s = _y1 * _y2;}
    void a2(){if(0 < _eta) a2_pd(); else a2_nd();}
    void a2_pd();
    void a2_nd();
    void a1() {_a1 = _alpha1 + _s * (_alpha2 - _a2);}
    void b();
    void t1() {_t1 = _y1 * (_a1 - _alpha1);}
    void t2() {_t2 = _y2 * (_a2 - _alpha2);}
    int select_j_max_fabs(int i);
    double predict(int i);
    double calc_Ei(int i);
    void E1(int i);

    double _E1;
    double _E2;
    double _L;
    double _H;
    double _eta;
    double _s;
    double _eps;
};
#endif  //__SMO_H_
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
