/**
 * @file smo.cpp
 * @author yuyimiao(472891913@qq.com)
 * @date 2016/05/27 18:28:19
 * @brief 
 *  
 **/
#include "smo.h"
#include <cmath>
#include "util.h"

SMO::SMO(double C, double eps, double T) {
    _C = C;
    _eps = eps;
    _T = T;
    _b = 0.0;
}
void SMO::LH() {
    if(_y1 == _y2) {
        _L = MAX(_alpha2 + _alpha1 - _C, 0);
        _H = MIN(_alpha1 + _alpha2, _C);
    }
    else {
        _L = MAX(_alpha2 - _alpha1, 0);
        _H = MIN(_C, _C + _alpha1 + _alpha2);
    }
}
void SMO::a2_pd() {
    _a2 = _alpha2 + _y2 * (_E1 - _E2) / _eta;
    clip_alpha(_a2, _H, _L);
}
void SMO::a2_nd() {
    double f1 = _y1 * (_E1 + _b) - _alpha1 * _K11 - _s * _alpha2 * _K12;
    double f2 = _y2 * (_E2 + _b) - _s * _alpha1 * _K12 - _alpha2 * _K22;
    double L1 = _alpha1 + _s * (_alpha2 - _L);
    double H1 = _alpha1 + _s * (_alpha2 - _H);
    double Psi_L = L1 * f1 + _L * f2 + (L1 * L1 * _K11 + _L * _L * _K22) / 2 + _s * _L * L1 * _K12;
    double Psi_H = H1 * f1 + _H * f2 + (H1 * H1 * _K11 + _H * _H * _K22) / 2 + _s * _H * H1 * _K12;
    _a2 = _alpha2;
    clip_alpha(_a2, (Psi_L < Psi_H - _eps), (Psi_L > Psi_H + _eps), _H, _L);
}
void SMO::b() {
    double b1 = _E1 + _y1 * (_a1 - _alpha1) * _K11 + _y2 * (_a2 - _alpha2) * _K12 + _b;
    double b2 = _E2 + _y1 * (_a1 - _alpha1) * _K12 + _y2 * (_a2 - _alpha2) * _K22 + _b;
    if(0 < _a1 && _a1 < _C)
        _b = b1;
    else if(0 < _a2 && _a2 < _C)
        _b = b2;
    else
        _b = (b1 + b2) / 2.0;
}
bool SMO::optimize() {
    LH();
    CHECK_RELATION(_L, _H, ==, false);
    eta(); s();
    clip_alpha(_E1, (_err_o > 0 && _err_o < _C), true, _err_o, _err_n);
    a2(); 
    CHECK_RELATION(abs(_a2 - _alpha2), _eps * (_a2 + _alpha2 + _eps), <, false);  
    a1(); b(); t1(); t2();
    return true;
}
bool SMO::optimize(int i, int j) {
    CHECK_RELATION(i, j, ==, 0);
    _alpha1 = _alphas[i];
    _alpha2 = _alphas[j];
    _y1 = _label[i];
    _y2 = _label[j];
    _K11 = _K[i * _num_in + i];
    _K12 = _K[i * _num_in + j];
    _K22 = _K[j * _num_in + j];
    if(!optimize())
        return false;
    _alphas[i] = _a1;
    _alphas[j] = _a2;
    _error[i] = _t1;
    _error[j] = _t2;
    return true;
}
int SMO::optimize(int i) {
    E1(i);
    double ri = _E1 * _label[i];
    if((ri < -_T && _alphas[i] < _C) || (ri > _T && _alphas[i] > 0)) {
        int j = select_j_max_fabs(i);
        if(0 <= j && optimize(i, j))
            return 1;
        j = select_j_rand(i, _size_in);
        if(0 <= j && optimize(i, j))
            return 1;
    }
    return 0;
}
void SMO::process(int iterator) {
    int num_change = 0;
    bool entire_set = true;
    int iter = 0;
    while(iter < iterator && (num_change > 0 || entire_set)) {
        num_change = 0;
        if (entire_set) {
            for(int i = 0; i < _num_in; ++i)
                num_change += optimize(i);
        }
        else {
            for(int i = 0; i < _num_in; ++i) {
                if(0 < _alphas[i] && _alphas[i] < _C)
                    num_change += optimize(i);
            }
            iter += 1;
        }
        if(entire_set)
            entire_set = false;
        else if(!num_change)
            entire_set = true;
    }
}
double SMO::predict(int i) {
    double output = 0;
    for(int k = 0; k < _num_in; k++)
        if(_alphas[k] > 0)
            output += _alphas[k] * _label[k] * _K[k * _num_in + i];
    return output -= _b;
}
int SMO::select_j_max_fabs(int i) {
    int j = -1;
    double max = 0;
    for(int k = 0; k < _num_in; k++) {
        if(_alphas[k] <= 0 || _alphas[k] >= _C) {
            continue;
        }
        _E2 = _error[k];
        double temp = fabs(_E1 - _E2);
        if(temp <= max) {
            continue;
        }
        max = temp;
        j = k;
    }
    return j;
}
double SMO::error_rate() {
    int count = 0;
    for(int i = 0; i < _num_in; ++i)
        if(predict(i) * _label[i] > 0) 
            count++;
    return (double) count / (_num_in);
}
double SMO::calc_Ei(int i) {
    return predict(i) - _label[i];
}
void SMO::E1(int i) {
    _err_o = _error[i];
    _err_n = calc_Ei(i);
    clip_alpha(_E1, (_err_o > 0 && _err_o < _C), true, _err_o, _err_n);
}
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
