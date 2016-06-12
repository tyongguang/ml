#ifndef UTIL_H
#define UTIL_H
#include <iostream>
using namespace std;

#define NEW_POINTERV(ptr, type, size) \
ptr = new type [size]
#define NEW_POINTERM(ptr, type, size_row, size_col) \
ptr = new type* [size_col]; \
for(int i = 0; i < size_col; ++i) \
    ptr[i] = new type [size_row]
#define FREE_POINTER(ptr) \
if (NULL != ptr) {free(ptr); ptr = NULL;}
#define FREE_POINTERV(ptr) \
if (NULL != ptr) {delete[] ptr; ptr = NULL;}
#define FREE_POINTERM(ptr,size) \
if (NULL != ptr) { \
    for(int i = 0; i < size; ++i) { \
        delete[] ptr[i]; \
    } \
    ptr = NULL; \
}
#define SHOW_VAR_NAME(var) printf("cur var: %s\n", #var)
#define CHECK_PTR_BASE(ptr, ...) \
if (NULL == ptr) { \
    printf("[%s] is NULL. %s\n", #ptr, #__VA_ARGS__); \
    __VA_ARGS__; \
}
#define CHECK_PTR(ptr, ...) CHECK_PTR_BASE(ptr, return __VA_ARGS__)
#define CHECK_CHAR_PTR_BASE(ptr, ...) \
if (NULL == ptr || 0 == strlen(ptr)) { \
    printf("char [%s] is NULL or zero len.\n", #ptr); \
    __VA_ARGS__; \
} 
#define CHECK_CHAR_PTR(ptr, ...) CHECK_CHAR_PTR_BASE(ptr, return __VA_ARGS__)
#define CHECK_RET_BASE(x, ...) \
if (0 >= x) { \
    printf("[%s] <= 0. %s\n", #x, #__VA_ARGS__); \
    __VA_ARGS__; \
}
#define CHECK_RET(x, ...) CHECK_RET_BASE(x, return __VA_ARGS__)
#define CHECK_RELATION_BASE(x, y, ...) \
if (x > y) { \
    printf("[%s] > [%s]. %s\n", #x, #y, #__VA_ARGS__); \
    __VA_ARGS__; \
}
#define CHECK_RELATION(x, y, ...) CHECK_RELATION_BASE(x, y, return __VA_ARGS__)
template <typename T>
void print_data(T* vec, int size) {
    CHECK_PTR_BASE(vec, return);
    for (int i = 0; i < size; ++i)
        cout << *(vec + i) << ' ';
    cout << endl;
}
template <typename T>
void print_data(T** matrix, int row_size, int col_size) {
    CHECK_PTR_BASE(matrix, return);
    for (int i = 0; i < row_size; ++i) {
        for (int j = 0; j < col_size; ++j)
            cout << *(matrix + i * col_size + j) << ' ';
        cout << endl;
    }
}
void set_value(double value, double* vec, int size);
void set_value(double value, double** matrix, int row_size, int col_size);
double uniform(double min, double max);
int arg_max(double* array, int size);
double sigmoid(double x);
void softmax(double* x, int size);
void dump_vec(const char* file_name, double* vec, int size);
void dump_matrix(const char* file_name, double** matrix, int row_size, int col_size);
long load_vec(const char* file_name, double* vec, int size, long offset);
long load_matrix(const char* file_name, double** matrix, int row_size, int col_size, long offset);
#endif
