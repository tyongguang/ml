#ifndef UTIL_H
#define UTIL_H
#include <iostream>
using namespace std;

#define NEW_POINTERV(ptr, type, size) \
ptr = new type [size]
#define NEW_POINTERM(ptr, type, size_row, size_col) \
ptr = new type* [size_row]; \
for(int i = 0; i < size_row; ++i) \
    ptr[i] = new type [size_col];
#define FREE_POINTER(ptr) \
if (NULL != ptr) {delete ptr; ptr = NULL;}
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
#define CHECK_RELATION_BASE(x, y, r, ...) \
if (x r y) { \
    __VA_ARGS__; \
}
#define CHECK_RELATION(x, y, r, ...) CHECK_RELATION_BASE(x, y, r, return __VA_ARGS__)
#define MAX(a,b)  ((a)>(b)?(a):(b))
#define MIN(a,b)  ((a)<(b)?(a):(b))
struct Shape4d {
    int m;
    int n;
    int num;
    int c;
};
template <typename T>
void print_data(T* vec, int size) {
    CHECK_PTR_BASE(vec, return);
    for (int i = 0; i < size; ++i)
        cout << *(vec + i) << ' ';
    cout << endl;
}
template <typename T>
void print_data(T* matrix, int row_size, int col_size) {
    CHECK_PTR_BASE(matrix, return);
    for (int i = 0; i < row_size; ++i) {
        for (int j = 0; j < col_size; ++j)
            cout << *(matrix + i * col_size + j) << ' ';
        cout << endl;
    }
}
template <typename T>
void print_data(T** matrix, int row_size, int col_size) {
    CHECK_PTR_BASE(matrix, return);
    for (int i = 0; i < row_size; ++i) {
        for (int j = 0; j < col_size; ++j)
            cout << matrix[i][j] << ' ';
        cout << endl;
    }
}
template <typename T>
void print_data(T* vec, int size, char split) {
    CHECK_PTR_BASE(vec, return);
    for (int i = 0; i < size; ++i)
        cout << *(vec + i) << split;
    cout << endl;
}
template <typename T>
void print_data(T* matrix, int row_size, int col_size, char split) {
    CHECK_PTR_BASE(matrix, return);
    for (int i = 0; i < row_size; ++i) {
        for (int j = 0; j < col_size; ++j)
            cout << *(matrix + i * col_size + j) << split;
        cout << endl;
    }
}
template <typename T>
void print_data(T** matrix, int row_size, int col_size, char split) {
    CHECK_PTR_BASE(matrix, return);
    for (int i = 0; i < row_size; ++i) {
        for (int j = 0; j < col_size; ++j)
            cout << matrix[i][j] << split;
        cout << endl;
    }
}
template <typename T>
void load_matrix(const char* file_name, T* matrix, int row_size, int col_size) {
    int i = 0;
    FILE *fp = 0;
    if((fp = fopen(file_name, "r")) == NULL) {
        printf("file open error: %s\n", file_name);
        return;
    }
    while(!feof(fp)) {
        char c = fgetc(fp);
        if('0' > c || '9' < c)
            continue;
        *(matrix + col_size * (int)(i / col_size) + i % col_size) = (T)(c - 48);
        i++;
    }
    fclose(fp);
}
template <typename T>
void load_matrix(const char* file_name, T** matrix, int row_size, int col_size) {
    int i = 0;
    FILE *fp = 0;
    if((fp = fopen(file_name, "r")) == NULL) {
        printf("file open error: %s\n", file_name);
        return;
    }
    while(!feof(fp)) {
        char c = fgetc(fp);
        if('0' > c || '9' < c)
            continue;
        matrix[(int)(i / col_size)][i % col_size] = (T)(c - 48);
        i++;
    }
    fclose(fp);
}
template <typename T>
void clip_alpha(T& a, T h, T l) {
    if(a > h)
        a = h;
    else if(l > a)
        a = l;
}
template <typename T>
void clip_alpha(T& a, bool b_h, bool b_l, T h, T l) {
    if(b_h)
        a = h;
    else if(b_l)
        a = l;
}
void set_value(double value, double* vec, int size);
void set_value(double value, double** matrix, int row_size, int col_size);
void crt_data(double* x, int size);
void crt_data(double** x, int row_size, int col_size);
double copy(double* x, double* y, int size);
double copy(double** x, double** y, int row_size, int col_size);
double add(double* x, double* y, int size);
double sum(double* vec, int size);
double sum(double* x, double* y, int size);
void sum(double* vec, double* sum_vec, int size_vec, int size_sum);
double transpose(double* x, double* y, int size_m, int size_n);
double rbf(double x, double delta);
void rbf(double* x, double* y, double* k, int size, int num_x, double delta);
void rbf(double* x, double* z, int size, int num_x, double delta) ;
double rbf(double* x, double* y, int size, double delta);
double distance(double* x, double* y, int size);
void distance(double* x, double* y, double* z, int size, int num_x);
void means(double* x, double* y, int size, int num_x);
void minos(double* x, double* y, int size);
void minos(double* x, double* y, double* z, int size);
void minos(double* x, double* y, int size, int num_x);
void minos(double* x, double* y, double* z, int size, int num_x);
void multi(double* x, double* y, double* z, int size_m, int size_n, int size_o);
void multi(double* x, double* y, double* z, int size_m, int size_n);
void multi(double* x, double* z, int size, int num);
double sec(double x);
double uniform(double min, double max);
void mat_func(double* vec, int size, double (*func)(double x));
int arg_max(double* array, int size);
int arg_min(double* array, int size);
double sigmoid(double x);
void softmax(double* x, int size);
void dump_vec(const char* file_name, double* vec, int size);
void dump_matrix(const char* file_name, double** matrix, int row_size, int col_size);
void dump_matrix(const char* file_name, double* matrix, int row_size, int col_size);
long load_vec(const char* file_name, double* vec, int size, long offset);
long load_matrix(const char* file_name, double** matrix, int row_size, int col_size, long offset);
long load_matrix(const char* file_name, double* matrix, int row_size, int col_size, long offset);
void list_folder_files(const char* path);
void txt2pb(const char* txt_file_name, const char* pb_file_name, int row_size, int col_size);
void folder_treat(const char* folder_path, void (*func)(const char* file_name));
void folder_txt2pb(const char* txt_folder_path, const char* pb_folder_path, int row_size, int col_size);
void load_heart_scale(const char* file, double* data, int* label, int size_in);
void conv2d(double* x, double* y, double* o, Shape4d shape_x, Shape4d shape_y);
double** conv2d(double** x, double** y, Shape4d shape_x, Shape4d shape_y);
void conv2d(double** x, double** y, double** o, Shape4d shape_x, Shape4d shape_y);
void conv2d(double** x, double** y, double* o, Shape4d shape_x, Shape4d shape_y);
void conv2d(double* x, double** y, double* o, Shape4d shape_x, Shape4d shape_y);
void conv2d(double* x, double** y, double** o, Shape4d shape_x, Shape4d shape_y);
void conv2d(double* x, double* y, double** o, Shape4d shape_x, Shape4d shape_y);
void conv2d_inv(double* x, double* y, double* o, Shape4d shape_x, Shape4d shape_y);
Shape4d padding(double* x, double* o, Shape4d shape_x, Shape4d shape_y);
void pool_max(double* x, double* o, Shape4d shape_x, Shape4d shape_y);
void pool_max(double* x, double* o, int* pos, Shape4d shape_x, Shape4d shape_y);
void pool_max(double* x, double* o, double* pos, Shape4d shape_x, Shape4d shape_y);
void pool_max(double* x, double* o, Shape4d shape_x, Shape4d shape_y, double b);
void pool_max(double* x, double* o, int* pos, Shape4d shape_x, Shape4d shape_y, double b);
void pool_max(double* x, double* o, double* pos, Shape4d shape_x, Shape4d shape_y, double b);
void pool_max(double* x, double* o, Shape4d shape_x, Shape4d shape_y, double* b);
void pool_max(double* x, double* o, int* pos, Shape4d shape_x, Shape4d shape_y, double* b);
void pool_max(double* x, double* o, double* pos, Shape4d shape_x, Shape4d shape_y, double* b);
void pool_max(double* x, double* o, double** pos, Shape4d shape_x, Shape4d shape_y, double* b);
void pool_max_inverse(double* x, double* o, Shape4d shape_x, Shape4d shape_y);
void pool_max_inverse(double* x, int* pos, double* o, Shape4d shape_x, Shape4d shape_y);
void pool_max_inverse(double* x, double* pos, double* o, Shape4d shape_x, Shape4d shape_y);
void pool_max_inverse(double* x, double** pos, double* o, Shape4d shape_x, Shape4d shape_y);
void rot180(double* x, Shape4d shape);
void rot180(double* x, double*y, Shape4d shape);
void rot180(double** x, double** y, Shape4d shape);
void rot180(double** x, Shape4d shape);
Shape4d reshape(Shape4d shape);
void reshape(double* x, double*y, Shape4d shape);
void reshape(double** x, double**y, Shape4d shape);
void reshape_a(double** x, double**y, Shape4d shape);
Shape4d shape_2d(int m, int n, int num, int c);
void shape_print(Shape4d shape);
void shape_print(Shape4d* shape);
int size_shape(Shape4d shape);
int size3_shape(Shape4d shape);
int size3c_shape(Shape4d shape);
int size2_shape(Shape4d shape);
int size2c_shape(Shape4d shape);
void shape_copy(Shape4d& shape_x, Shape4d& shape_y);
int select_j_rand(int i, int m);
void load_digits(double* input_data, int label, int size_row, int size_col, int num);
#endif
