#include <ctime>
#include <cmath>
#include <string>
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <float.h>
#include "util.h"

using namespace std;

void swap(double* x, double* y) {
    double tmp = *x;
    *x = *y;
    *y = tmp;
}
int arg_max(double* array, int size) {
    CHECK_PTR_BASE(array, return -1);
    double max = -1;
    int idx_max = -1;
    for(int i = 0; i < size; ++i) {
        if (array[i] > max) {
            max = array[i];
            idx_max = i;
        }
    }
    return idx_max;
}
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
void softmax(double* x, int size) {                     
    CHECK_PTR_BASE(x, return);
    double max = 0.0;                                            
    double sum = 0.0;                                            
    for(int i = 0; i < size; ++i) {                              
        if(max < x[i])                                           
            max = x[i];                                          
    }                                                             
    for(int i = 0; i < size; ++i) {                              
        x[i] = exp(x[i] - max);                                    
        sum += x[i];                                             
    }                                                             
    for(int i = 0; i < size; ++i) 
        x[i] /= sum;                                             
}
double uniform(double min, double max) {
    return rand() / (RAND_MAX + 1.0) * (max - min) + min;
}
void set_value(double value, double* vec, int size) {
    CHECK_PTR_BASE(vec, return);
    for(int i = 0; i < size; ++i)
        vec[i] = value;
}
void crt_data(double* x, int size) {
    for(int i = 0; i < size; ++i)
        x[i] = i % 10;
}
void crt_data(double** x, int row_size, int col_size) {
    for(int i = 0; i < row_size; ++i)
        for(int j = 0; j < col_size; ++j)
            x[i][j] = (i + j) % 10;
}
double copy(double* x, double* y, int size) {
    for(int i = 0; i < size; ++i) 
        y[i] = x[i];
}
double copy(double** x, double** y, int row_size, int col_size) {
    for(int i = 0; i < row_size; ++i) 
        copy(x[i], y[i], col_size);
}
double add(double* x, double* y, int size) {
    for(int i = 0; i < size; ++i) 
        x[i] += y[i];
}
double sum(double* vec, int size) {
    double sum = 0.0;
    for(int i = 0; i < size; ++i)
        sum += vec[i];
    return sum;
}
void sum(double* vec, double* sum_vec, int size_vec, int size_sum) {
    int size_unit = size_vec / size_sum;
    for(int i = 0; i < size_sum; ++i)
        sum_vec[i] = sum(vec + i * size_unit, size_unit);
}
double sec(double x) {
    return 1 / (cos(x) * cos(x));
}
void mat_func(double* vec, int size, double (*func)(double x)) {
    CHECK_PTR_BASE(vec, return);
    for(int i = 0; i < size; ++i)
        vec[i] = func(vec[i]);
}
void set_value(double value, double** matrix, int row_size, int col_size) {
    CHECK_PTR_BASE(matrix, return);
    for(int i = 0; i < row_size; ++i)
        for(int j = 0; j < col_size; ++j)
            matrix[i][j] = value;
}
void dump_vec(const char* file_name, double* vec, int size) {
    CHECK_PTR_BASE(vec, return);
    CHECK_PTR_BASE(file_name, return);
    FILE* pf;
    if((pf = fopen(file_name, "ab")) == NULL) {
        printf("file could not be opened.\n");
        return;
    }
    int elem_size = sizeof(double);
    if (fwrite((const void*)vec, elem_size, size, pf) != size) {
        fputs ("dump vec error.\n",stderr);
        return;
    }
    fclose(pf);
}
void dump_matrix(const char* file_name, double** matrix, int row_size, int col_size) {
    CHECK_PTR_BASE(matrix, return);
    CHECK_PTR_BASE(file_name, return);
    FILE* pf;
    if((pf = fopen(file_name, "ab")) == NULL) { 
        printf("file could not be opened.\n"); 
        return;
    } 
    int elem_size = sizeof(double);
    for (int i = 0; i < row_size; ++i) {
        if (fwrite((const void*)matrix[i], elem_size, col_size, pf) != col_size) {
            fputs ("dump matrix error.\n", stderr);
            return;
        }
    }
    fclose(pf);
}
void dump_matrix(const char* file_name, double* matrix, int row_size, int col_size) {
    CHECK_PTR_BASE(matrix, return);
    CHECK_PTR_BASE(file_name, return);
    FILE* pf;
    if((pf = fopen(file_name, "ab")) == NULL) { 
        printf("file could not be opened.\n"); 
        return;
    } 
    int elem_size = sizeof(double);
    for (int i = 0; i < row_size; ++i) {
        if (fwrite((const void*)(matrix + i * col_size), elem_size, col_size, pf) != col_size) {
            fputs ("dump matrix error.\n", stderr);
            return;
        }
    }
    fclose(pf);
}
long load_vec(const char* file_name, double* vec, int size, long offset) {
    CHECK_PTR_BASE(vec, return -1);
    CHECK_PTR_BASE(file_name, return -1);
    FILE* pf;
    long read_size;
    if((pf = fopen(file_name, "rb")) == NULL) {
        printf("file could not be opened.\n");
        return -1;
    }
    //让文件指针偏移到正确位置
    fseek(pf, offset, SEEK_SET);
    int elem_size = sizeof(double);
    read_size = fread(vec, elem_size, size, pf);
    if (read_size != size) {
        fputs ("load vec error.\n",stderr);
        return -1;
    }
    fclose(pf);
    return size * elem_size;
}
long load_matrix(const char* file_name, double** matrix, int row_size, int col_size, long offset) {
    CHECK_PTR_BASE(matrix, return -1);
    CHECK_PTR_BASE(file_name, return -1);
    FILE* pf;
    long read_size;
    if((pf = fopen(file_name, "rb")) == NULL) { 
        printf("file could not be opened.\n"); 
        return -1;
    } 
    //让文件指针偏移到正确位置
    fseek(pf, offset, SEEK_SET);
    int elem_size = sizeof(double);
    for (int i = 0; i < row_size; ++i) {
        read_size = fread((void*)matrix[i], elem_size, col_size, pf);
        if (read_size != col_size) {
            fputs ("load matrix error.\n",stderr);
            return -1;
        }
    }
    fclose(pf);
    return row_size * col_size * elem_size;  
}
long load_matrix(const char* file_name, double* matrix, int row_size, int col_size, long offset) {
    CHECK_PTR_BASE(matrix, return -1);
    CHECK_PTR_BASE(file_name, return -1);
    FILE* pf;
    long read_size;
    if((pf = fopen(file_name, "rb")) == NULL) { 
        printf("file could not be opened.\n"); 
        return -1;
    } 
    //让文件指针偏移到正确位置
    fseek(pf, offset, SEEK_SET);
    int elem_size = sizeof(double);
    for (int i = 0; i < row_size; ++i) {
        read_size = fread((void*)(matrix + i * col_size), elem_size, col_size, pf);
        if (read_size != col_size) {
            fputs ("load matrix error.\n",stderr);
            return -1;
        }
    }
    fclose(pf);
    return row_size * col_size * elem_size;  
}
void list_folder_files(const char* path) {
    DIR* dp;
    struct dirent* dir;
    int n = 0;
    if((dp = opendir(path)) == NULL)
        printf("can't open %s\n", path);
    while (((dir = readdir(dp)) != NULL)) {
        char* file_name = dir->d_name;
        printf("%s\n",file_name);
    }
    closedir(dp);
}
void folder_treat(const char* path, void (*func)(const char* file_name)) {
    DIR* dp;
    struct dirent* dir;
    int n = 0;
    if((dp = opendir(path)) == NULL)
        printf("can't open %s\n", path);
    while (((dir = readdir(dp)) != NULL)) {
        char* file_name = dir->d_name;
        func(file_name);
    }
    closedir(dp);
}
void folder_txt2pb(const char* txt_folder_path, const char* pb_folder_path, int row_size, int col_size) {
    DIR* dp;
    struct dirent* dir;
    int n = 0;
    if((dp = opendir(txt_folder_path)) == NULL) {
        printf("can't open %s\n", txt_folder_path);
        return;
    }
    while (((dir = readdir(dp)) != NULL)) {
        string txt_file_name(dir->d_name);
        if(string::npos == txt_file_name.find(".txt", 0))
            continue;
        string pb_file_path(pb_folder_path);
        pb_file_path += string("/") + txt_file_name.substr(0, txt_file_name.size() - 3) + string("data");
        txt_file_name = string(txt_folder_path) + string("/") + txt_file_name;
        txt2pb(txt_file_name.c_str(), pb_file_path.c_str(), row_size, col_size);
    }
}
void txt2pb(const char* txt_file_name, const char* pb_file_name, int row_size, int col_size) {
    double** data = 0;
    NEW_POINTERM(data, double, row_size, col_size); 
    set_value(0.0, data, row_size, col_size);
    load_matrix(txt_file_name, data, row_size, col_size);
    dump_matrix(pb_file_name, data, row_size, col_size);
    FREE_POINTERM(data, col_size);
}
void conv2d(double* x, double* y, double* o, Shape4d shape_x, Shape4d shape_y) {
    int m_o = shape_x.m - shape_y.m + 1;                          
    int n_o = shape_x.n - shape_y.n + 1;
    int size_o = m_o * n_o;
    int size_x = shape_x.m * shape_x.n;
    int size_y = shape_y.m * shape_y.n;
    for(int i = 0; i < m_o; ++i) { 
        for(int j = 0; j < n_o; ++j) {
            double sum = 0.0;
            for(int k = 0; k < shape_y.c; ++k) {
                int idx_x_b, idx_y_b, idx_x, idx_y;
                idx_x_b = k * size_x + i * shape_x.m  + j;
                idx_y_b = k * size_y;
                for(int u = 0; u < shape_y.m; ++u) {
                    for(int v = 0; v < shape_y.n; ++v) {
                        idx_x = idx_x_b + u * shape_x.m + v; 
                        idx_y = idx_y_b + u * shape_y.m + v; 
                        sum += x[idx_x] * y[idx_y];
                    }
                }
            }
            o[i * m_o + j] = sum;
        }
    }
}
double** conv2d(double** x, double** y, Shape4d shape_x, Shape4d shape_y) {
    double** o = 0;
    int m_o = shape_x.m - shape_y.m + 1;
    int n_o = shape_x.n - shape_y.n + 1;
    int size_o = m_o * n_o;
    NEW_POINTERM(o, double, shape_x.num * shape_y.num, shape_x.c * m_o * n_o); 
    for(int i = 0; i < shape_x.num; ++i)
        for(int j = 0; j < shape_y.num; ++j)
            conv2d(x[i], y[j], o[i * size_o + j], shape_x, shape_y);
    return o;
}
void conv2d(double** x, double** y, double** o, Shape4d shape_x, Shape4d shape_y) {
    int m_o = shape_x.m - shape_y.m + 1;
    int n_o = shape_x.n - shape_y.n + 1;
    int size_o = m_o * n_o;
    for(int i = 0; i < shape_x.num; ++i)
        for(int j = 0; j < shape_y.num; ++j)
            conv2d(x[i], y[j], o[i], shape_x, shape_y);
}
void conv2d(double** x, double** y, double* o, Shape4d shape_x, Shape4d shape_y) {
    int m_o = shape_x.m - shape_y.m + 1;
    int n_o = shape_x.n - shape_y.n + 1;
    int size_o = m_o * n_o;
    for(int i = 0; i < shape_x.num; ++i)
        for(int j = 0; j < shape_y.num; ++j)
            conv2d(x[i], y[j], o + i * size_o, shape_x, shape_y);
}
void conv2d(double* x, double** y, double* o, Shape4d shape_x, Shape4d shape_y) {
    int size_x = size3_shape(shape_x);
    int size_y = size3_shape(shape_y);
    int m_o = shape_x.m - shape_y.m + 1;
    int n_o = shape_x.n - shape_y.n + 1;
    int size2_o = m_o * n_o;
    int size_o = size2_o * shape_x.c;
    for(int i = 0; i < shape_x.num; ++i)
        for(int j = 0; j < shape_y.num; ++j)
            conv2d(x + i * size_x, y[j], o + i * size_o + j * size2_o, shape_x, shape_y);
}
void conv2d(double* x, double** y, double** o, Shape4d shape_x, Shape4d shape_y) {
    int size_x = shape_x.m * shape_x.n * shape_x.c;
    int m_o = shape_x.m - shape_y.m + 1;
    int n_o = shape_x.n - shape_y.n + 1;
    int size_o = m_o * n_o;
    for(int i = 0; i < shape_x.num; ++i)
        for(int j = 0; j < shape_y.num; ++j)
            conv2d(x + i * size_x, y[j], o[i], shape_x, shape_y);
}
void conv2d(double* x, double* y, double** o, Shape4d shape_x, Shape4d shape_y) {
    int size_x = shape_x.m * shape_x.n * shape_x.c;
    int size_y = shape_y.m * shape_y.n * shape_y.c;
    int m_o = shape_x.m - shape_y.m + 1;
    int n_o = shape_x.n - shape_y.n + 1;
    int size_o = m_o * n_o;
    for(int i = 0; i < shape_x.num; ++i)
        for(int j = 0; j < shape_y.num; ++j)
            conv2d(x + i * size_x, y + j * size_y, o[i], shape_x, shape_y);
}
void pool_max(double* x, double* o, Shape4d shape_x, Shape4d shape_y) {
    pool_max(x, o, shape_x, shape_y, 0.0);
}
void pool_max(double* x, double* o, int* pos, Shape4d shape_x, Shape4d shape_y) {
    pool_max(x, o, pos, shape_x, shape_y, 0.0);
}
void pool_max(double* x, double* o, double* pos, Shape4d shape_x, Shape4d shape_y) {
    pool_max(x, o, pos, shape_x, shape_y, 0.0);
}
void pool_max(double* x, double* o, Shape4d shape_x, Shape4d shape_y, double b) {
    int m_o = shape_x.m / shape_y.m;                          
    int n_o = shape_x.n / shape_y.n;
    int size_o = m_o * n_o;
    for(int i = 0; i < m_o; ++i) { 
        for(int j = 0; j < n_o; ++j) {
            double max = -DBL_MIN;
            int idx = (i * shape_x.m + j) * shape_y.m;
            for(int u = 0; u < shape_y.m; ++u) {
                for(int v = 0; v < shape_y.n; ++v) {
                    int idx_x = idx + u * shape_x.m + v; 
                    if(max < x[idx_x])
                        max = x[idx_x];
                }
            }
            o[i * m_o + j] = tanh(max + b);
        }
    }
}
void pool_max(double* x, double* o, int* pos, Shape4d shape_x, Shape4d shape_y, double b) {
    int m_o = shape_x.m / shape_y.m;                          
    int n_o = shape_x.n / shape_y.n;
    int size_o = m_o * n_o;
    for(int i = 0; i < m_o; ++i) { 
        for(int j = 0; j < n_o; ++j) {
            double max = -DBL_MIN;
            int max_idx = 0;
            int idx = (i * shape_x.m + j) * shape_y.m;
            for(int u = 0; u < shape_y.m; ++u) {
                for(int v = 0; v < shape_y.n; ++v) {
                    int idx_x = idx + u * shape_x.m + v; 
                    if(max < x[idx_x]) {
                        max = x[idx_x];
                        max_idx = idx_x;
                    }
                }
            }
            o[i * m_o + j] = tanh(max + b);
            pos[i * m_o + j] = max_idx;
        }
    }
}
void pool_max(double* x, double* o, double* pos, Shape4d shape_x, Shape4d shape_y, double b) {
    int m_o = shape_x.m / shape_y.m;                          
    int n_o = shape_x.n / shape_y.n;
    int size_o = m_o * n_o;
    for(int i = 0; i < m_o; ++i) { 
        for(int j = 0; j < n_o; ++j) {
            double max = -DBL_MIN;
            int max_idx = 0;
            int idx = (i * shape_x.m + j) * shape_y.m;
            for(int u = 0; u < shape_y.m; ++u) {
                for(int v = 0; v < shape_y.n; ++v) {
                    int idx_x = idx + u * shape_x.m + v; 
                    if(max < x[idx_x]) {
                        max = x[idx_x];
                        max_idx = idx_x;
                    }
                }
            }
            o[i * m_o + j] = tanh(max + b);
            pos[i * m_o + j] = max_idx;
        }
    }
}
void pool_max(double* x, double* o, Shape4d shape_x, Shape4d shape_y, double* b) {
    int m_o = shape_x.m / shape_y.m;
    int n_o = shape_x.n / shape_y.n;
    int size_o = m_o * n_o;
    int size_x = shape_x.m * shape_x.n;
    int num_o = size2c_shape(shape_x);
    for(int i = 0; i < num_o; ++i)
        pool_max(x + i * size_x, o + i * size_o, shape_x, shape_y, *(b + i));
}
void pool_max(double* x, double* o, int* pos, Shape4d shape_x, Shape4d shape_y, double* b) {
    int m_o = shape_x.m / shape_y.m;
    int n_o = shape_x.n / shape_y.n;
    int size_o = m_o * n_o;
    int size_x = shape_x.m * shape_x.n;
    int num_o = size2c_shape(shape_x);
    for(int i = 0; i < num_o; ++i)
        pool_max(x + i * size_x, o + i * size_o, pos + i * size_o, shape_x, shape_y, *(b + i));
}
void pool_max(double* x, double* o, double* pos, Shape4d shape_x, Shape4d shape_y, double* b) {
    int m_o = shape_x.m / shape_y.m;
    int n_o = shape_x.n / shape_y.n;
    int size_o = m_o * n_o;
    int size_x = shape_x.m * shape_x.n;
    int num_o = size2c_shape(shape_x);
    for(int i = 0; i < num_o; ++i)
        pool_max(x + i * size_x, o + i * size_o, pos + i * size_o, shape_x, shape_y, *(b + i));
}
void pool_max(double* x, double* o, double** pos, Shape4d shape_x, Shape4d shape_y, double* b) {
    int m_o = shape_x.m / shape_y.m;
    int n_o = shape_x.n / shape_y.n;
    int size_o = m_o * n_o;
    int size_x = size2_shape(shape_x);
    int num_o = size2c_shape(shape_x);
    for(int i = 0; i < num_o; ++i)
        pool_max(x + i * size_x, o + i * size_o, pos[i], shape_x, shape_y, *(b + i));
}
void pool_max_inverse(double* x, double* o, Shape4d shape_x, Shape4d shape_y) {
    int m_o = shape_x.m * shape_y.m;                          
    int n_o = shape_x.n * shape_y.n;
    int size_o = m_o * n_o;
    set_value(0.0, o, size_o);
    for(int i = 0; i < shape_x.m; ++i)
        for(int j = 0; j < shape_x.n; ++j)
            o[(shape_y.m * i) * m_o + j * shape_y.n] = x[i * shape_x.m + j];
}
void pool_max_inverse(double* x, int* pos, double* o, Shape4d shape_x, Shape4d shape_y) {
    int m_o = shape_x.m * shape_y.m;                          
    int n_o = shape_x.n * shape_y.n;
    int size_o = m_o * n_o;
    set_value(0.0, o, size_o);
    for(int i = 0; i < shape_x.m; ++i)
        for(int j = 0; j < shape_x.n; ++j)
            o[pos[i * shape_x.m + j]] = x[i * shape_x.m + j];
}
void pool_max_inverse(double* x, double* pos, double* o, Shape4d shape_x, Shape4d shape_y) {
    int m_o = shape_x.m * shape_y.m;                          
    int n_o = shape_x.n * shape_y.n;
    int size_o = m_o * n_o;
    set_value(0.0, o, size_o);
    for(int i = 0; i < shape_x.m; ++i)
        for(int j = 0; j < shape_x.n; ++j)
            o[(int)pos[i * shape_x.m + j]] = x[i * shape_x.m + j];
}
void pool_max_inverse(double* x, double** pos, double* o, Shape4d shape_x, Shape4d shape_y) {
    int m_o = shape_x.m * shape_y.m;                          
    int n_o = shape_x.n * shape_y.n;
    int size_x = size2_shape(shape_x);
    int num_o = size2c_shape(shape_x);
    for(int i = 0; i < num_o; ++i)
        pool_max_inverse(x + i * size_x, pos[i], shape_x, shape_y);
}
Shape4d padding(double* x, double* o, Shape4d shape_x, Shape4d shape_y) {
    int m_o = shape_x.m + shape_y.m - 1;
    int n_o = shape_x.n + shape_y.n - 1;
    int size_o = m_o * n_o;
    set_value(0.0, o, size_o);
    int idx_m = (shape_y.m - 1) / 2;
    int idx_n = (shape_y.n - 1) / 2;
    for(int i = 0; i < shape_x.m; ++i)
        for(int j = 0; j < shape_x.n; ++j)
            o[(idx_n + i) * m_o + idx_m + j] = x[i * shape_x.m + j];
    return shape_2d(m_o, n_o, 1, 1);
}
void rot180(double* x, Shape4d shape) {
    for(int i = 0; i < shape.m / 2; ++i)
        for(int j = 0; j < shape.n; ++j)
            swap(x + i * shape.m + j, x + (shape.m - 1 - i) * shape.m + j);
    for(int i = 0; i < shape.m; ++i)
        for(int j = 0; j < shape.n / 2; ++j)
            swap(x + i * shape.m + j, x + i * shape.m + shape.n - 1 - j);
}
void rot180(double* x, double*y, Shape4d shape) {
    int size = size_shape(shape);
    copy(x, y, size);
    rot180(y, shape);
}
void rot180(double** x, double** y, Shape4d shape) {
    int size_2d = size2_shape(shape);
    for(int i = 0; i < shape.c; ++i) {
        int idx_x = i * size_2d;
        for(int j = 0; j < shape.num; ++j) {
            int idx_y = j * size_2d;
            copy(x[j] + idx_x, y[i] + idx_y, size_2d);
            rot180(y[i] + idx_y, shape);
        }
    }
}
Shape4d reshape(Shape4d shape) {
    return shape_2d(shape.m, shape.n, shape.c, shape.num);
}
void reshape(double** x, double** y, Shape4d shape) {
    int size_2d = size2_shape(shape);
    for(int i = 0; i < shape.c; ++i) {
        int idx_x = i * size_2d;
        for(int j = 0; j < shape.num; ++j) {
            int idx_y = j * size_2d;
            copy(x[j] + idx_x, y[i] + idx_y, size_2d);
        }
    }
}
void reshape(double* x, double*y, Shape4d shape) {
    int size_2d = size2_shape(shape);
    int size_3d = size3_shape(shape);
    int size_3d_c = shape.num * shape.m * shape.n;
    for(int i = 0; i < shape.c; ++i) {
        int idx_x = i * size_2d;
        int idx_c = i * size_3d_c;
        for(int j = 0; j < shape.num; ++j) {
            int idx_y = j * size_2d;
            copy(x + j * size_3d + idx_x, y + idx_c + idx_y, size_2d);
        }
    }
}
void reshape_a(double** x, double** y, Shape4d shape) {
    int size_2d = size2_shape(shape);
    for(int i = 0; i < shape.c; ++i) {
        int idx_x = i * size_2d;
        for(int j = 0; j < shape.num; ++j) {
            int idx_y = j * size_2d;
            add(y[i] + idx_y, x[j] + idx_x, size_2d);
        }
    }
}
Shape4d shape_2d(int m, int n, int num, int c) {
    Shape4d shape_img;
    shape_img.m = m;
    shape_img.n = n;
    shape_img.num = num;
    shape_img.c = c;
    return shape_img;
}
void shape_print(Shape4d shape) {
    cout << shape.num << " " << shape.c << " " << shape.m << " " << shape.n << endl;
}
void shape_print(Shape4d* shape) {
    cout << shape->num << " " << shape->c << " " << shape->m << " " << shape->n << endl;
}
int size2_shape(Shape4d shape) {
    return shape.m * shape.n;
}
int size2c_shape(Shape4d shape) {
    return shape.num * shape.c;
}
int size3_shape(Shape4d shape) {
    return shape.m * shape.n * shape.c;
}
int size3c_shape(Shape4d shape) {
    return shape.m * shape.n * shape.num;
}
int size_shape(Shape4d shape) {
    return shape.m * shape.n * shape.c * shape.num;
}
void shape_copy(Shape4d& shape_x, Shape4d& shape_y) {
    shape_y.m = shape_x.m;
    shape_y.n = shape_x.n;
    shape_y.c = shape_x.c;
    shape_y.num = shape_x.num;
}
void load_digits(double* input_data, int label, int size_row, int size_col, int num) {
    char file[50];
    for(int i = 0; i < num; ++i) {
        sprintf(file, "digits_txt/%d_%d.txt", label, i);
        load_matrix(file, input_data + i * size_row * size_col, size_row, size_col);
    }
}
