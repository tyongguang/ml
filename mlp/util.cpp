#include <ctime>
#include <cmath>
#include <string>
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include "util.h"

using namespace std;

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
double** conv2d(double** x, double** y, shape_2d shape_x, shape_2d shape_y) {
    double** c = 0;
    NEW_POINTERM(c, double, shape_x.m, shape_x.n); 
    for(int i = 0; i < shape_x.m; ++i) {
        for(int j = 0; j < shape_x.n; ++j) {
            for(int k = 0; k < shape_y.m; ++k) {
                for(int l = 0; l < shape_y.n; ++l) {

                }
            }
        }
    }
    return c;
}
double* conv2d(double* x, double* y, shape_2d shape_x, shape_2d shape_y) {
    double* c = 0;
    NEW_POINTERV(c, double, shape_x.m * shape_x.n);  
}
