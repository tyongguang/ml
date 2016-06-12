#include <ctime>
#include <cmath>
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
        printf("File could not be opened ");
        return;
    }
    int elem_size = sizeof(double);
    if (fwrite((const void*)vec, elem_size, size, pf) != size) {
        fputs ("Writing ppw error",stderr);
        return;
    }
    fclose(pf);
}
void dump_matrix(const char* file_name, double** matrix, int row_size, int col_size) {
    CHECK_PTR_BASE(matrix, return);
    CHECK_PTR_BASE(file_name, return);
    FILE* pf;
    if((pf = fopen(file_name, "wb")) == NULL) { 
        printf("File could not be opened "); 
        return;
    } 
    int elem_size = sizeof(double);
    for (int i = 0; i < row_size; ++i) {
        if (fwrite((const void*)matrix[i], elem_size, col_size, pf) != col_size) {
            fputs ("Writing ppw error", stderr);
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
        printf("File could not be opened ");
        return -1;
    }
    //让文件指针偏移到正确位置
    fseek(pf, offset, SEEK_SET);
    int elem_size = sizeof(double);
    read_size = fread(vec, elem_size, size, pf);
    if (read_size != size) {
        fputs ("Reading pb error",stderr);
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
        printf("File could not be opened "); 
        return -1;
    } 
    //让文件指针偏移到正确位置
    fseek(pf, offset, SEEK_SET);
    int elem_size = sizeof(double);
    for (int i = 0; i < row_size; ++i) {
        read_size = fread((void*)matrix[i], elem_size, col_size, pf);
        if (read_size != col_size) {
            fputs ("Reading ppw error",stderr);
            return -1;
        }
    }
    fclose(pf);
    return row_size * col_size * elem_size;  
}
