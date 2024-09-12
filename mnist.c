#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define X_OFFSET 0x10
#define Y_OFFSET 8
#define IMAGE_SIZE 28

// returns an allocated array which must be freed
float *tensor_from_disk(const char *path, const int offset, int *len)
{
    FILE *f = fopen(path, "rb"); // open file at path
    fseek(f, 0L, SEEK_END);      // get length
    int f_size = ftell(f);
    *len = f_size - offset;                      // set read array length
    float *arr = malloc(sizeof(float) * (*len)); // get some memory to store read bytes
    fread(arr, 1, *len, f);                      // copy "length" bytes from file into the array
    fclose(f);                                   // close the file
    return arr;
}

int main()
{
    int X_train_len;
    float *X_train = tensor_from_disk("./downloads/X_train.gunzip", X_OFFSET, &X_train_len);
    int Y_train_len;
    float *Y_train = tensor_from_disk("./downloads/Y_train.gunzip", Y_OFFSET, &Y_train_len);
    int X_test_len;
    float *X_test = tensor_from_disk("./downloads/X_test.gunzip", X_OFFSET, &X_test_len);
    int Y_test_len;
    float *Y_test = tensor_from_disk("./downloads/Y_test.gunzip", Y_OFFSET, &Y_test_len);

    int train_len = X_train_len / (IMAGE_SIZE * IMAGE_SIZE);
    assert(train_len == Y_train_len); // we should have as many images as labels
    int test_len = X_test_len / (IMAGE_SIZE * IMAGE_SIZE);
    assert(test_len == Y_test_len);

    printf("train set size: %d | test set size: %d\n", train_len, test_len);
    return 0;
}