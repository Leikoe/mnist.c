#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define X_OFFSET 0x10
#define Y_OFFSET 8
#define IMAGE_SIZE 28

// returns an allocated array which must be freed
void *tensor_from_disk(const char *path, const size_t offset, const size_t item_size, size_t *len)
{
    FILE *f = fopen(path, "rb"); // open file at path
    fseek(f, 0L, SEEK_END);      // get length
    int f_size = ftell(f);
    rewind(f);                // go back to the beginning
    *len = f_size - offset;   // set read array length
    char *arr = malloc(*len); // get some memory to store read bytes
    fread(arr, 1, *len, f);   // copy "length" bytes from file into the array
    fclose(f);                // close the file
    *len /= item_size;
    return arr;
}

// from llm.c
void matmul_forward_naive(float *out,
                          const float *inp, const float *weight, const float *bias,
                          int B, int T, int C, int OC)
{
// the most naive implementation of matrix multiplication
// this serves as an algorithmic reference, and as a fallback for
// unfriendly input shapes inside matmul_forward(), below.
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            int bt = b * T + t;
            for (int o = 0; o < OC; o++)
            {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++)
                {
                    val += inp[bt * C + i] * weight[o * C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

void conv2d_forward(
    // out_H = H - K_H + 1
    // out_W = W - K_W + 1
    float *out,           // (B, K_C, out_H, out_W)
    const float *in,      // (B, C, H, W)
    const float *kernels, // (K_C, C, K_H, K_W)
    const float *bias,    // (K_C)
    const int B, const int C, const int H, const int W,
    const int K_C, const int K_H, const int K_W)
{
    int out_H = H - K_H + 1;
    int out_W = W - K_W + 1;

    for (int b = 0; b < B; b++)
    {
        for (int k_c = 0; k_c < K_C; k_c++)
        {
            float acc[out_H][out_W];
            for (int j = 0; j < out_H; j++)
            {
                for (int i = 0; i < out_W; i++)
                {
                    acc[j][i] = (bias != NULL) ? bias[k_c] : 0.0f;
                }
            }

            for (int c = 0; c < C; c++)
            {
                int bk_cc = (b * C * H * W) + (c * H * W);
                // slide kernel
                for (int j = 0; j < out_H; j++)
                {
                    for (int i = 0; i < out_W; i++)
                    {
                        float inner_acc = 0.0;
                        // inner correlation
                        for (int k_j = 0; k_j < K_H; k_j++)
                        {
                            for (int k_i = 0; k_i < K_W; k_i++)
                            {
                                float a = in[bk_cc + ((j + k_j) * W) + (i + k_i)];
                                float b = kernels[(k_c * C * K_H * K_W) + (c * K_H * K_W) + (k_j * K_W) + k_i];
                                inner_acc += a * b;
                                // printf("%f * %f k_j: %d k_i: %d j: %d i: %d makes (%d, %d) idx: %d\n", a, b, k_j, k_i, j, i, j + k_j, i + k_i, ((j + k_j) * out_W) + (i + k_i));
                            }
                        }
                        acc[j][i] += inner_acc;
                        // printf("storing %f at (%d, %d)\n", inner_acc, j, i);
                    }
                }
            }

            // store the output
            for (int j = 0; j < out_H; j++)
            {
                for (int i = 0; i < out_W; i++)
                {
                    out[(b * K_C * out_H * out_W) + (k_c * out_H * out_W) + (j * out_W) + i] = acc[j][i];
                }
            }
        }
    }
}

void relu_forward(float *out, const float *in, const size_t N)
{
    for (int i = 0; i < N; i++)
    {
        float tmp = in[i];
        out[i] = (tmp > 0) ? tmp : 0;
    }
}

// ----------------------------------------------------------------------------
// Mnist model definition

// end model

void printn(const float *in, const size_t N)
{
    printf("[");
    for (int i = 0; i < N; i++)
    {
        printf("%f", in[i]);
        if (i != N - 1)
        {
            printf(", ");
        }
    }
    printf("]\n");
}

int main()
{
    size_t X_train_len;
    unsigned char *X_train = tensor_from_disk("./downloads/X_train.gunzip", X_OFFSET, sizeof(unsigned char), &X_train_len);
    size_t Y_train_len;
    unsigned char *Y_train = tensor_from_disk("./downloads/Y_train.gunzip", Y_OFFSET, sizeof(unsigned char), &Y_train_len);
    size_t X_test_len;
    unsigned char *X_test = tensor_from_disk("./downloads/X_test.gunzip", X_OFFSET, sizeof(unsigned char), &X_test_len);
    size_t Y_test_len;
    unsigned char *Y_test = tensor_from_disk("./downloads/Y_test.gunzip", Y_OFFSET, sizeof(unsigned char), &Y_test_len);

    int train_len = X_train_len / (IMAGE_SIZE * IMAGE_SIZE);
    assert(train_len == Y_train_len); // we should have as many images as labels
    int test_len = X_test_len / (IMAGE_SIZE * IMAGE_SIZE);
    assert(test_len == Y_test_len);

    printf("train set size: %d | test set size: %d\n", train_len, test_len);

    size_t params_len;
    float *params = tensor_from_disk("./tensor.bin", 0, sizeof(float), &params_len);
    printf("%zu\n", params_len);

    float *weights = params;
    float *bias = params + 32 * 1 * 5 * 5;

    float img[IMAGE_SIZE * IMAGE_SIZE];
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++)
    {
        img[i] = (float)X_train[i];
    }

    int out_size = IMAGE_SIZE - 5 + 1;
    float out[1 * 32 * out_size * out_size];
    conv2d_forward(out, img, params, bias, 1, 1, IMAGE_SIZE, IMAGE_SIZE, 32, 5, 5);
    printn(out, 10);

    float out_relu[1 * 32 * out_size * out_size];
    relu_forward(out_relu, out, 1 * 32 * out_size * out_size);
    printn(out_relu, 10);

    // for (int j = 0; j < out_size; j++)
    // {
    //     for (int i = 0; i < out_size; i++)
    //     {
    //         printf("%f ", out[(j * out_size) + i]);
    //     }
    //     printf("\n");
    // }

    return 0;
}
