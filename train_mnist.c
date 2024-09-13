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
    const float *bias,    // (K_C, out_H, out_W)
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
                    acc[j][i] = (bias != NULL) ? bias[(k_c * out_H * out_W) + (j * out_W) + i] : 0.0f;
                }
            }

            for (int c = 0; c < C; c++)
            {
                int bk_cc = (b * C * out_H * out_W) + (c * out_H * out_W);
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

// ----------------------------------------------------------------------------
// Mnist model definition

// end model

int main()
{
    // int X_train_len;
    // float *X_train = tensor_from_disk("./downloads/X_train.gunzip", X_OFFSET, &X_train_len);
    // int Y_train_len;
    // float *Y_train = tensor_from_disk("./downloads/Y_train.gunzip", Y_OFFSET, &Y_train_len);
    // int X_test_len;
    // float *X_test = tensor_from_disk("./downloads/X_test.gunzip", X_OFFSET, &X_test_len);
    // int Y_test_len;
    // float *Y_test = tensor_from_disk("./downloads/Y_test.gunzip", Y_OFFSET, &Y_test_len);

    // int train_len = X_train_len / (IMAGE_SIZE * IMAGE_SIZE);
    // assert(train_len == Y_train_len); // we should have as many images as labels
    // int test_len = X_test_len / (IMAGE_SIZE * IMAGE_SIZE);
    // assert(test_len == Y_test_len);

    // printf("train set size: %d | test set size: %d\n", train_len, test_len);

#define b 2
#define c 1
#define h 3
#define w 3
    float arr[b * c * h * w] = {1, 6, 2, 5, 3, 1, 7, 0, 4,
                                1, 6, 2, 5, 3, 1, 7, 0, 4};

#define k_c 1
#define k_h 2
#define k_w 2
    float kernels[k_c * c * k_h * k_w] = {1, 2, -1, 0};

#define out_h (h - k_h + 1)
#define out_w (w - k_w + 1)

    float out[b * k_c * out_h * out_w];
    conv2d_forward(out, arr, kernels, NULL, b, c, h, w, k_c, k_h, k_w);

    for (int z = 0; z < b; z++)
    {
        for (int j = 0; j < out_h; j++)
        {
            for (int i = 0; i < out_w; i++)
            {
                printf("%f ", out[z * out_h * out_w + j * out_w + i]);
            }
            printf("\n");
        }
        printf("--------\n");
    }

    return 0;
}