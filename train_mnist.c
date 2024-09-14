#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define X_OFFSET 0x10
#define Y_OFFSET 8
#define IMAGE_SIZE 28

double NAN = 0.0 / 0.0;
double POS_INF = 1.0 / 0.0;
double NEG_INF = -1.0 / 0.0;

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

void maxpool2d_forward(
    // we are using stride = kernel size here. e.g: (2, 2) kernel => (2, 2) stride
    // out_H = H / K_H
    // out_W = W / K_W
    float *out,      // (B, C, out_H, out_W)
    const float *in, // (B, C, H, W)
    const int B, const int C, const int H, const int W,
    const int K_W, const int K_H)
{
    int out_H = H - K_H + 1;
    int out_W = W - K_W + 1;
    for (int b = 0; b < B; b++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int j = 0; j < out_H; j++)
            {
                for (int i = 0; i < out_W; i++)
                {
                    float max = in[b * C * H * W + c * H * W + (j * K_H * W) + (i * K_W)]; // init to first or NEG_INF ?
                    for (int k_j = 0; k_j < K_H; k_j++)
                    {
                        for (int k_i = 0; k_i < K_H; k_i++)
                        {
                            float v = in[b * C * H * W + c * H * W + ((j * K_H + k_j) * W) + (i * K_W + k_i)];
                            if (v > max)
                            {
                                max = v;
                            }
                        }
                    }
                    out[b * C * out_H * out_W + c * out_H * out_W + j * out_W + i] = max;
                }
            }
        }
    }
}

// out = x @ weight.T + bias
void linear_forward(
    float *out,          // (B, 1, out_features)
    const float *x,      // (B, 1, in_features)
    const float *weight, // (out_features, in_features)
    const float *bias,   // (out_features)
    const int B, const int in_features, const int out_features)
{
    for (int b = 0; b < B; b++)
    {
        for (int j = 0; j < 1; j++)
        {
            for (int i = 0; i < out_features; i++)
            {
                float acc = (bias != NULL) ? bias[i] : 0.0;
                for (int k = 0; k < in_features; k++)
                {
                    acc += x[(b * 1 * in_features) + (j * in_features) + i] * weight[i * in_features + k];
                }
                out[b * out_features + i] = acc;
            }
        }
    }
}

void argmax_forward(
    int *out,        // (B,)
    const float *in, // (B, N)
    const int B, const int N)
{
    for (int b = 0; b < B; b++)
    {
        int argmax = 0;
        float max = in[b * N];
        for (int i = 0; i < N; i++)
        {
            float tmp = in[b * N + i];
            if (tmp > max)
            {
                argmax = i;
                max = tmp;
            }
        }
        out[b] = argmax;
    }
}

// ----------------------------------------------------------------------------
// Mnist model definition

// C = input channels
// OC = output channels
// KS = kernel size
// OS = output size

#define CONV2D_1_C 1 // conv1
#define CONV2D_1_OC 32
#define CONV2D_1_KS 5
#define CONV2D_1_OS (IMAGE_SIZE - CONV2D_1_KS + 1)

#define CONV2D_2_C CONV2D_1_OC // conv2
#define CONV2D_2_OC 32
#define CONV2D_2_KS 5
#define CONV2D_2_OS (CONV2D_1_OS - CONV2D_2_KS + 1)

#define MAXPOOL2D_1_KS 2 // maxpool1
#define MAXPOOL2D_1_OS (CONV2D_2_OS - MAXPOOL2D_1_KS + 1)

#define CONV2D_3_C CONV2D_2_OC // conv3
#define CONV2D_3_OC 64
#define CONV2D_3_KS 3
#define CONV2D_3_OS (MAXPOOL2D_1_OS - CONV2D_3_KS + 1)

#define CONV2D_4_C CONV2D_3_OC // conv4
#define CONV2D_4_OC 64
#define CONV2D_4_KS 3
#define CONV2D_4_OS (CONV2D_3_OS - CONV2D_4_KS + 1)

#define MAXPOOL2D_2_KS 2 // maxpool1
#define MAXPOOL2D_2_OS (CONV2D_4_OS - MAXPOOL2D_2_KS + 1)

#define LINEAR_1_IF 576 // linear
#define LINEAR_1_OF 10

// the parameters of the model
#define NUM_PARAMETER_TENSORS 10
struct ParameterTensors
{
    float *conv1w;   // (CONV2D_1_OC, CONV2D_1_C, CONV2D_1_KS, CONV2D_1_KS)
    float *conv1b;   // (CONV2D_1_OC)
    float *conv2w;   // (CONV2D_2_OC, CONV2D_2_C, CONV2D_2_KS, CONV2D_2_KS)
    float *conv2b;   // (CONV2D_2_OC)
    float *conv3w;   // (CONV2D_3_OC, CONV2D_3_C, CONV2D_3_KS, CONV2D_3_KS)
    float *conv3b;   // (CONV2D_3_OC)
    float *conv4w;   // (CONV2D_4_OC, CONV2D_4_C, CONV2D_4_KS, CONV2D_4_KS)
    float *conv4b;   // (CONV2D_4_OC)
    float *linear1w; // (LINEAR_1_OF, LINEAR_1_IF)
    float *linear1b; // (LINEAR_1_OF)
};

void fill_in_parameter_sizes(size_t *param_sizes)
{
    param_sizes[0] = CONV2D_1_OC * CONV2D_1_C * CONV2D_1_KS * CONV2D_1_KS; // conv1w
    param_sizes[1] = CONV2D_1_OC;                                          // conv1b
    param_sizes[2] = CONV2D_2_OC * CONV2D_2_C * CONV2D_2_KS * CONV2D_2_KS; // conv2w
    param_sizes[3] = CONV2D_2_OC;                                          // conv2b
    param_sizes[4] = CONV2D_3_OC * CONV2D_3_C * CONV2D_3_KS * CONV2D_3_KS; // conv3w
    param_sizes[5] = CONV2D_3_OC;                                          // conv3b
    param_sizes[6] = CONV2D_4_OC * CONV2D_4_C * CONV2D_4_KS * CONV2D_4_KS; // conv4w
    param_sizes[7] = CONV2D_4_OC;                                          // conv4b
    param_sizes[8] = LINEAR_1_OF * LINEAR_1_IF;                            // linear1w
    param_sizes[9] = LINEAR_1_OF;                                          // linear1b
}

// allocate memory for the parameters and point the individual tensors to the right places
float *malloc_and_point_parameters(struct ParameterTensors *params, size_t *param_sizes)
{
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    float *params_memory = (float *)malloc(num_parameters * sizeof(float));
    // assign all the tensors
    float **ptrs[] = {
        &params->conv1w, &params->conv1b,
        &params->conv2w, &params->conv2b,
        &params->conv3w, &params->conv3b,
        &params->conv4w, &params->conv4b,
        &params->linear1w, &params->linear1b};
    float *params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 11
struct ActivationTensors
{
    float *conv2d_1;      // (B, CONV2D_1_OC, CONV2D_1_OS, CONV2D_1_OS)
    float *conv2d_1_relu; // (B, CONV2D_1_OC, CONV2D_1_OS, CONV2D_1_OS)
    float *conv2d_2;      // (B, CONV2D_2_OC, CONV2D_2_OS, CONV2D_2_OS)
    float *conv2d_2_relu; // (B, CONV2D_2_OC, CONV2D_2_OS, CONV2D_2_OS)
    float *maxpool2d_1;   // (B, CONV2D_2_OC, MAXPOOL2D_1_OS, MAXPOOL2D_1_OS)
    float *conv2d_3;      // (B, CONV2D_3_OC, CONV2D_3_OS, CONV2D_3_OS1)
    float *conv2d_3_relu; // (B, CONV2D_3_OC, CONV2D_3_OS, CONV2D_3_OS1)
    float *conv2d_4;      // (B, CONV2D_4_OC, CONV2D_4_OS, CONV2D_4_OS)
    float *conv2d_4_relu; // (B, CONV2D_4_OC, CONV2D_4_OS, CONV2D_4_OS)
    float *maxpool2d_2;   // (B, CONV2D_4_OC, MAXPOOL2D_2_OS, MAXPOOL2D_2_OS)
    float *linear_1;      // (B, LINEAR_1_OF)
    float *argmax;        // (B,)
};

void fill_in_activation_sizes(size_t *act_sizes, int B)
{
    act_sizes[0] = B * CONV2D_1_OC * CONV2D_1_OS * CONV2D_1_OS;       // conv1
    act_sizes[1] = B * CONV2D_1_OC * CONV2D_1_OS * CONV2D_1_OS;       // conv1 relu
    act_sizes[2] = B * CONV2D_2_OC * CONV2D_2_OS * CONV2D_2_OS;       // conv2
    act_sizes[3] = B * CONV2D_2_OC * CONV2D_2_OS * CONV2D_2_OS;       // conv2 relu
    act_sizes[4] = B * CONV2D_2_OC * MAXPOOL2D_1_OS * MAXPOOL2D_1_OS; // maxpool1
    act_sizes[5] = B * CONV2D_3_OC * CONV2D_3_OS * CONV2D_3_OS;       // conv3
    act_sizes[6] = B * CONV2D_3_OC * CONV2D_3_OS * CONV2D_3_OS;       // conv3 relu
    act_sizes[7] = B * CONV2D_4_OC * CONV2D_4_OS * CONV2D_4_OS;       // conv4
    act_sizes[8] = B * CONV2D_4_OC * CONV2D_4_OS * CONV2D_4_OS;       // conv4 relu
    act_sizes[9] = B * CONV2D_4_OC * MAXPOOL2D_2_OS * MAXPOOL2D_2_OS; // maxpool2
    act_sizes[10] = B * LINEAR_1_OF;                                  // linear
}

float *malloc_and_point_activations(struct ActivationTensors *acts, size_t *act_sizes)
{
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++)
    {
        num_activations += act_sizes[i];
    }
    float *acts_memory = (float *)malloc(num_activations * sizeof(float));
    float **ptrs[] = {
        &acts->conv2d_1,
        &acts->conv2d_1_relu,
        &acts->conv2d_2,
        &acts->conv2d_2_relu,
        &acts->maxpool2d_1,
        &acts->conv2d_3,
        &acts->conv2d_3_relu,
        &acts->conv2d_4,
        &acts->conv2d_4_relu,
        &acts->maxpool2d_2,
        &acts->linear_1};
    float *acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++)
    {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

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

    int batch_size = 1;

    // params
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    fill_in_parameter_sizes(param_sizes);
    struct ParameterTensors params;
    float *params_handle = malloc_and_point_parameters(&params, param_sizes);

    // load weights
    FILE *f = fopen("./params.bin", "rb");
    fseek(f, 0L, SEEK_END);
    int f_size = ftell(f);
    rewind(f);
    fread(params_handle, 1, f_size, f);
    fclose(f);
    int params_len = f_size / sizeof(float);

    // activations
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    fill_in_activation_sizes(act_sizes, batch_size);
    struct ActivationTensors activations;
    float *activations_handle = malloc_and_point_activations(&activations, act_sizes);

    // network inference
    float inputs[batch_size * IMAGE_SIZE * IMAGE_SIZE];
    for (int i = 0; i < batch_size * IMAGE_SIZE * IMAGE_SIZE; i++)
    {
        // inputs[i] = (float)X_train[i];
        inputs[i] = i;
    }
    printn(inputs, 10);

    // forward pass
    conv2d_forward(activations.conv2d_1, inputs, params.conv1w, params.conv1b, batch_size, CONV2D_1_C, IMAGE_SIZE, IMAGE_SIZE, CONV2D_1_OC, CONV2D_1_KS, CONV2D_1_KS);
    printn(activations.conv2d_1, 10);
    relu_forward(activations.conv2d_1_relu, activations.conv2d_1, batch_size * CONV2D_1_OC * CONV2D_1_OS * CONV2D_1_OS);
    printn(activations.conv2d_1_relu, 10);

    conv2d_forward(activations.conv2d_2, activations.conv2d_1_relu, params.conv2w, params.conv2b, batch_size, CONV2D_2_C, CONV2D_1_OS, CONV2D_1_OS, CONV2D_2_OC, CONV2D_2_KS, CONV2D_2_KS);
    printn(activations.conv2d_2, 10);
    relu_forward(activations.conv2d_2_relu, activations.conv2d_2, batch_size * CONV2D_2_OC * CONV2D_2_OS * CONV2D_2_OS);
    printn(activations.conv2d_2_relu, 10);

    maxpool2d_forward(activations.maxpool2d_1, activations.conv2d_2_relu, batch_size, CONV2D_2_OC, CONV2D_2_OS, CONV2D_2_OS, MAXPOOL2D_1_KS, MAXPOOL2D_1_KS);
    printn(activations.maxpool2d_1, 10);

    conv2d_forward(activations.conv2d_3, activations.maxpool2d_1, params.conv3w, params.conv3b, batch_size, CONV2D_3_C, MAXPOOL2D_1_OS, MAXPOOL2D_1_OS, CONV2D_3_OC, CONV2D_3_KS, CONV2D_3_KS);
    printn(activations.conv2d_3, 10);
    relu_forward(activations.conv2d_3_relu, activations.conv2d_3, batch_size * CONV2D_3_OC * CONV2D_3_OS * CONV2D_3_OS);
    printn(activations.conv2d_3_relu, 10);

    conv2d_forward(activations.conv2d_4, activations.conv2d_3_relu, params.conv4w, params.conv4b, batch_size, CONV2D_4_C, CONV2D_3_OS, CONV2D_3_OS, CONV2D_4_OC, CONV2D_4_KS, CONV2D_4_KS);
    printn(activations.conv2d_4, 10);
    relu_forward(activations.conv2d_4_relu, activations.conv2d_4, batch_size * CONV2D_4_OC * CONV2D_4_OS * CONV2D_4_OS);
    printn(activations.conv2d_4_relu, 10);

    maxpool2d_forward(activations.maxpool2d_2, activations.conv2d_4_relu, batch_size, CONV2D_4_OC, CONV2D_4_OS, CONV2D_4_OS, MAXPOOL2D_2_KS, MAXPOOL2D_2_KS);
    printn(activations.maxpool2d_2, 10);
    linear_forward(activations.linear_1, activations.maxpool2d_2, params.linear1w, params.linear1b, batch_size, LINEAR_1_IF, LINEAR_1_OF);
    printn(activations.linear_1, LINEAR_1_OF);

    int argmax[batch_size * LINEAR_1_OF];
    argmax_forward(argmax, activations.linear_1, batch_size, LINEAR_1_OF);
    printf("y_pred = %d | y = %d\n", argmax[0], Y_train[0]);

    free(activations_handle);
    free(params_handle);
    return 0;
}
