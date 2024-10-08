#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#ifdef OMP
#include <omp.h>
#endif

#define MAX_STEPS 70
#define EVAL_STEPS 10
#define BATCH_SIZE 8

#define X_OFFSET 0x10
#define Y_OFFSET 8
#define IMAGE_SIZE 28

// UTILS

// returns an allocated array which must be freed
void *tensor_from_disk(const char *path, const size_t offset, const size_t item_size, size_t *len)
{
    FILE *f = fopen(path, "rb"); // open file at path
    fseek(f, 0L, SEEK_END);      // get length
    int f_size = ftell(f);
    rewind(f);                // go back to the beginning
    *len = f_size - offset;   // set read array length
    char *arr = malloc(*len); // get some memory to store read bytes
    fseek(f, offset, SEEK_SET);   // seek to offset
    fread(arr, 1, *len, f);   // copy "length" bytes from file into the array
    fclose(f);                // close the file
    *len /= item_size;
    return arr;
}

float random_float()
{
    return (float)rand() / (float)RAND_MAX;
}

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

// modified from https://stackoverflow.com/questions/11641629/generating-a-uniform-distribution-of-integers-in-c
// guess I'll find out if it's a good enough uniform distribution we're sampling from
// update: I now know it's not (compared to pytorch initialized untrained model)
float uniform_distribution(float rangeLow, float rangeHigh) {
    double myRand = rand()/(1.0 + RAND_MAX);
    float range = rangeHigh - rangeLow + 1.0;
    float myRand_scaled = (myRand * range) + rangeLow;
    return myRand_scaled;
}

void fill_uniform(float *out, const float low, const float high, const int N) {
    for (int i = 0; i < N; i++) {
        out[i] = uniform_distribution(low, high);
    }
}

// DATALOADER
struct DataLoader {
    int batch_size;
    float *inputs;
    int *targets;
    size_t len;
    unsigned char *imgs;
    unsigned char *labels;
};

void dataloader_init(struct DataLoader *dl, unsigned char *imgs, unsigned char *labels, const size_t len, const int B) {
    dl->batch_size = B;
    dl->inputs = (float *)malloc(B * IMAGE_SIZE * IMAGE_SIZE * sizeof(float));
    dl->targets = (int *)malloc(B * sizeof(int));
    dl->imgs = imgs;
    dl->labels = labels;
    dl->len = len;
}

void dataloader_next_batch(struct DataLoader *self) {
    for (int i = 0; i < self->batch_size; i++) {
        int idx = random_float() * self->len;
        for (int j = 0; j < IMAGE_SIZE * IMAGE_SIZE; j++) {
            self->inputs[i * IMAGE_SIZE * IMAGE_SIZE + j] = (float)self->imgs[idx * IMAGE_SIZE * IMAGE_SIZE + j];
        }
        self->targets[i] = self->labels[idx];
    }
}

void dataloader_free(struct DataLoader *self) {
    free(self->inputs);
    free(self->targets);
}


// OPS

void _conv2d_forward(
    // out_H = H - K_H + 1
    // out_W = W - K_W + 1
    float *out,           // (B, K_C, out_H, out_W)
    const float *in,      // (B, C, H, W)
    const float *kernels, // (K_C, C, K_H, K_W)
    const float *bias,    // (K_C)
    const int B, const int C, const int H, const int W,
    const int K_C, const int K_H, const int K_W,
    const bool flip_kernels    // should be false by default
)
{
    int out_H = H - K_H + 1;
    int out_W = W - K_W + 1;

    #pragma omp parallel for
    for (int b = 0; b < B; b++)
    {
        for (int k_c = 0; k_c < K_C; k_c++)
        {
            // correlation
            for (int j = 0; j < out_H; j++)
            {
                for (int i = 0; i < out_W; i++)
                {
                    float channeled_correlation_sum = 0.0;
                    for (int c = 0; c < C; c++)
                    {
                        // element wise multiplication
                        for (int k_j = 0; k_j < K_H; k_j++)
                        {
                            for (int k_i = 0; k_i < K_W; k_i++)
                            {
                                float a = in[(b * C * H * W) + (c * H * W) + ((j + k_j) * W) + (i + k_i)];
                                int k_j_maybe_t = (flip_kernels) ? (K_H - k_j - 1) : k_j;
                                int k_i_maybe_t = (flip_kernels) ? (K_W - k_i - 1) : k_i;
                                float b = kernels[(k_c * C * K_H * K_W) + (c * K_H * K_W) + (k_j_maybe_t * K_W) + k_i_maybe_t];
                                channeled_correlation_sum += a * b;
                            }
                        }
                    }
                    out[(b * K_C * out_H * out_W) + (k_c * out_H * out_W) + (j * out_W) + i] = channeled_correlation_sum;
                }
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
    const int K_C, const int K_H, const int K_W
) {
    _conv2d_forward(out, in, kernels, bias, B, C, H, W, K_C, K_H, K_W, false);
}

void conv2d_backward(
    // out_H = H - K_H + 1
    // out_W = W - K_W + 1
    float *din,           // (B, C, H, W)
    float *dkernels,      // (K_C, C, K_H, K_W)
    float *dbias,         // (K_C)
    const float *dout,    // (B, K_C, out_H, out_W)
    const float *in,      // (B, C, H, W)
    const float *kernels, // (K_C, C, K_H, K_W)
    const float *bias,    // (K_C)
    const int B, const int C, const int H, const int W,
    const int K_C, const int K_H, const int K_W)
{
    int out_H = H - K_H + 1;
    int out_W = W - K_W + 1;

    // din = conv2dfull(in, rot180(dout))
    if (din != NULL) {
        // _conv2d_forward(din, in, dout, NULL, B, C, H, W, K_C, K_H, K_W, true);
        int pad_h = (K_H - 1);
        int pad_w = (K_W - 1);
        float * doutp = calloc(B * K_C * (out_H + pad_h * 2) * (out_W + pad_w * 2), sizeof(float));
        for(int b = 0; b < B; b++) {
            for (int k_c = 0; k_c < K_C; k_c++) {
                for (int i = 0; i < out_H; i++) {
                    for (int j = 0; j < out_W; j++) {
                        doutp[(b * K_C * out_H * out_W) + (k_c * out_H * out_W) + ((i + pad_h) * out_W) + (j + pad_w)] = dout[(b * K_C * out_H * out_W) + (k_c * out_H * out_W) + (i * out_W) + j];
                    }
                }
            }
        }

        for (int b = 0; b < B; b++) {
            for (int k_c = 0; k_c < K_C; k_c++) {
                for (int i = 0; i < H; i++) {
                    for (int j = 0; j < W; j++) {
                        for (int k_h = 0; k_h < K_H; k_h++) {
                            for (int k_w = 0; k_w < K_W; k_w++) {
                                for (int c = 0; c < C; c++) {
                                    // if ((i + k_h - pad_h) > pad_h && (i + k_h - pad_h) < H - pad_h
                                    //     && (j + k_w - pad_w) > pad_w && (j + k_w - pad_w) < W - pad_w) {
                                    //     din[(b * C * H * W) + (c * H * W) + (i * W) + j] = dout[(b * K_C * out_H * out_W) + (k_c * out_H * out_W) + ((i + k_h - pad_h) * out_W) + (j + k_w - pad_w)]
                                    //                                             * kernels[(k_c * C * K_H * K_W) + (c * K_H * K_W) + ((K_H - k_h - 1) * K_W) + (K_W - k_w - 1)];
                                    // }
                                    din[(b * C * H * W) + (c * H * W) + (i * W) + j] += doutp[(b * K_C * out_H * out_W) + (k_c * out_H * out_W) + ((i + k_h) * out_W) + (j + k_w)]
                                        * kernels[(k_c * C * K_H * K_W) + (c * K_H * K_W) + ((K_H - k_h - 1) * K_W) + (K_W - k_w - 1)];
                                }
                            }
                        }
                    }
                }
            }
        }
        free(doutp);
    }

    // dkernels = sum(conv2d(in, dout), dim=0)
    // for n in range(N):       # On parcourt toutes les images
    //     for f in range(F):   # On parcourt tous les filtres
    //         for i in range(HH): # indices du résultat
    //             for j in range(WW):
    //                 for k in range(H_): # indices du filtre
    //                     for l in range(W_):
    //                         for c in range(C): # profondeur
    //                             dw[f,c,i,j] += xp[n, c, stride*i+k, stride*j+l] * dout[n, f, k, l]

    for (int b = 0; b < B; b++) {
        for (int k_c = 0; k_c < K_C; k_c++) {
            for (int k_h = 0; k_h < K_H; k_h++) {
                for (int k_w = 0; k_w < K_W; k_w++) {
                    for (int i = 0; i < out_H; i++) {
                        for (int j = 0; j < out_W; j++) {
                            for (int c = 0; c < C; c++) {
                                dkernels[(k_c * C * K_H * K_W) + (c * K_H * K_W) + (k_h * K_W) + k_w] += in[(b * C * H * W) + (c * H * W) + ((i + k_h) * W) + (j + k_w)]
                                    * dout[(b * K_C * out_H * out_W) + (k_c * out_H * out_W) + (i * out_W) + j];
                            }
                        }
                    }
                }
            }
        }
    }


    // dbias[k_c] = sum(dout[b][k_c][out_h][out_w]) (sum on all axis except filters)
    for (int k_c = 0; k_c < K_C; k_c++) {
        for (int b = 0; b < B; b++) {
            for (int j = 0; j < out_H; j++) {
                for (int i = 0; i < out_W; i++) {
                    dbias[k_c] += dout[(b * K_C * out_H * out_W) + (k_c * out_H * out_W) + (j * out_W) + i];
                }
            }
        }
    }
}

void relu_forward(float *out, const float *in, const size_t N)
{
    for (size_t i = 0; i < N; i++)
    {
        float tmp = in[i];
        out[i] = (tmp > 0) ? tmp : 0;
    }
}

void relu_backward(float *din, const float *dout, const float *in, const size_t N) {
    for (size_t i = 0; i < N; i++) {
        din[i] = (in[i] > 0.0) * dout[i];
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
    int out_H = H / K_H;
    int out_W = W / K_W;

    #pragma omp parallel for
    for (int b = 0; b < B; b++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int j = 0; j < out_H; j++)
            {
                for (int i = 0; i < out_W; i++)
                {
                    float max = in[(b * C * H * W) + (c * H * W) + (j * K_H * W) + (i * K_W)]; // init to first or NEG_INF ?
                    for (int k_j = 0; k_j < K_H; k_j++)
                    {
                        for (int k_i = 0; k_i < K_H; k_i++)
                        {
                            float v = in[(b * C * H * W) + (c * H * W) + ((j * K_H + k_j) * W) + (i * K_W + k_i)];
                            if (v > max)
                            {
                                max = v;
                            }
                        }
                    }
                    out[(b * C * out_H * out_W) + (c * out_H * out_W) + (j * out_W) + i] = max;
                }
            }
        }
    }
}

void maxpool2d_backward(
    // we are using stride = kernel size here. e.g: (2, 2) kernel => (2, 2) stride
    // out_H = H / K_H
    // out_W = W / K_W
    float *din, // (B, C, H, W)
    const float *dout,      // (B, C, out_H, out_W)
    const float *in, // (B, C, H, W)
    const int B, const int C, const int H, const int W,
    const int K_W, const int K_H)
{
    int out_H = H / K_H;
    int out_W = W / K_W;
    // here, "din" index which has the max in "in" gets assigned the gradient
    #pragma omp parallel for
    for (int b = 0; b < B; b++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int j = 0; j < out_H; j++)
            {
                for (int i = 0; i < out_W; i++)
                {
                    int argmax = (b * C * H * W) + (c * H * W) + (j * K_H * W) + (i * K_W); // init to first or NEG_INF ?
                    for (int k_j = 0; k_j < K_H; k_j++)
                    {
                        for (int k_i = 0; k_i < K_H; k_i++)
                        {
                            int v_i = (b * C * H * W) + (c * H * W) + ((j * K_H + k_j) * W) + (i * K_W + k_i);
                            if (in[v_i] > in[argmax])
                            {
                                argmax = v_i;
                            }
                        }
                    }
                    din[argmax] = dout[(b * C * out_H * out_W) + (c * out_H * out_W) + (j * out_W) + i];
                }
            }
        }
    }
}

// out = x @ weight.T + bias
void linear_forward(
    float *out,          // (B, out_features)
    const float *x,      // (B, in_features)
    const float *weight, // (out_features, in_features)
    const float *bias,   // (out_features)
    const int B, const int in_features, const int out_features)
{
    #pragma omp parallel for
    for (int b = 0; b < B; b++)
    {
        for (int i = 0; i < out_features; i++)
        {
            float acc = (bias != NULL) ? bias[i] : 0.0;
            for (int k = 0; k < in_features; k++)
            {
                acc += x[(b * in_features) + k] * weight[i * in_features + k];
            }
            out[b * out_features + i] = acc;
        }
    }
}

void linear_backward(
    float *din,           // (B, in_features)
    float *dweight,       // (out_features, in_features)
    float *dbias,         // (out_features)
    const float *dout,    // (B, out_features)
    const float *in,      // (B, in_features)
    const float *weight,  // (out_features, in_features)
    const int B, const int in_features, const int out_features)
{
    // din = dout @ weight
    #pragma omp parallel for
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < in_features; i++) {
            float acc = 0.0;
            for (int o = 0; o < out_features; o++) {
                acc += dout[b * out_features + o] * weight[o * in_features + i];
            }
            din[b * out_features + i] = acc;
        }
    }

    // dweight = dout.T @ in
    #pragma omp parallel for
    for (int o = 0; o < out_features; o++) {
        for (int i = 0; i < in_features; i++) {
            float acc = 0.0;
            for (int b = 0; b < B; b++) {
                acc += dout[b * out_features + o] * in[b * in_features + i];
            }
            dweight[o * in_features + i] = acc;
        }
    }

    // dbias = sum(dout, axis=1)    // sum on the out_features axis
    #pragma omp parallel for
    for (int b = 0; b < B; b++)
    {
        for (int o = 0; o < out_features; o++)
        {
            dbias[o] += dout[b * out_features + o];
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

void softmax_forward(
    float *probs,      // (B, C)
    const float *logits, // (B, C)
    const int B, const int C // C is for classes
) {
    // for each batch
    //      probs = exp(x) / sum(exp(x))
    for (int b = 0; b < B; b++) {
        float *probs_b = probs + b * C;
        const float *logits_b = logits + b * C;

        // maxval is only calculated and subtracted for numerical stability
        float maxval = -10000.0f; // TODO something better
        for (int i = 0; i < C; i++) {
            if (logits_b[i] > maxval) {
                maxval = logits_b[i];
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < C; i++) {
            probs_b[i] = expf(logits_b[i] - maxval);
            sum += probs_b[i];
        }
        for (int i = 0; i < C; i++) {
            probs_b[i] /= sum;
        }
    }
}

// computes the mean loss over the batch
void sparse_categorical_crossentropy_forward(
    float *losses,      // (B,)
    const float *probs, // (B, C)
    const int *targets, // (B,)
    const int B, const int C // C is for classes
) {
    for (int b = 0; b < B; b++) {
        int target_class = targets[b];
        losses[b] = -logf(probs[b * C + target_class]);
    }
}

void sparse_categorical_crossentropy_softmax_backward(
    float* dlogits, // (B,C)
    const float* dlosses, // (B,)
    const float* probs,   // (B,C)
    const int* targets,   // (B,)
    const int B, const int C    // C is for classes
) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        float* dlogits_b = dlogits + b * C;
        const float* probs_b = probs + b * C;
        float dloss = dlosses[b]; // dloss for this batch index
        int ix = targets[b]; // target class for this batch index
        for (int i = 0; i < C; i++) {
            float p = probs_b[i];
            float indicator = i == ix ? 1.0f : 0.0f;
            dlogits_b[i] += (p - indicator) * dloss;
        }
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

#define CONV2D_2_C 32 // conv2
#define CONV2D_2_OC 32
#define CONV2D_2_KS 5
#define CONV2D_2_OS (CONV2D_1_OS - CONV2D_2_KS + 1)

#define MAXPOOL2D_1_KS 2 // maxpool1
#define MAXPOOL2D_1_OS (CONV2D_2_OS / MAXPOOL2D_1_KS)

#define CONV2D_3_C 32 // conv3
#define CONV2D_3_OC 64
#define CONV2D_3_KS 3
#define CONV2D_3_OS (MAXPOOL2D_1_OS - CONV2D_3_KS + 1)

#define CONV2D_4_C 64 // conv4
#define CONV2D_4_OC 64
#define CONV2D_4_KS 3
#define CONV2D_4_OS (CONV2D_3_OS - CONV2D_4_KS + 1)

#define MAXPOOL2D_2_KS 2 // maxpool1
#define MAXPOOL2D_2_OS (CONV2D_4_OS / MAXPOOL2D_2_KS)

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

#define NUM_ACTIVATION_TENSORS 13
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
    float *probs;       // (B, LINEAR_1_OF)
    float* losses;        // (B,)
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
    act_sizes[11] = B * LINEAR_1_OF;                                  // softmax
    act_sizes[12] = B;                                                // losses
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
        &acts->linear_1,
        &acts->probs,
        &acts->losses
    };
    float *acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++)
    {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

struct Model {
    // the weights (parameters) of the model, and their sizes
    struct ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    struct ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    struct ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    struct ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    float* inputs; // the input images for the current forward pass
    int* targets; // the target labels for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
};

void model_forward(struct Model *model, const float *inputs, const int* targets, const int B) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // allocate space for all the activations if needed (done here, lazily)
    if(model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = B;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, B);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (float*)malloc(B * sizeof(float));
        model->targets = (int*)malloc(B * sizeof(int)); // might be unused if we never have targets but it's small
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size) {
            printf("Model: B=%d, Desired: B=%d\n", model->batch_size, (int)B);
            exit(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets
    memcpy(model->inputs, inputs, B * sizeof(float));
    if (targets != NULL) {
        memcpy(model->targets, targets, B * sizeof(int));
    }

    // forward pass
    struct ParameterTensors params = model->params; // for brevity
    struct ActivationTensors acts = model->acts;

    conv2d_forward(acts.conv2d_1, inputs, params.conv1w, params.conv1b, B, CONV2D_1_C, IMAGE_SIZE, IMAGE_SIZE, CONV2D_1_OC, CONV2D_1_KS, CONV2D_1_KS);
    relu_forward(acts.conv2d_1_relu, acts.conv2d_1, B * CONV2D_1_OC * CONV2D_1_OS * CONV2D_1_OS);

    conv2d_forward(acts.conv2d_2, acts.conv2d_1_relu, params.conv2w, params.conv2b, B, CONV2D_2_C, CONV2D_1_OS, CONV2D_1_OS, CONV2D_2_OC, CONV2D_2_KS, CONV2D_2_KS);
    relu_forward(acts.conv2d_2_relu, acts.conv2d_2, B * CONV2D_2_OC * CONV2D_2_OS * CONV2D_2_OS);

    maxpool2d_forward(acts.maxpool2d_1, acts.conv2d_2_relu, B, CONV2D_2_OC, CONV2D_2_OS, CONV2D_2_OS, MAXPOOL2D_1_KS, MAXPOOL2D_1_KS);

    conv2d_forward(acts.conv2d_3, acts.maxpool2d_1, params.conv3w, params.conv3b, B, CONV2D_3_C, MAXPOOL2D_1_OS, MAXPOOL2D_1_OS, CONV2D_3_OC, CONV2D_3_KS, CONV2D_3_KS);
    relu_forward(acts.conv2d_3_relu, acts.conv2d_3, B * CONV2D_3_OC * CONV2D_3_OS * CONV2D_3_OS);

    conv2d_forward(acts.conv2d_4, acts.conv2d_3_relu, params.conv4w, params.conv4b, B, CONV2D_4_C, CONV2D_3_OS, CONV2D_3_OS, CONV2D_4_OC, CONV2D_4_KS, CONV2D_4_KS);
    relu_forward(acts.conv2d_4_relu, acts.conv2d_4, B * CONV2D_4_OC * CONV2D_4_OS * CONV2D_4_OS);

    maxpool2d_forward(acts.maxpool2d_2, acts.conv2d_4_relu, B, CONV2D_4_OC, CONV2D_4_OS, CONV2D_4_OS, MAXPOOL2D_2_KS, MAXPOOL2D_2_KS);
    linear_forward(acts.linear_1, acts.maxpool2d_2, params.linear1w, params.linear1b, B, LINEAR_1_IF, LINEAR_1_OF);
    softmax_forward(acts.probs, acts.linear_1, B, LINEAR_1_OF);

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        sparse_categorical_crossentropy_forward(model->acts.losses, acts.probs, targets, B, LINEAR_1_OF);
        // for convenience also evaluate the mean loss
        float mean_loss = 0.0f;
        for (int i=0; i<B; i++) { mean_loss += model->acts.losses[i]; }
        mean_loss /= B;
        model->mean_loss = mean_loss;
    } else {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }
}

void model_zero_grad(struct Model *model) {
    if(model->grads_memory != NULL) { memset(model->grads_memory, 0, model->num_parameters * sizeof(float)); }
    if(model->grads_acts_memory != NULL) { memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float)); }
}

void model_backward(struct Model *model) {

    // double check we forwarded previously, with targets
    if (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model->grads_memory == NULL) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
        model_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_size;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    struct ParameterTensors params = model->params; // for brevity
    struct ParameterTensors grads = model->grads;
    struct ActivationTensors acts = model->acts;
    struct ActivationTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,) positions in the batch
    float dloss_mean = 1.0f / B;
    for (int i = 0; i < B; i++) { grads_acts.losses[i] = dloss_mean; }

    sparse_categorical_crossentropy_softmax_backward(grads_acts.linear_1, grads_acts.losses, acts.probs, model->targets, B, LINEAR_1_OF);
    linear_backward(grads_acts.maxpool2d_2, grads.linear1w, grads.linear1b, grads_acts.linear_1, acts.maxpool2d_2, params.linear1w, B, LINEAR_1_IF, LINEAR_1_OF);
    maxpool2d_backward(grads_acts.conv2d_4_relu, grads_acts.maxpool2d_2, acts.conv2d_4_relu, B, CONV2D_4_OC, CONV2D_4_OS, CONV2D_4_OS, MAXPOOL2D_2_KS, MAXPOOL2D_2_KS);

    relu_backward(grads_acts.conv2d_4, grads_acts.conv2d_4_relu, acts.conv2d_4, B * CONV2D_4_OC * CONV2D_4_OS * CONV2D_4_OS);
    conv2d_backward(grads_acts.conv2d_3_relu, grads.conv4w, grads.conv4b, grads_acts.conv2d_4, acts.conv2d_3_relu, params.conv4w, params.conv4b, B, CONV2D_4_C, CONV2D_3_OS, CONV2D_3_OS, CONV2D_4_OC, CONV2D_4_KS, CONV2D_4_KS);

    relu_backward(grads_acts.conv2d_3, grads_acts.conv2d_3_relu, acts.conv2d_3, B * CONV2D_3_OC * CONV2D_3_OS * CONV2D_3_OS);
    conv2d_backward(grads_acts.maxpool2d_1, grads.conv3w, grads.conv3b, grads_acts.conv2d_3, acts.maxpool2d_1, params.conv3w, params.conv3b, B, CONV2D_3_C, MAXPOOL2D_1_OS, MAXPOOL2D_1_OS, CONV2D_3_OC, CONV2D_3_KS, CONV2D_3_KS);

    maxpool2d_backward(grads_acts.conv2d_2_relu, grads_acts.maxpool2d_1, acts.conv2d_2_relu, B, CONV2D_2_OC, CONV2D_2_OS, CONV2D_2_OS, MAXPOOL2D_1_KS, MAXPOOL2D_1_KS);

    relu_backward(grads_acts.conv2d_2, grads_acts.conv2d_2_relu, acts.conv2d_2, B * CONV2D_2_OC * CONV2D_2_OS * CONV2D_2_OS);
    conv2d_backward(grads_acts.conv2d_1_relu, grads.conv2w, grads.conv2b, grads_acts.conv2d_2, acts.conv2d_1_relu, params.conv2w, params.conv2b, B, CONV2D_2_C, CONV2D_1_OS, CONV2D_1_OS, CONV2D_2_OC, CONV2D_2_KS, CONV2D_2_KS);

    relu_backward(grads_acts.conv2d_1, grads_acts.conv2d_1_relu, acts.conv2d_1, B * CONV2D_1_OC * CONV2D_1_OS * CONV2D_1_OS);
    conv2d_backward(NULL, grads.conv1w, grads.conv1b, grads_acts.conv2d_1, model->inputs, params.conv1w, params.conv1b, B, CONV2D_1_C, IMAGE_SIZE, IMAGE_SIZE, CONV2D_1_OC, CONV2D_1_KS, CONV2D_1_KS);
}

void model_build_from_checkpoint(struct Model *model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopen(checkpoint_path, "rb");

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("loaded num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    fread(model->params_memory, sizeof(float), num_parameters, model_file);
    fclose(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

void model_build_init_weights(struct Model *model) {
    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("loaded num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);

    // init conv2d 1
    {
        float k = 1.0/(CONV2D_1_C * CONV2D_1_KS * CONV2D_1_KS);
        float sqrt_k = sqrtf(k);
        fill_uniform(model->params.conv1w, -sqrt_k, sqrt_k, CONV2D_1_OC * CONV2D_1_C * CONV2D_1_KS * CONV2D_1_KS);
        fill_uniform(model->params.conv1b, -sqrt_k, sqrt_k, CONV2D_1_OC);
    }

    // init conv2d 2
    {
        float k = 1.0/(CONV2D_2_C * CONV2D_2_KS * CONV2D_2_KS);
        float sqrt_k = sqrtf(k);
        fill_uniform(model->params.conv1w, -sqrt_k, sqrt_k, CONV2D_2_OC * CONV2D_2_C * CONV2D_2_KS * CONV2D_2_KS);
        fill_uniform(model->params.conv1b, -sqrt_k, sqrt_k, CONV2D_2_OC);
    }

    // init conv2d 3
    {
        float k = 1.0/(CONV2D_3_C * CONV2D_3_KS * CONV2D_3_KS);
        float sqrt_k = sqrtf(k);
        fill_uniform(model->params.conv1w, -sqrt_k, sqrt_k, CONV2D_3_OC * CONV2D_3_C * CONV2D_3_KS * CONV2D_3_KS);
        fill_uniform(model->params.conv1b, -sqrt_k, sqrt_k, CONV2D_3_OC);
    }

    // init conv2d 4
    {
        float k = 1.0/(CONV2D_4_C * CONV2D_4_KS * CONV2D_4_KS);
        float sqrt_k = sqrtf(k);
        fill_uniform(model->params.conv1w, -sqrt_k, sqrt_k, CONV2D_4_OC * CONV2D_4_C * CONV2D_4_KS * CONV2D_4_KS);
        fill_uniform(model->params.conv1b, -sqrt_k, sqrt_k, CONV2D_4_OC);
    }

    // init linear
    {
        float k = 1.0/LINEAR_1_IF;
        float sqrt_k = sqrtf(k);
        fill_uniform(model->params.linear1w, -sqrt_k, sqrt_k, LINEAR_1_OF * LINEAR_1_IF);
        fill_uniform(model->params.linear1b, -sqrt_k, sqrt_k, LINEAR_1_OF);
    }

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

void model_free(struct Model *model) {
    free(model->params_memory);
    free(model->grads_memory);
    free(model->m_memory);
    free(model->v_memory);
    free(model->acts_memory);
    free(model->grads_acts_memory);
    free(model->inputs);
    free(model->targets);
}


void model_update(struct Model *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL) {
        model->m_memory = (float*)calloc(model->num_parameters, sizeof(float));
        model->v_memory = (float*)calloc(model->num_parameters, sizeof(float));
    }

    for (size_t i = 0; i < model->num_parameters; i++) {
        float param = model->params_memory[i];
        float grad = model->grads_memory[i];

        // update the first moment (momentum)
        float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        // update
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

// end model

int main()
{
    struct Model model;
    model_build_from_checkpoint(&model, "params.bin");
    // model_build_init_weights(&model);

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

    int B = BATCH_SIZE;

    struct DataLoader train_loader, test_loader;
    dataloader_init(&train_loader, X_train, Y_train, train_len, B);
    dataloader_init(&test_loader, X_test, Y_test, test_len, B);

    struct timespec start, end;
    for (int step = 0; step < MAX_STEPS; step++) {
        if ((step % 10 == 0) || (step == MAX_STEPS - 1)) {
            float test_loss = 0.0;
            float accuracy = 0.0;
            for (int i = 0; i < EVAL_STEPS; i++) {
                dataloader_next_batch(&test_loader);
                model_forward(&model, test_loader.inputs, test_loader.targets, B);
                test_loss += model.mean_loss;
                int argmax[B];
                argmax_forward(argmax, model.acts.probs, B, LINEAR_1_OF);
                for (int b = 0; b < B; b++) {
                    if (argmax[b] == test_loader.targets[b]) {
                        accuracy++;
                    }
                }
            }
            accuracy /= EVAL_STEPS * B;
            test_loss /= EVAL_STEPS;
            printf("accuracy: %.1f%% val loss %f\n", accuracy * 100.0, test_loss);
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        model_forward(&model, train_loader.inputs, train_loader.targets, B);
        model_zero_grad(&model);
        model_backward(&model);
        model_update(&model, 0.001, 0.9, 0.999, 1e-8, 0.01, step+1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }

    printf("final inference samples:\n");
    int argmax[B];
    argmax_forward(argmax, model.acts.probs, B, LINEAR_1_OF);
    for (int i = 0; i < B; i++)
    {
        printf("    y_pred = %d | y = %d\n", argmax[i], model.targets[i]);
    }

    dataloader_free(&test_loader);
    dataloader_free(&train_loader);
    model_free(&model);
    free(X_train);
    free(Y_train);
    free(X_test);
    free(Y_test);
    return 0;
}
