#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <endian.h>
#include <string.h>

typedef enum _LayerType {
    LAYER_INPUT = 0,
    LAYER_FULL,
    LAYER_CONV
} LayerType;


typedef struct Layer {

    int lid;                    /* Layer ID */
    struct Layer* lprev;       /* Previous layer pointer */
    struct Layer* lnext;       /* Next layer pointer */
    int depth, width, height;   /* Shape */
    int nnodes;                 /* Number of nodes */
    double* outputs;            /* Output values of nodes */
    double* gradients;          /* Gradients for backpropagation */
    double* errors;             /* Computed errors for training */
    int nbiases;                /* Number of biases */
    double* biases;             /* Bias values */
    double* u_biases;           /* Updates to biases from training */
    int nweights;               /* Number of weights */
    double* weights;            /* Weight values */
    double* u_weights;          /* Updates to weights from training */

    LayerType ltype;            /* Layer type */
    union {
        struct {
        } full;
        struct {
            int kernsize;       /* kernel size (>0) */
            int padding;        /* padding size */
            int stride;         /* stride (>0) */
        } conv;
    };

} Layer;

/* rnd(): uniform random [0.0, 1.0] */
static inline double rnd(){return ((double)rand() / RAND_MAX);}

/* nrnd(): normal random (std=1.0) */
static inline double nrnd(){return (rnd()+rnd()+rnd()+rnd()-2.0) * 1.724; /* std=1.0 */}

/* tanh_g(y): hyperbolic tangent gradient */
static inline double tanh_g(double y){return 1.0 - y*y;}

/* relu(x): ReLU */
static inline double relu(double x){return (0 < x)? x : 0;}
/* relu_g(y): ReLU gradient */
static inline double relu_g(double y){return (0 < y)? 1 : 0;}


static Layer* Layer_create(
    Layer* lprev, LayerType ltype,
    int depth, int width, int height,
    int nbiases, int nweights)
{
    Layer* self = (Layer*)calloc(1, sizeof(Layer));
    if (self == NULL) return NULL;

    self->lprev = lprev;
    self->lnext = NULL;
    self->ltype = ltype;
    self->lid = 0;
    if (lprev != NULL) {
        lprev->lnext = self;
        self->lid = lprev->lid+1;
    }
    self->depth = depth;
    self->width = width;
    self->height = height;

    self->nnodes = depth * width * height;
    self->outputs = (double*)calloc(self->nnodes, sizeof(double));
    self->gradients = (double*)calloc(self->nnodes, sizeof(double));
    self->errors = (double*)calloc(self->nnodes, sizeof(double));

    self->nbiases = nbiases;
    self->biases = (double*)calloc(self->nbiases, sizeof(double));
    self->u_biases = (double*)calloc(self->nbiases, sizeof(double));

    self->nweights = nweights;
    self->weights = (double*)calloc(self->nweights, sizeof(double));
    self->u_weights = (double*)calloc(self->nweights, sizeof(double));

    return self;
}

void Layer_destroy(Layer* self){
    free(self->outputs);
    free(self->gradients);
    free(self->errors);

    free(self->biases);
    free(self->u_biases);
    free(self->weights);
    free(self->u_weights);

    free(self);
}

static void Layer_feedForw_full(Layer* self){
    Layer* lprev = self->lprev;

    int k = 0;
    for (int i = 0; i < self->nnodes; i++) {

        double x = self->biases[i];
        for (int j = 0; j < lprev->nnodes; j++)
            x += (lprev->outputs[j] * self->weights[k++]);
        self->outputs[i] = x;
    }

    if (self->lnext == NULL) {

        double m = -1;
        for (int i = 0; i < self->nnodes; i++) {
            double x = self->outputs[i];
            if (m < x) { m = x; }
        }
        double t = 0;
        for (int i = 0; i < self->nnodes; i++) {
            double x = self->outputs[i];
            double y = exp(x-m);
            self->outputs[i] = y;
            t += y;
        }
        for (int i = 0; i < self->nnodes; i++) {
            self->outputs[i] /= t;
            self->gradients[i] = 1;
        }
    } else {
        for (int i = 0; i < self->nnodes; i++) {
            double x = self->outputs[i];
            double y = tanh(x);
            self->outputs[i] = y;
            self->gradients[i] = tanh_g(y);
        }
    }
}

static void Layer_feedBack_full(Layer* self){
    Layer* lprev = self->lprev;

    for (int j = 0; j < lprev->nnodes; j++) lprev->errors[j] = 0;
    
    int k = 0;
    for (int i = 0; i < self->nnodes; i++) {
        double dnet = self->errors[i] * self->gradients[i];
        for (int j = 0; j < lprev->nnodes; j++) {
            lprev->errors[j] += self->weights[k] * dnet;
            self->u_weights[k] += dnet * lprev->outputs[j];
            k++;
        }
        self->u_biases[i] += dnet;
    }

}

__global__ void conv_forward_kernel(double* input, double* output, double* weights, double* biases,
                                    int inputWidth, int inputHeight, int inputDepth,
                                    int outputWidth, int outputHeight, int outputDepth,
                                    int kernelSize, int padding, int stride) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int tz = threadIdx.z + blockIdx.z * blockDim.z;

    if (tx < outputWidth && ty < outputHeight && tz < outputDepth) {
        double value = 0.0;
        int from_x = tx * stride - padding;
        int from_y = ty * stride - padding;
        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                for (int k = 0; k < inputDepth; ++k) {
                    int ix = from_x + i;
                    int iy = from_y + j;
                    if (ix >= 0 && ix < inputWidth && iy >= 0 && iy < inputHeight) {
                        int input_idx = (k * inputHeight + iy) * inputWidth + ix;
                        int weight_idx = ((tz * inputDepth + k) * kernelSize + j) * kernelSize + i;
                        value += input[input_idx] * weights[weight_idx];
                    }
                }
            }
        }
        int bias_idx = tz;
        output[(tz * outputHeight + ty) * outputWidth + tx] = value + biases[bias_idx];
    }
}


void forward_convolution_layer(Layer* layer) {
    double* d_output;
    cudaMalloc(&d_output, sizeof(double) * layer->depth * layer->width * layer->height);
    double* d_weights;
    double* d_biases;
    cudaMalloc(&d_weights, sizeof(double) * layer->nweights);
    cudaMalloc(&d_biases, sizeof(double) * layer->nbiases);
    cudaMemcpy(d_weights, layer->weights, sizeof(double) * layer->nweights, cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, layer->biases, sizeof(double) * layer->nbiases, cudaMemcpyHostToDevice);
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((layer->width  + 15) / 16, (layer->height + 15) / 16, layer->depth);
    conv_forward_kernel<<<gridDim, blockDim>>>(layer->lprev->output, d_output, d_weights, d_biases,
                                               layer->prev->width, layer->prev->height, layer->prev->depth,
                                               layer->width, layer->height, layer->depth,
                                               layer->conv.kernsize, layer->conv.padding, layer->conv.stride);

    cudaMemcpy(layer->outputs, d_output, sizeof(double) * layer->depth * layer->width * layer->height, cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
}

static void Layer_feedBack_conv(Layer* self){
    Layer* lprev = self->lprev;
    for (int j = 0; j < lprev->nnodes; j++) lprev->errors[j] = 0;

    int kernsize = self->conv.kernsize;
    int i = 0;
    for (int z1 = 0; z1 < self->depth; z1++) {
        int qbase = z1 * lprev->depth * kernsize * kernsize;
        for (int y1 = 0; y1 < self->height; y1++) {
            int y0 = self->conv.stride * y1 - self->conv.padding;
            for (int x1 = 0; x1 < self->width; x1++) {
                int x0 = self->conv.stride * x1 - self->conv.padding;
                double dnet = self->errors[i] * self->gradients[i];
                for (int z0 = 0; z0 < lprev->depth; z0++) {
                    int pbase = z0 * lprev->width * lprev->height;
                    for (int dy = 0; dy < kernsize; dy++) {
                        int y = y0+dy;
                        if (0 <= y && y < lprev->height) {
                            int p = pbase + y*lprev->width;
                            int q = qbase + dy*kernsize;
                            for (int dx = 0; dx < kernsize; dx++) {
                                int x = x0+dx;
                                if (0 <= x && x < lprev->width) {
                                    lprev->errors[p+x] += self->weights[q+dx] * dnet;
                                    self->u_weights[q+dx] += dnet * lprev->outputs[p+x];
                                }
                            }
                        }
                    }
                }
                self->u_biases[z1] += dnet;
                i++;
            }
        }
    }
}

void Layer_setInputs(Layer* self, const double* values){
    for (int i = 0; i < self->nnodes; i++) 
        self->outputs[i] = values[i];
    
    Layer* layer = self->lnext;
    while (layer != NULL) {
        switch (layer->ltype) {
        case LAYER_FULL:
            Layer_feedForw_full(layer);
            break;
        case LAYER_CONV:
            Layer_feedForw_conv(layer);
            break;
        default:
            break;
        }
        layer = layer->lnext;
    }
}

void Layer_getOutputs(const Layer* self, double* outputs){
    for (int i = 0; i < self->nnodes; i++) 
        outputs[i] = self->outputs[i];
}

double Layer_getErrorTotal(const Layer* self){
    double total = 0;
    for (int i = 0; i < self->nnodes; i++) {
        double e = self->errors[i];
        total += e*e;
    }
    return (total / self->nnodes);
}

void Layer_learnOutputs(Layer* self, const double* values){
    for (int i = 0; i < self->nnodes; i++) 
        self->errors[i] = (self->outputs[i] - values[i]);

    Layer* layer = self;
    while (layer != NULL) {
        switch (layer->ltype) {
        case LAYER_FULL:
            Layer_feedBack_full(layer);
            break;
        case LAYER_CONV:
            Layer_feedBack_conv(layer);
            break;
        default:
            break;
        }
        layer = layer->lprev;
    }
}

void Layer_update(Layer* self, double rate){
    for (int i = 0; i < self->nbiases; i++) {
        self->biases[i] -= rate * self->u_biases[i];
        self->u_biases[i] = 0;
    }
    for (int i = 0; i < self->nweights; i++) {
        self->weights[i] -= rate * self->u_weights[i];
        self->u_weights[i] = 0;
    }
    if (self->lprev != NULL) 
        Layer_update(self->lprev, rate);
}

Layer* Layer_create_input(int depth, int width, int height){return Layer_create(NULL, LAYER_INPUT, depth, width, height, 0, 0);}

Layer* Layer_create_full(Layer* lprev, int nnodes, double std){
    Layer* self = Layer_create(
        lprev, LAYER_FULL, nnodes, 1, 1,
        nnodes, nnodes * lprev->nnodes);

    for (int i = 0; i < self->nweights; i++) 
        self->weights[i] = std * nrnd();
    return self;
}

Layer* Layer_create_conv(
    Layer* lprev, int depth, int width, int height,
    int kernsize, int padding, int stride, double std){
    Layer* self = Layer_create(
        lprev, LAYER_CONV, depth, width, height,
        depth, depth * lprev->depth * kernsize * kernsize);

    self->conv.kernsize = kernsize;
    self->conv.padding = padding;
    self->conv.stride = stride;

    for (int i = 0; i < self->nweights; i++) 
        self->weights[i] = std * nrnd();
    return self;
}


typedef struct _IdxFile
{
    int ndims;
    uint32_t* dims;
    uint8_t* data;
} IdxFile;

IdxFile* IdxFile_read(FILE* fp){

    struct {
        uint16_t magic;
        uint8_t type;
        uint8_t ndims;

    } header;
    if (fread(&header, sizeof(header), 1, fp) != 1) return NULL;
    if (header.magic != 0) return NULL;
    if (header.type != 0x08) return NULL;
    if (header.ndims < 1) return NULL;

    IdxFile* self = (IdxFile*)calloc(1, sizeof(IdxFile));
    if (self == NULL) return NULL;
    self->ndims = header.ndims;
    self->dims = (uint32_t*)calloc(self->ndims, sizeof(uint32_t));
    if (self->dims == NULL) return NULL;
    
    if (fread(self->dims, sizeof(uint32_t), self->ndims, fp) == self->ndims) {
        uint32_t nbytes = sizeof(uint8_t);
        for (int i = 0; i < self->ndims; i++) {
            uint32_t size = be32toh(self->dims[i]);
            nbytes *= size;
            self->dims[i] = size;
        }
        self->data = (uint8_t*) malloc(nbytes);
    }

    return self;
}

void IdxFile_destroy(IdxFile* self){
    if (self->dims != NULL) {
        free(self->dims);
        self->dims = NULL;
    }
    if (self->data != NULL) {
        free(self->data);
        self->data = NULL;
    }
}


uint8_t IdxFile_get1(IdxFile* self, int i){
    return self->data[i];
}


void IdxFile_get3(IdxFile* self, int i, uint8_t* out){
    size_t n = self->dims[1] * self->dims[2];
    memcpy(out, &self->data[i*n], n);
}


/* main */
int main(int argc, char* argv[])
{
    /* argv[1] = train images */
    /* argv[2] = train labels */
    /* argv[3] = test images */
    /* argv[4] = test labels */
    if (argc < 4) return 100;

    /* Use a fixed random seed for debugging. */
    srand(0);
    /* Initialize layers. */
    /* Input layer - 1x28x28. */
    Layer* linput = Layer_create_input(1, 28, 28);
    /* Conv1 layer - 16x14x14, 3x3 conv, padding=1, stride=2. */
    /* (14-1)*2+3 < 28+1*2 */
    Layer* lconv1 = Layer_create_conv(linput, 16, 14, 14, 3, 1, 2, 0.1);
    /* Conv2 layer - 32x7x7, 3x3 conv, padding=1, stride=2. */
    /* (7-1)*2+3 < 14+1*2 */
    Layer* lconv2 = Layer_create_conv(lconv1, 32, 7, 7, 3, 1, 2, 0.1);
    /* FC1 layer - 200 nodes. */
    Layer* lfull1 = Layer_create_full(lconv2, 200, 0.1);
    /* FC2 layer - 200 nodes. */
    Layer* lfull2 = Layer_create_full(lfull1, 200, 0.1);
    /* Output layer - 10 nodes. */
    Layer* loutput = Layer_create_full(lfull2, 10, 0.1);

    /* Read the training images & labels. */
    IdxFile* images_train = NULL;
    FILE* fp = fopen(argv[1], "rb");
    if (fp == NULL) return 111;
    images_train = IdxFile_read(fp);
    if (images_train == NULL) return 111;
    fclose(fp);
    
    IdxFile* labels_train = NULL;
    
    fp = fopen(argv[2], "rb");
    if (fp == NULL) return 111;
    labels_train = IdxFile_read(fp);
    if (labels_train == NULL) return 111;
    fclose(fp);

    fprintf(stderr, "training...\n");
    double rate = 0.1;
    double etotal = 0;
    int nepoch = 10;
    int batch_size = 32;
    int train_size = images_train->dims[0];
    for (int i = 0; i < nepoch * train_size; i++) {
        uint8_t img[28*28];
        double x[28*28];
        double y[10];
        int index = rand() % train_size;
        IdxFile_get3(images_train, index, img);
        for (int j = 0; j < 28*28; j++) x[j] = img[j]/255.0;
        
        Layer_setInputs(linput, x);
        Layer_getOutputs(loutput, y);
        int label = IdxFile_get1(labels_train, index);
        for (int j = 0; j < 10; j++) {
            y[j] = (j == label)? 1 : 0;
        }
        Layer_learnOutputs(loutput, y);
        etotal += Layer_getErrorTotal(loutput);
        if ((i % batch_size) == 0) {
            // Minibatch: update the network for every n samples.
            Layer_update(loutput, rate/batch_size);
        }
        if ((i % 1000) == 0) {
            fprintf(stderr, "i=%d, error=%.4f\n", i, etotal/1000);
            etotal = 0;
        }
    }

    IdxFile_destroy(images_train);
    IdxFile_destroy(labels_train);

    
    IdxFile* images_test = NULL;
    fp = fopen(argv[3], "rb");
    if (fp == NULL) return 111;
    images_test = IdxFile_read(fp);
    if (images_test == NULL) return 111;
    fclose(fp);

    IdxFile* labels_test = NULL;
    fp = fopen(argv[4], "rb");
    if (fp == NULL) return 111;
    labels_test = IdxFile_read(fp);
    if (labels_test == NULL) return 111;
    fclose(fp);

    fprintf(stderr, "testing...\n");
    int ntests = images_test->dims[0];
    int ncorrect = 0;
    for (int i = 0; i < ntests; i++) {
        uint8_t img[28*28];
        double x[28*28];
        double y[10];
        IdxFile_get3(images_test, i, img);
        for (int j = 0; j < 28*28; j++) {
            x[j] = img[j]/255.0;
        }
        Layer_setInputs(linput, x);
        Layer_getOutputs(loutput, y);
        int label = IdxFile_get1(labels_test, i);
        //Pick the most probable label.
        int mj = -1;
        for (int j = 0; j < 10; j++) {
            if (mj < 0 || y[mj] < y[j]) {
                mj = j;
            }
        }
        if (mj == label) 
            ncorrect++;
        if ((i % 1000) == 0) fprintf(stderr, "i=%d\n", i);
    }
    fprintf(stderr, "ntests=%d, ncorrect=%d\n", ntests, ncorrect);

    IdxFile_destroy(images_test);
    IdxFile_destroy(labels_test);

    Layer_destroy(linput);
    Layer_destroy(lconv1);
    Layer_destroy(lconv2);
    Layer_destroy(lfull1);
    Layer_destroy(lfull2);
    Layer_destroy(loutput);

    return 0;
}
