extern "C"{
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
void forward_convolution_layer(Layer* layer)
}