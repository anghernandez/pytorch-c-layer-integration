#ifndef DENSE_H
#define DENSE_H

void dense_tanh_forward(
    float *x,
    float *weights,
    float *bias,
    float *y,
    int batch_size,
    int in_features,
    int out_features
);

#endif