// dense_tanh.c

// Librería matemática de C.
// Se usa para expf(), que calcula e^x en float.
#include <math.h>



// =====================================================
// TANH FORWARD

// Esta función aplica la activación tanh a un arreglo.
// Entrada:
//   z -> arreglo de entrada
// Salida:
//   y -> arreglo donde se guardará tanh(z)
// total_size:
//   cantidad total de elementos
// Fórmula:
// tanh(x) = (e^x - e^-x) / (e^x + e^-x)
// =====================================================
void tanh_forward(
    float *z,
    float *y,
    int total_size
) {

    // Recorre todos los elementos del arreglo
    for (int i = 0; i < total_size; i++) {

        // Calcula e^(z[i])
        float exp_pos = expf(z[i]);

        // Calcula e^(-z[i])
        float exp_neg = expf(-z[i]);

        // Calcula tanh manualmente
        y[i] = (exp_pos - exp_neg) /
               (exp_pos + exp_neg);
    }
}



// =====================================================
// DENSE FORWARD
// Implementa una capa densa manual:
//
// z = x @ weights.T + bias

// x:
//   entrada de la red
//   shape conceptual:
//   (batch_size, in_features)
//
// weights:
//   pesos de la capa
//   shape conceptual:
//   (out_features, in_features)
//
// bias:
//   bias de cada neurona
//   shape:
//   (out_features)
//
// z:
//   salida de la capa
//   shape:
//   (batch_size, out_features)
// =====================================================
void dense_forward(
    float *x,
    float *weights,
    float *bias,
    float *z,
    int batch_size,
    int in_features,
    int out_features
) {

    // Recorre cada muestra del batch
    for (int i = 0; i < batch_size; i++) {

        // Recorre cada neurona de salida
        for (int j = 0; j < out_features; j++) {

            // Acumulador de la suma ponderada
            float suma = 0.0f;

            // Recorre cada feature de entrada
            for (int k = 0; k < in_features; k++) {

                // Multiplica:
                // entrada * peso
                //
                // x[i * in_features + k]
                // accede a:
                // fila i, columna k
                //
                // weights[j * in_features + k]
                // accede a:
                // neurona j, feature k
                //
                // Todo está linealizado en memoria.
                suma += x[i * in_features + k] *
                        weights[j * in_features + k];
            }

            // Agrega bias de la neurona
            suma += bias[j];

            // Guarda resultado en z
            //
            // z[i * out_features + j]
            // corresponde a:
            // fila i, neurona j
            z[i * out_features + j] = suma;
        }
    }
}



// =====================================================
// DENSE + TANH FORWARD

// Fusiona:
//
// 1. Capa densa
// 2. Activación tanh
//
// Hace:
// y = tanh(x @ weights.T + bias)
//
// =====================================================
void dense_tanh_forward(
    float *x,
    float *weights,
    float *bias,
    float *y,
    int batch_size,
    int in_features,
    int out_features
) {

    // Recorre cada muestra
    for (int i = 0; i < batch_size; i++) {

        // Recorre cada neurona
        for (int j = 0; j < out_features; j++) {

            // Acumulador del producto punto
            float suma = 0.0f;

            // Producto punto:
            // x · weights
            for (int k = 0; k < in_features; k++) {

                suma += x[i * in_features + k] *
                        weights[j * in_features + k];
            }

            // Agrega bias
            suma += bias[j];

            // =================================================
            // ACTIVACIÓN TANH
          

            // e^(suma)
            float exp_pos = expf(suma);

            // e^(-suma)
            float exp_neg = expf(-suma);

            // Aplica tanh y guarda salida final
            y[i * out_features + j] =
                (exp_pos - exp_neg) /
                (exp_pos + exp_neg);
        }
    }
}