
// PYBIND11
// pybind11 permite crear módulos Python
// escritos en C/C++.
// Con esto podemos hacer:
// import dense_pybind
// desde Python.
// =====================================================

// Librería principal de pybind11
#include <pybind11/pybind11.h>

// Soporte para arreglos numpy
#include <pybind11/numpy.h>



// ENLAZAR C CON C++

// dense.h fue escrito en C puro.
// extern "C":
// evita el "name mangling" de C++.
// Sin esto, C++ cambia internamente el nombre
// de las funciones y luego Python no encuentra:
// dense_tanh_forward
// Este bloque le dice a C++:
// "estas funciones vienen de C"
// =====================================================
extern "C" {

    #include "dense.h"
}




namespace py = pybind11;


// WRAPPER PYTHON -> C

// Esta función actúa como puente.
// Python llama:
// dense_tanh_forward(x, weights, bias)
// y esta función:
// 1. recibe numpy arrays
// 2. obtiene punteros reales
// 3. llama la función C
// 4. devuelve numpy array
// ===================================================
py::array_t<float> dense_tanh_pybind(

    // Array numpy de entrada
    py::array_t<float> x,

    // Pesos
    py::array_t<float> weights,

    // Bias
    py::array_t<float> bias
) {

    // =================================================
    // REQUEST BUFFER
    // request() obtiene información interna del array:
    // - shape
    // - strides
    // - puntero en memoria
    //
    // Similar a acceder al tensor interno.
    // =================================================
    auto x_buf = x.request();
    auto w_buf = weights.request();
    auto b_buf = bias.request();



    // =================================================
    // EXTRAER DIMENSIONES
    // x:
    // (batch_size, in_features)
    //
    // weights:
    // (out_features, in_features)
    // =================================================

    // Cantidad de muestras
    int batch_size = x_buf.shape[0];

    // Cantidad de features
    int in_features = x_buf.shape[1];

    // Cantidad de neuronas de salida
    int out_features = w_buf.shape[0];



    // =================================================
    // CREAR ARRAY DE SALIDA
    // Shape:
    // (batch_size, out_features)
    // py::array_t<float>
    // crea un numpy array desde C++.
    // =================================================
    py::array_t<float> y({
        batch_size,
        out_features
    });



    // Obtiene acceso interno al buffer
    auto y_buf = y.request();



    // =================================================
    // LLAMAR FUNCIÓN C
    // Aquí ocurre
    // static_cast<float*>:
    // convierte el puntero genérico
    // a float*
    // ptr:
    // dirección REAL en memoria
    // Estamos pasando:
    // numpy -> C 
    //
    // sin copiar datos manualmente.
    // =================================================
    dense_tanh_forward(

        // x
        static_cast<float *>(x_buf.ptr),

        // weights
        static_cast<float *>(w_buf.ptr),

        // bias
        static_cast<float *>(b_buf.ptr),

        // salida y
        static_cast<float *>(y_buf.ptr),

        // dimensiones
        batch_size,
        in_features,
        out_features
    );



    // =================================================
    // RETORNAR RESULTADO
    //
    // Devuelve un numpy array a Python.
    // =================================================
    return y;
}



// =====================================================
// CREAR MÓDULO PYTHON
// Esto crea:
// import dense_pybind
// dense_pybind:
// nombre del módulo
// m:
// objeto módulo Python
// =====================================================
PYBIND11_MODULE(dense_pybind, m) {

    // =================================================
    // EXPORTAR FUNCIÓN
    // Python verá:
    // dense_pybind.dense_tanh_forward(...)
    // que internamente llama:
    // dense_tanh_pybind(...)
    // y luego C puro.
    // =================================================
    m.def(
        "dense_tanh_forward",
        &dense_tanh_pybind
    );
}