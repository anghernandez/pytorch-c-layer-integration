# =====================================================
# IMPORTACIONES
# =====================================================

# ctypes:
# permite llamar funciones escritas en C
# desde Python
import ctypes

# numpy:
# manejo de arreglos/matrices
# compatibles con memoria continua para C
import numpy as np

# os:
# manejo de rutas y directorios
import os

# torch:
# se usa aquí únicamente para generar
# pesos aleatorios reproducibles
import torch



# =====================================================
# FIJAR SEMILLA
# Hace que los pesos aleatorios sean siempre iguales.


torch.manual_seed(0)



# =====================================================
# OBTENER RUTA DE LA LIBRERÍA .SO

# Obtiene la ruta absoluta del archivo actual
# Ejemplo:
# /home/gabrielqg/Descargas/Asistencia FPGA
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construye:
# /home/.../libdense.so
# Esto evita problemas de rutas relativas.
lib_path = os.path.join(base_dir, "libdense.so")



# =====================================================
# CARGAR LIBRERÍA C

# Carga la librería compartida (.so)
# Equivalente a:
# "importar" el código C dentro de Python
lib = ctypes.CDLL(lib_path)



# =====================================================
# DEFINIR TIPOS DE ARGUMENTOS

# Aquí se le dice a ctypes:
# "la función en C espera estos tipos"
# Firma en C:
# void dense_tanh_forward(
#     float *x,
#     float *weights,
#     float *bias,
#     float *y,
#     int batch_size,
#     int in_features,
#     int out_features
# )
# ctypes necesita esto para:
# - convertir tipos correctamente
# - evitar errores de memoria
# - saber cómo pasar punteros
# =====================================================
lib.dense_tanh_forward.argtypes = [

    # float *x
    ctypes.POINTER(ctypes.c_float),

    # float *weights
    ctypes.POINTER(ctypes.c_float),

    # float *bias
    ctypes.POINTER(ctypes.c_float),

    # float *y
    ctypes.POINTER(ctypes.c_float),

    # int batch_size
    ctypes.c_int,

    # int in_features
    ctypes.c_int,

    # int out_features
    ctypes.c_int
]



# =====================================================
# INPUT DE LA RED
# Shape:
# (2,3)
# 2 muestras
# 3 features
# dtype=float32 es IMPORTANTE
# porque en C usamos:
# float

x = np.array([
    [0.5, -1.0, 2.0],
    [1.0,  0.0, -0.5]
], dtype=np.float32)



# =====================================================
# PESOS

# torch.randn(2,3):
# crea matriz aleatoria:
# shape:
# (2,3)
# 2 neuronas de salida
# 3 features de entrada
# *0.01:
# inicialización pequeña típica en DL
# .numpy():
# convierte tensor -> numpy
# astype(np.float32):
# asegura compatibilidad con C
# =====================================================
weights = (
    torch.randn(2, 3) * 0.01
).numpy().astype(np.float32)



# =====================================================
# BIAS

#
# Un bias por neurona de salida.
#
# shape:
# (2,)
# =====================================================
bias = torch.zeros(2).numpy().astype(np.float32)



# =====================================================
# SALIDA
#
# Reserva memoria para la salida.
#
# shape:
# (2,2)
#
# batch_size = 2
# out_features = 2
#
# Inicialmente todo en cero.
# =====================================================
y = np.zeros((2, 2), dtype=np.float32)



# =====================================================
# LLAMADA A FUNCIÓN C
#
# Aquí Python llama la función escrita en C.
#
# data_as(...)
# obtiene el puntero REAL en memoria.
#
# Esto literalmente le pasa a C:
#
# dirección RAM del arreglo numpy
#
# Muy parecido a cómo PyTorch trabaja
# internamente con tensores.
# =====================================================
lib.dense_tanh_forward(

    # x
    x.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    ),
    # weights
    weights.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    ),
    # bias
    bias.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    ),
    # y
    y.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    ),
    # batch_size
    2,
    # in_features
    3,
    # out_features
    2
)



# =====================================================
# IMPRIMIR RESULTADOS


print("Weights:")
print(weights)

print("\nBias:")
print(bias)

print("\nSalida C:")
print(y)