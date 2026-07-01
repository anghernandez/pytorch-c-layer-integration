# PyTorch C Layer Integration

Example showing how to integrate a native C implementation with Python using:

- ctypes
- pybind11

The implemented layer is:

- Dense
- Hyperbolic Tangent (tanh)

The forward pass is implemented in C, while the backward pass is handled by PyTorch Autograd.

---

## Project Structure

```
.
├── src/
│   ├── dense.c
│   ├── dense.h
│   └── dense_pybind.cpp
├── python/
│   ├── dense_ctypes.py
│   └── test_pybind.py
```

---

## Requirements

- Python 3.14
- GCC / G++
- pybind11
- NumPy
- PyTorch

---

## Installation

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install Python packages:

```bash
pip install -r requirements.txt
```

---

## Build

Compile the C source:

```bash
gcc -c src/dense.c -o dense.o -fPIC
```

Compile the pybind11 module:

```bash
g++ -O3 -Wall -shared -std=c++11 -fPIC \
-I/usr/include/python3.14 \
-I"$PWD/venv/lib/python3.14/site-packages/pybind11/include" \
src/dense_pybind.cpp dense.o \
-o dense_pybind$(python3-config --extension-suffix)
```

Compile the shared library for ctypes:

```bash
gcc -shared dense.o -o libdense.so -lm
```

---

## Run ctypes example

```bash
python python/dense_ctypes.py
```

---

## Run pybind11 example

```bash
python python/test_pybind.py
```

---

## Architecture

```
Python

↓

ctypes / pybind11

↓

Native C

↓

Dense + tanh

↓

Python
```

---

## Future Work

Replace the native C implementation with an FPGA accelerator while preserving the same Python interface.
