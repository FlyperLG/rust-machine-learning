# rust-machine-learning

## Project Overview

This repository contains several Rust projects I used to get familiar with the Rust programming language. Each subdirectory (except for `rust_poly_net`) is a standalone crate that introduced basic Rust concepts and programming patterns:

- **guessing_game**: A simple number guessing game that introduces user input, random number generation, and control flow in Rust.
- **minigrep**: A minimal implementation of the classic `grep` utility, showing how to handle command-line arguments, file I/O, and error handling.
- **rectangles**: Demonstrates Rust's enums, structs, and trait implementations by modeling geometric shapes and their properties.
- **tic_tac_toe**: A console-based tic-tac-toe game, illustrating more advanced struct usage, game logic, and user interaction.

---

## rust_poly_net: Custom Machine Learning Framework

The `rust_poly_net` crate is the centerpiece of this repository. It is a custom machine learning framework built in Rust, with a unique focus: supporting multiple number representations for neural network computations.

### Why Not Just Use Floats?

Traditional machine learning frameworks rely heavily on floating-point numbers (floats) for representing and processing data. While floats are widely used, they have several disadvantages:
- **Precision limitations**: Floats can introduce rounding errors and loss of accuracy, especially in deep networks or when working with very large or very small numbers.
- **Non-uniform error**: The distribution of representable numbers is not uniform, which can affect the stability and reliability of some algorithms.
- **Hardware dependency**: Floating-point behavior can vary across different hardware platforms.

### Exploring Posits and Other Representations

Posits are an alternative to floats, offering:
- Higher accuracy for the same number of bits,
- More uniform error distribution,
- Better dynamic range.

The `rust_poly_net` framework is designed to make it easy to switch between floats, posits, and potentially other numeric types. This allows for in-depth analysis and experimentation with how different number representations affect machine learning performance, accuracy, and stability.

### Framework Features

- **Pluggable number types**: Easily swap between `f32`, `f64`, and custom types like `P32` (posit).
- **Custom traits**: Abstracts over numeric operations to support extensibility.
- **Support for standard ML operations**: Uses crates like `ndarray` for efficient tensor operations.
- **Research-oriented**: Ideal for exploring the impact of numeric representation on neural network training and inference.

---

### rust_poly_net File Overview

- **Cargo.toml**  
  Project manifest file specifying dependencies and metadata for the crate.

- **src/lib.rs**  
  Core library code. Defines numeric traits and abstractions to support multiple number representations (floats, posits, etc.) for machine learning operations.

- **src/architecture.rs**  
  Contains the neural network architecture, including the definition of layers and the main `Network` struct, as well as forward propagation logic.

- **src/dataloader.rs**  
  Implements a data loader for the [MNIST dataset](https://raw.githubusercontent.com/fgnt/mnist/master), handling reading and preprocessing of image and label data.

- **src/main.rs**  
  Example executable that demonstrates how to use the framework: loads MNIST data, builds a network, and trains a model.

- **docs/future_improvements.md**  
  Notes and suggestions for future enhancements, including model accuracy, performance, and code structure improvements.

- **docs/performance_through_simd.md**  
  Discussion and tips on optimizing performance, especially when using custom number types, including SIMD strategies.