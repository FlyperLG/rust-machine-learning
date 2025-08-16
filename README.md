# rust-machine-learning
This repository contains several Rust projects I used to get familiar with the Rust programming language. Each subdirectory (except for `rust_poly_net`) is a standalone crate that introduced basic Rust concepts and programming patterns:


## Project Overview
- **guessing_game**: A simple number guessing game that introduces user input, random number generation, and control flow in Rust.
- **minigrep**: A minimal implementation of the classic `grep` utility, showing how to handle command-line arguments, file I/O, and error handling.
- **rectangles**: Demonstrates Rust's enums, structs, and trait implementations by modeling geometric shapes and their properties.
- **tic_tac_toe**: A console-based tic-tac-toe game, illustrating more advanced struct usage, game logic, and user interaction.
- **rust_poly_net**: A custom machine learning framework that supports multiple numeric representations, including floats and posits, for neural network computations. This is the main focus of the repository and is designed for research into the effects of different number types on machine learning performance.

---

## How to Run the Projects
The first step is to ensure you have Rust and Cargo installed. You can follow the instructions on the [official Rust website](https://www.rust-lang.org/tools/install) to set up your environment.

The **guessing_game**, **minigrep**, **rectangles** and **tic_tac_toe** project can be run independently. To run a specific project, navigate to its directory and use Cargo:
```bash
cargo run
```

The rust_poly_net can be run using multiple commands, depending on the specific functionality you want to test or demonstrate. For example, to run the main application in release mode, you can use:
```bash
cargo run -r
```
This will compile the project in release mode for better performance and execute the main function. In the `main.rs` file, you can find multiple functions which allow different functionalitys.
- `performance_benchmark()`: This function benchmarks the performance of different numeric types used in the framework. It runs multiple iterations and collects accuracy metrics for each type.
- `train_basic_network()`: This function trains a basic neural network using the `f32` numeric type. It demonstrates how to set up a simple feedforward network and train it on a dataset. To modify the `hyperparameters` you need to change the `run_training_for_type` function inside the `lib.rs` file, which is called by `train_basic_network()`.

To run the timing benchmark, you can use:
```bash
cargo bench
```
This will execute the benchmarks defined in the `benches` directory, allowing you to compare the performance of different numeric types.


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

- **Pluggable number types**: Easily swap between `f32`, `f16`, and custom types like `Posit16_1` (posit).
- **Custom traits**: Abstracts over numeric operations to support extensibility.
- **Support for standard ML operations**: Uses crates like `ndarray` for efficient tensor operations.
- **Research-oriented**: Ideal for exploring the impact of numeric representation on neural network training and inference.

---