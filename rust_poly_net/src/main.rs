use std::{
    fmt::Display,
    ops::{AddAssign, Neg, Sub, SubAssign},
};

use ndarray::{Array, Array2, LinalgScalar, ScalarOperand, arr2};
use num_traits::One;
use rand_distr::{Distribution, StandardNormal};
use rust_poly_net::{Exponential, Float64, FromNumber};

fn main() {
    let x = arr2(&[
        // a
        [
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 1. },
        ],
        // b
        [
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 0. },
        ],
        // c
        [
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 1. },
            Float64 { value: 0. },
        ],
    ]);
    let labels = arr2(&[
        [
            Float64 { value: 1. },
            Float64 { value: 0. },
            Float64 { value: 0. },
        ],
        [
            Float64 { value: 0. },
            Float64 { value: 1. },
            Float64 { value: 0. },
        ],
        [
            Float64 { value: 0. },
            Float64 { value: 0. },
            Float64 { value: 1. },
        ],
    ]);
    let mut w1 = generate_weights(30, 5);
    let mut w2 = generate_weights(5, 3);

    train(x, labels, &mut w1, &mut w2, 0.1, 10000000);
}

fn sigmoid<T>(x: &Array2<T>) -> Array2<T>
where
    T: FromNumber + LinalgScalar + Exponential + Neg<Output = T> + Ord,
{
    T::from_f64(-709.0);
    x.mapv(|v| {
        let v = v.max(T::from_f64(-709.0)).min(T::from_f64(709.0));
        T::one() / (T::one() + (-v).exp())
    })
}

fn sigmoid_derivative<T>(x: &Array2<T>) -> Array2<T>
where
    T: One + Sub<Output = T> + Copy,
{
    x.mapv(|v| v * (T::one() - v))
}

fn forward<T>(x: &Array2<T>, w1: &Array2<T>, w2: &Array2<T>) -> (Array2<T>, Array2<T>)
where
    T: LinalgScalar + FromNumber + Exponential + Neg<Output = T> + One + Ord,
{
    let z1 = x.dot(w1);
    let a1 = sigmoid(&z1);

    let z2 = a1.dot(w2);
    let a2 = sigmoid(&z2);

    (a1, a2)
}

fn generate_weights<T>(x: usize, y: usize) -> Array2<T>
where
    T: FromNumber,
{
    let mut rng = rand::rng();
    let mut list = Vec::with_capacity(x * y);
    for _ in 0..(x * y) {
        let value: f64 = StandardNormal.sample(&mut rng);
        list.push(T::from_f64(0.1 * value));
    }
    Array::from_shape_vec((x, y), list).unwrap()
}

fn loss<T>(prediction: &Array2<T>, labels: &Array2<T>) -> T
where
    T: LinalgScalar + FromNumber + Exponential + Ord,
{
    let diff = prediction - labels;
    let loss = diff.mapv(|v| v.powi(2)).sum() / T::from_f64(labels.len() as f64);
    loss
}

fn back_prop<T>(
    x: &Array2<T>,
    a1: &Array2<T>,
    a2: &Array2<T>,
    labels: &Array2<T>,
    w1: &mut Array2<T>,
    w2: &mut Array2<T>,
    lr: f64,
) where
    T: LinalgScalar + FromNumber + SubAssign + AddAssign + ScalarOperand,
{
    let d2 = a2 - labels;
    let d2_sigmoid = &d2 * &sigmoid_derivative(a2);
    let d1_sigmoid = d2_sigmoid.dot(&w2.t()) * sigmoid_derivative(a1);

    let w2_update = a1.t().dot(&d2_sigmoid);
    let w1_update = x.t().dot(&d1_sigmoid);

    *w2 -= &(&w2_update * T::from_f64(lr));
    *w1 -= &(&w1_update * T::from_f64(lr));
}

fn train<T>(
    x: Array2<T>,
    labels: Array2<T>,
    w1: &mut Array2<T>,
    w2: &mut Array2<T>,
    lr: f64,
    epochs: i32,
) where
    T: LinalgScalar
        + FromNumber
        + Exponential
        + Neg<Output = T>
        + One
        + Ord
        + Display
        + SubAssign
        + AddAssign
        + ScalarOperand,
{
    for epoch in 0..epochs {
        let (a1, a2) = forward(&x, &w1, &w2);

        let l: T = loss(&a2, &labels);

        back_prop(&x, &a1, &a2, &labels, w1, w2, lr);

        println!("epochs: {} ==== loss: {}", epoch + 1, l);
    }
}
