use ndarray::{Array, Array2, arr2};
use rand_distr::{Distribution, StandardNormal};
use rust_poly_net::Float64;

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

fn sigmoid(x: &Array2<Float64>) -> Array2<Float64> {
    x.mapv(|v| {
        let v = v
            .max(Float64 { value: -709.0 })
            .min(Float64 { value: 709.0 });
        Float64 { value: 1.0 } / (Float64 { value: 1.0 } + (-v).exp())
    })
}

fn sigmoid_derivative(x: &Array2<Float64>) -> Array2<Float64> {
    x.mapv(|v| v * (Float64 { value: 1.0 } - v))
}

fn forward(
    x: &Array2<Float64>,
    w1: &Array2<Float64>,
    w2: &Array2<Float64>,
) -> (Array2<Float64>, Array2<Float64>) {
    let z1 = x.dot(w1);
    let a1 = sigmoid(&z1);

    let z2 = a1.dot(w2);
    let a2 = sigmoid(&z2);

    (a1, a2)
}

fn generate_weights(x: usize, y: usize) -> Array2<Float64> {
    let mut rng = rand::rng();
    let mut list = Vec::with_capacity(x * y);
    for _ in 0..(x * y) {
        let value: f64 = StandardNormal.sample(&mut rng);
        list.push(Float64 { value: 0.1 * value });
    }
    Array::from_shape_vec((x, y), list).unwrap()
}

fn loss(prediction: &Array2<Float64>, labels: &Array2<Float64>) -> f64 {
    let diff = prediction - labels;
    let loss = diff.mapv(|v| v.powi(2)).sum()
        / Float64 {
            value: labels.len() as f64,
        };
    loss.value
}

fn back_prop(
    x: &Array2<Float64>,
    a1: &Array2<Float64>,
    a2: &Array2<Float64>,
    labels: &Array2<Float64>,
    w1: &mut Array2<Float64>,
    w2: &mut Array2<Float64>,
    lr: f64,
) {
    let d2 = a2 - labels;
    let d2_sigmoid = &d2 * &sigmoid_derivative(a2);
    let d1_sigmoid = d2_sigmoid.dot(&w2.t()) * sigmoid_derivative(a1);

    let w2_update = a1.t().dot(&d2_sigmoid);
    let w1_update = x.t().dot(&d1_sigmoid);

    *w2 -= &(&w2_update * Float64 { value: lr });
    *w1 -= &(&w1_update * Float64 { value: lr });
}

fn train(
    x: Array2<Float64>,
    labels: Array2<Float64>,
    w1: &mut Array2<Float64>,
    w2: &mut Array2<Float64>,
    lr: f64,
    epochs: i32,
) {
    for epoch in 0..epochs {
        let (a1, a2) = forward(&x, &w1, &w2);

        let l: f64 = loss(&a2, &labels);

        back_prop(&x, &a1, &a2, &labels, w1, w2, lr);

        println!("epochs: {} ==== loss: {}", epoch + 1, l);
    }
}
