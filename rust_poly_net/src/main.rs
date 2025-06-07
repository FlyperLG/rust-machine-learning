use ndarray::{Array, Array2, arr2};
use rand_distr::{Distribution, StandardNormal};

fn main() {
    let x = arr2(&[
        // a
        [
            0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.,
            0., 1., 1., 0., 0., 0., 0., 1.,
        ],
        // b
        [
            0., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 1., 1., 0., 0., 1., 0., 0.,
            1., 0., 0., 1., 1., 1., 1., 0.,
        ],
        // c
        [
            0., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 1., 1., 1., 1., 0.,
        ],
    ]);
    let labels = arr2(&[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
    let mut w1 = generate_weights(30, 5);
    let mut w2 = generate_weights(5, 3);

    train(x, labels, &mut w1, &mut w2, 0.1, 1000);
}

fn sigmoid(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| {
        let v = v.max(-709.0).min(709.0);
        1.0 / (1.0 + (-v).exp())
    })
}

fn sigmoid_derivative(x: &Array2<f64>) -> Array2<f64> {
    x * &(1.0 - x)
}

fn forward(x: &Array2<f64>, w1: &Array2<f64>, w2: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let z1 = x.dot(w1);
    let a1 = sigmoid(&z1);

    let z2 = a1.dot(w2);
    let a2 = sigmoid(&z2);

    (a1, a2)
}

fn generate_weights(x: usize, y: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let mut list = Vec::with_capacity(x * y);
    for _ in 0..(x * y) {
        let value: f64 = StandardNormal.sample(&mut rng);
        list.push(0.1 * value);
    }
    Array::from_shape_vec((x, y), list).unwrap()
}

fn loss(prediction: &Array2<f64>, labels: &Array2<f64>) -> f64 {
    let diff = prediction - labels;
    diff.mapv(|v| v.powi(2)).sum() / labels.len() as f64
}

fn back_prop(
    x: &Array2<f64>,
    a1: &Array2<f64>,
    a2: &Array2<f64>,
    labels: &Array2<f64>,
    w1: &mut Array2<f64>,
    w2: &mut Array2<f64>,
    lr: f64,
) {
    let d2 = a2 - labels;
    let d2_sigmoid = &d2 * &sigmoid_derivative(a2);
    let d1_sigmoid = d2_sigmoid.dot(&w2.t()) * sigmoid_derivative(a1);

    let w2_update = a1.t().dot(&d2_sigmoid);
    let w1_update = x.t().dot(&d1_sigmoid);

    *w2 -= &(lr * &w2_update);
    *w1 -= &(lr * &w1_update);
}

fn train(
    x: Array2<f64>,
    labels: Array2<f64>,
    w1: &mut Array2<f64>,
    w2: &mut Array2<f64>,
    lr: f64,
    epochs: i32,
) {
    for epoch in 0..epochs {
        let (a1, a2) = forward(&x, &w1, &w2);

        let l = loss(&a2, &labels);

        back_prop(&x, &a1, &a2, &labels, w1, w2, lr);

        println!("epochs: {} ==== loss: {}", epoch + 1, l);
    }
}
