mod dataloader;
use crate::dataloader::MnistDataloader;

use ndarray::{Array, Array2, ArrayView1, ArrayView2, s};
use num_traits::{One, Zero};
use rand_distr::{Distribution, StandardNormal};
use rust_poly_net::{Float64, MlScalar};

fn main() {
    let mut w1 = generate_weights(28 * 28, 20);
    let mut w2 = generate_weights(20, 10);

    let mut train_dataloader = MnistDataloader::<f64>::new("./data/mnist");
    train_dataloader.load_data().unwrap();
    let train_data = train_dataloader.train_data;
    let train_labels = one_hot_encode(&train_dataloader.train_labels.view(), 10);
    train(train_data, train_labels, &mut w1, &mut w2, 0.001, 10, 64);
}

fn one_hot_encode<T>(labels: &ArrayView1<u8>, num_classes: usize) -> Array2<T>
where
    T: Clone + Zero + One,
{
    let batch_size = labels.len();
    let mut one_hot = Array2::zeros((batch_size, num_classes));

    for (i, &label_index) in labels.iter().enumerate() {
        assert!(
            (label_index as usize) < num_classes,
            "Label index {} is out of bounds for {} classes",
            label_index,
            num_classes
        );
        one_hot[[i, label_index as usize]] = T::one();
    }
    one_hot
}

fn sigmoid<T: MlScalar>(x: &Array2<T>) -> Array2<T> {
    x.mapv(|v| {
        let v = v.max(T::from(-709.0).unwrap()).min(T::from(709.0).unwrap());
        T::one() / (T::one() + (-v).exp())
    })
}

fn sigmoid_derivative<T: MlScalar>(x: &Array2<T>) -> Array2<T> {
    x.mapv(|v| v * (T::one() - v))
}

fn forward<T: MlScalar>(
    x: &ArrayView2<T>,
    w1: &Array2<T>,
    w2: &Array2<T>,
) -> (Array2<T>, Array2<T>) {
    let z1 = x.dot(w1);
    let a1 = sigmoid(&z1);

    let z2 = a1.dot(w2);
    let a2 = sigmoid(&z2);

    (a1, a2)
}

fn generate_weights<T: MlScalar>(x: usize, y: usize) -> Array2<T> {
    let mut rng = rand::rng();
    let mut list = Vec::with_capacity(x * y);
    for _ in 0..(x * y) {
        let value: f64 = StandardNormal.sample(&mut rng);
        list.push(T::from(0.1 * value).unwrap());
    }
    Array::from_shape_vec((x, y), list).unwrap()
}

fn loss<T: MlScalar>(prediction: &Array2<T>, labels: &ArrayView2<T>) -> T {
    let diff = prediction - labels;
    let num_samples = T::from(labels.shape()[0]).unwrap();
    let loss = diff.mapv(|v| v.powi(2)).sum() / num_samples;
    loss
}

fn back_prop<T: MlScalar>(
    x: &ArrayView2<T>,
    a1: &Array2<T>,
    a2: &Array2<T>,
    labels: &ArrayView2<T>,
    w1: &mut Array2<T>,
    w2: &mut Array2<T>,
    lr: f64,
) {
    let d2 = a2 - labels;
    let d2_sigmoid = &d2 * &sigmoid_derivative(a2);
    let d1_sigmoid = d2_sigmoid.dot(&w2.t()) * sigmoid_derivative(a1);

    let w2_update = a1.t().dot(&d2_sigmoid);
    let w1_update = x.t().dot(&d1_sigmoid);

    *w2 -= &(&w2_update * T::from(lr).unwrap());
    *w1 -= &(&w1_update * T::from(lr).unwrap());
}

fn train<T: MlScalar>(
    x: Array2<T>,
    labels: Array2<T>,
    w1: &mut Array2<T>,
    w2: &mut Array2<T>,
    lr: f64,
    epochs: i32,
    batch_size: usize,
) {
    let num_samples = x.shape()[0];

    for epoch in 0..epochs {
        let mut epoch_loss = T::zero();
        let mut num_batches = 0;

        for i in (0..num_samples).step_by(batch_size) {
            let end = (i + batch_size).min(num_samples);
            let x_batch = x.slice(s![i..end, ..]);
            let labels_batch = labels.slice(s![i..end, ..]);

            let (a1, a2) = forward(&x_batch, &w1, &w2);

            let l: T = loss(&a2, &labels_batch);

            back_prop(&x_batch, &a1, &a2, &labels_batch, w1, w2, lr);

            epoch_loss += l;
            num_batches += 1;
        }

        println!(
            "Epoch: {}/{} ==== Average loss: {}",
            epoch + 1,
            epochs,
            epoch_loss / T::from(num_batches).unwrap()
        );
    }
}
