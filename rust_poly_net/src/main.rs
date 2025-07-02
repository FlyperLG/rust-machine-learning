mod architecture;
mod dataloader;

use crate::{
    architecture::{LinearLayer, Network, Sigmoid},
    dataloader::MnistDataloader,
};

use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use num_traits::{One, Zero};
use rand_distr::{Distribution, StandardNormal};
use rust_poly_net::{Float64, MlPosit, MlScalar};
use std::time::Instant;

fn main() {
    let mut train_dataloader = MnistDataloader::<f64>::new("./data/mnist");
    train_dataloader.load_data().unwrap();
    let train_data = train_dataloader.train_data;
    let train_labels = one_hot_encode(&train_dataloader.train_labels.view(), 10);

    let network: Network<f64> = Network::new(vec![
        Box::new(LinearLayer::new(28 * 28, 20, Box::new(Sigmoid::new()))),
        Box::new(LinearLayer::new(20, 10, Box::new(Sigmoid::new()))),
    ]);

    train_model(train_data, train_labels, network, 0.001, 20, 64);
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

fn loss<T: MlScalar>(prediction: &Array2<T>, labels: &ArrayView2<T>) -> T {
    let diff = prediction - labels;
    let num_samples = T::from(labels.shape()[0]).unwrap();
    let loss = diff.mapv(|v| v.powi(2)).sum() / num_samples;
    loss
}

fn train_model<T: MlScalar>(
    x: Array2<T>,
    labels: Array2<T>,
    mut model: Network<T>,
    lr: f64,
    epochs: i32,
    batch_size: usize,
) {
    let num_samples = x.shape()[0];

    for epoch in 0..epochs {
        let mut epoch_loss = T::zero();
        let mut num_batches = 0;
        println!("Starting epoch {}/{} now.", epoch + 1, epochs);
        let epoch_timer = Instant::now();
        for i in (0..num_samples).step_by(batch_size) {
            let end = (i + batch_size).min(num_samples);
            let x_batch = x.slice(s![i..end, ..]);
            let labels_batch = labels.slice(s![i..end, ..]);

            let (activations, _) = model.forward(&x_batch);

            let l: T = loss(&activations[activations.len() - 1], &labels_batch);

            model.backward(&activations, &labels_batch, T::from(lr).unwrap());

            epoch_loss += l;
            num_batches += 1;
        }

        let duration = epoch_timer.elapsed();
        println!(
            "Epoch: {}/{} ==== Average loss: {} ==== Time taken: {:?}",
            epoch + 1,
            epochs,
            epoch_loss / T::from(num_batches).unwrap(),
            duration
        );
    }
}
