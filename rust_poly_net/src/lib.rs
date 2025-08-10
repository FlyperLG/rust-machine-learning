pub mod architecture;
pub mod dataloader;
pub mod number_representations;

use std::time::Instant;

use ndarray::{Array2, ArrayView1, ArrayView2, s};
use num_traits::{One, Zero};
pub use number_representations::core::MlScalar;

use crate::architecture::{HardSigmoid, LinearLayer, Network};
use crate::dataloader::MnistDataloader;

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
    x_test: Array2<T>,
    labels_test: Array2<T>,
    mut model: Network<T>,
    lr: f64,
    epochs: i32,
    batch_size: usize,
) -> f64 {
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

    let (final_activations, _) = model.forward(&x_test.view());
    let predictions = final_activations.last().unwrap();

    let mut correct_predictions = 0;
    for (pred_row, true_label_row) in predictions.rows().into_iter().zip(labels_test.rows()) {
        // Find the index of the max value in the prediction row. This is the predicted class label.
        let predicted_label = pred_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap();

        // In the one-hot encoded true label row, find the index of the '1'.
        // This gives the actual class label.
        let true_label = true_label_row
            .iter()
            .position(|&v| v == T::one())
            .expect("Invalid one-hot label: no '1' found in row.");

        if predicted_label == true_label {
            correct_predictions += 1;
        }
    }

    let accuracy = correct_predictions as f64 / labels_test.nrows() as f64;
    println!(
        "Final accuracy for type {}: {:.2}%",
        std::any::type_name::<T>(),
        accuracy * 100.0
    );
    accuracy
}

pub fn run_training_for_type<T: MlScalar>() -> f64 {
    // For benchmarking, you might want to use fewer samples/epochs
    // to keep the benchmark runtime reasonable.
    const NUM_TRAIN_SAMPLES: Option<usize> = None;
    const NUM_TEST_SAMPLES: Option<usize> = None;
    const EPOCHS: i32 = 20; // Keep this low for benchmarking

    // --- Load Data ---
    let mut dataloader =
        MnistDataloader::<T>::new("./data/mnist", NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES);
    dataloader.load_data().unwrap();
    let train_data = dataloader.train_data;
    let train_labels_one_hot = one_hot_encode(&dataloader.train_labels.view(), 10);
    let test_data = dataloader.test_data; // Dataloader uses one field for both
    let test_labels_one_hot = one_hot_encode(&dataloader.test_labels.view(), 10); // These are the raw u8 labels

    // --- Create Network ---
    let network: Network<T> = Network::new(vec![
        Box::new(LinearLayer::new(28 * 28, 20, Box::new(HardSigmoid::new()))),
        Box::new(LinearLayer::new(20, 10, Box::new(HardSigmoid::new()))),
    ]);

    // --- Train and return accuracy ---
    train_model(
        train_data,
        train_labels_one_hot,
        test_data,
        test_labels_one_hot,
        network,
        0.001,
        EPOCHS,
        64,
    )
}
