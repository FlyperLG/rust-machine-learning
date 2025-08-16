use std::{collections::HashMap, fs::File, io::Write, sync::mpsc, thread};

use rust_poly_net::{
    MlScalar,
    number_representations::{
        float::f16::MLf16,
        posit::{posit16_1::Posit16_1, posit16_2::Posit16_2, posit32_2::Posit32_2},
        softposit::{softposit16_1::Softposit16_1, softposit32_2::Softposit32_2},
    },
    run_training_for_type,
};
use serde::Serialize;

fn main() {
    train_basic_network();
    // performance_benchmark();
}

fn train_basic_network() {
    // This function is a placeholder for the actual training logic.
    // It should return the accuracy of the trained model.
    let result = run_training_for_type::<f32>();
    println!("Training completed with accuracy: {:.2}%", result * 100.0);
}

#[derive(Serialize)]
struct TypeResult {
    type_name: String,
    average_accuracy: f64,
    accuracies: Vec<f64>,
}

fn performance_benchmark() {
    let iterations = 5;

    // Create a multi-producer, single-consumer channel for results.
    let (sender, receiver) = mpsc::channel();

    println!("Spawning {} parallel training jobs...", 7);

    // Use scoped threads to ensure all threads complete before we proceed.
    thread::scope(|s| {
        // Spawn a new thread for each data type.
        s.spawn({
            let sender = sender.clone();
            move || run_and_send_results::<f32>("f32", iterations, sender)
        });
        s.spawn({
            let sender = sender.clone();
            move || run_and_send_results::<MLf16>("MLf16", iterations, sender)
        });
        s.spawn({
            let sender = sender.clone();
            move || run_and_send_results::<Posit16_1>("Posit16_1", iterations, sender)
        });
        s.spawn({
            let sender = sender.clone();
            move || run_and_send_results::<Posit16_2>("Posit16_2", iterations, sender)
        });
        s.spawn({
            let sender = sender.clone();
            move || run_and_send_results::<Posit32_2>("Posit32_2", iterations, sender)
        });
        s.spawn({
            let sender = sender.clone();
            move || run_and_send_results::<Softposit16_1>("Softposit16_1", iterations, sender)
        });
        s.spawn({
            let sender = sender.clone();
            move || run_and_send_results::<Softposit32_2>("Softposit32_2", iterations, sender)
        });
    }); // The scope ends here, all threads are guaranteed to be finished.

    println!("\nAll training jobs completed. Collecting results...");
    drop(sender);

    // Collect all the results from the channel into a HashMap.
    // The `receiver.iter()` will block until all senders are dropped, which happens
    // when the threads finish. This is a clean way to gather all results.
    let collected_accuracies: HashMap<String, Vec<f64>> = receiver.iter().collect();

    // The rest of the logic is identical to the previous version.
    let mut final_results: Vec<TypeResult> = Vec::new();
    for (type_name, accuracies) in collected_accuracies {
        let sum: f64 = accuracies.iter().sum();
        let average_accuracy = if accuracies.is_empty() {
            0.0
        } else {
            sum / accuracies.len() as f64
        };

        final_results.push(TypeResult {
            type_name,
            average_accuracy,
            accuracies,
        });
    }

    let json_results =
        serde_json::to_string_pretty(&final_results).expect("Failed to serialize results to JSON.");

    let mut file = File::create("training_results_parallel.json").expect("Failed to create file.");
    file.write_all(json_results.as_bytes())
        .expect("Failed to write to file.");

    println!("Parallel results have been written to training_results_parallel.json");
}

fn run_and_send_results<T: MlScalar + Send + 'static>(
    type_name: &'static str,
    iterations: u32,
    sender: mpsc::Sender<(String, Vec<f64>)>,
) {
    let mut accuracies = Vec::new();
    for i in 0..iterations {
        // You could print here to see interleaved execution, but it might be messy.
        // println!("Running iter {} for {}", i + 1, type_name);
        accuracies.push(run_training_for_type::<T>());
    }
    // Send the final collected accuracies back to the main thread.
    sender.send((type_name.to_string(), accuracies)).unwrap();
}
