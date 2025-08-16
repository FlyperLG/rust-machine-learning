use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use rust_poly_net::number_representations::{
    float::f16::MLf16,
    posit::{posit16_1::Posit16_1, posit16_2::Posit16_2, posit32_2::Posit32_2},
    softposit::{softposit16_1::Softposit16_1, softposit32_2::Softposit32_2},
};
use rust_poly_net::run_training_for_type;

macro_rules! create_benchmark_for_type {
    ($c:expr, $type_name:ident, $type:ty) => {
        let mut group = $c.benchmark_group(stringify!($type_name));

        // You can adjust sample size for longer-running benchmarks
        group.sample_size(25);

        group.bench_function("train_mnist", |b| {
            b.iter(|| {
                let accuracy = run_training_for_type::<$type>();
                // black_box prevents the compiler from optimizing away the function call
                black_box(accuracy);
            })
        });
        group.finish();
    };
}

fn benchmark_training_types(c: &mut Criterion) {
    create_benchmark_for_type!(c, f32, f32);
    create_benchmark_for_type!(c, F16, MLf16);
    create_benchmark_for_type!(c, Posit16_1, Posit16_1);
    create_benchmark_for_type!(c, Posit16_2, Posit16_2);
    create_benchmark_for_type!(c, Posit32_2, Posit32_2);
    create_benchmark_for_type!(c, Softposit16_1, Softposit16_1);
    create_benchmark_for_type!(c, Softposit32_2, Softposit32_2);
}

criterion_group!(benches, benchmark_training_types);
criterion_main!(benches);
