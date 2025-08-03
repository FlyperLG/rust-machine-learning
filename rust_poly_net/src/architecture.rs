use std::marker::PhantomData;

use ndarray::{Array, Array1, Array2, ArrayView2};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use rust_poly_net::MlScalar;

pub struct Network<T: MlScalar> {
    layers: Vec<Box<dyn Layer<Scalar = T>>>,
}

impl<T: MlScalar> Network<T> {
    pub fn new(layers: Vec<Box<dyn Layer<Scalar = T>>>) -> Self {
        Network { layers: layers }
    }
}

impl<T: MlScalar> Network<T> {
    pub fn forward(&self, input: &ArrayView2<T>) -> (Vec<Array2<T>>, Array2<T>) {
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.to_owned()); // store input

        let mut output = input.to_owned();
        for layer in &self.layers {
            output = layer.forward(&output);
            activations.push(output.clone());
        }

        (activations, output)
    }

    pub fn backward(
        &mut self,
        activations: &[Array2<T>],
        target: &ArrayView2<T>,
        learning_rate: T,
    ) {
        let mut grad = activations.last().unwrap() - target;

        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            grad = layer.backward(&activations[i], &grad, learning_rate);
        }
    }
}

pub trait Layer {
    type Scalar: MlScalar;

    fn forward(&self, input: &Array2<Self::Scalar>) -> Array2<Self::Scalar>;
    fn backward(
        &mut self,
        input: &Array2<Self::Scalar>,
        grad_output: &Array2<Self::Scalar>,
        learning_rate: Self::Scalar,
    ) -> Array2<Self::Scalar>;
}

pub struct LinearLayer<T: MlScalar> {
    weights: Array2<T>,
    biases: Array1<T>,
    activation: Box<dyn Activation<Scalar = T>>,
}

impl<T: MlScalar> LinearLayer<T> {
    pub fn new(
        input_neurons: usize,
        output_neurons: usize,
        activation: Box<dyn Activation<Scalar = T>>,
    ) -> Self {
        LinearLayer::<T> {
            weights: Self::generate_weights(input_neurons, output_neurons),
            biases: Self::generate_biases(output_neurons),
            activation: activation,
        }
    }

    fn generate_weights(x: usize, y: usize) -> Array2<T> {
        let mut rng = StdRng::seed_from_u64(69);
        let mut list = Vec::with_capacity(x * y);
        for _ in 0..(x * y) {
            let value: f64 = StandardNormal.sample(&mut rng);
            list.push(T::from(0.1 * value).unwrap());
        }
        Array::from_shape_vec((x, y), list).unwrap()
    }

    fn generate_biases(x: usize) -> Array1<T> {
        let mut rng = StdRng::seed_from_u64(42);
        let mut list = Vec::with_capacity(x);
        for _ in 0..x {
            let value: f64 = StandardNormal.sample(&mut rng);
            list.push(T::from(0.1 * value).unwrap());
        }
        Array::from_shape_vec(x, list).unwrap()
    }
}

impl<T: MlScalar> Layer for LinearLayer<T> {
    type Scalar = T;

    fn forward(&self, input: &Array2<Self::Scalar>) -> Array2<Self::Scalar> {
        let output = input.dot(&self.weights) + &self.biases;
        self.activation.activate(&output)
    }

    fn backward(
        &mut self,
        input: &Array2<Self::Scalar>,
        grad_output: &Array2<Self::Scalar>,
        learning_rate: Self::Scalar,
    ) -> Array2<Self::Scalar> {
        let z = input.dot(&self.weights) + &self.biases;
        let a_prime = self.activation.derivative(&z);
        let delta = grad_output * &a_prime;

        let grad_weights = input.t().dot(&delta);
        let grad_biases = delta.sum_axis(ndarray::Axis(0));

        self.weights -= &(&grad_weights * learning_rate);
        self.biases -= &(grad_biases * learning_rate);

        delta.dot(&self.weights.t())
    }
}

pub trait Activation {
    type Scalar: MlScalar;

    fn activate(&self, input: &Array2<Self::Scalar>) -> Array2<Self::Scalar>;
    fn derivative(&self, input: &Array2<Self::Scalar>) -> Array2<Self::Scalar>;
}

pub struct Sigmoid<T: MlScalar> {
    _marker: PhantomData<T>,
}

impl<T: MlScalar> Sigmoid<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: MlScalar> Activation for Sigmoid<T> {
    type Scalar = T;

    fn activate(&self, input: &Array2<Self::Scalar>) -> Array2<Self::Scalar> {
        input.mapv(|v| {
            let v = v.max(T::from(-709.0).unwrap()).min(T::from(709.0).unwrap());
            T::one() / (T::one() + (-v).exp())
        })
    }

    fn derivative(&self, input: &Array2<Self::Scalar>) -> Array2<Self::Scalar> {
        let activated_input = self.activate(input);
        // Then, apply the derivative formula: sigma(z) * (1 - sigma(z))
        activated_input.mapv(|v| v * (T::one() - v))
    }
}

pub struct ReLu<T: MlScalar> {
    _marker: PhantomData<T>,
}

impl<T: MlScalar> ReLu<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: MlScalar> Activation for ReLu<T> {
    type Scalar = T;

    fn activate(&self, input: &Array2<Self::Scalar>) -> Array2<Self::Scalar> {
        input.mapv(|v| if v > T::zero() { v } else { T::zero() })
    }

    fn derivative(&self, input: &Array2<Self::Scalar>) -> Array2<Self::Scalar> {
        input.mapv(|v| if v > T::zero() { T::one() } else { T::zero() })
    }
}

pub struct HardSigmoid<T: MlScalar> {
    _marker: PhantomData<T>,
}

impl<T: MlScalar> HardSigmoid<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: MlScalar> Activation for HardSigmoid<T> {
    type Scalar = T;

    fn activate(&self, input: &Array2<Self::Scalar>) -> Array2<Self::Scalar> {
        let slope = T::from(0.2).unwrap();
        let offset = T::from(0.5).unwrap();
        input.mapv(|v| {
            let v = v.max(T::from(-2.5).unwrap()).min(T::from(2.5).unwrap());
            (slope * v + offset).max(T::zero()).min(T::one())
        })
    }

    fn derivative(&self, input: &Array2<Self::Scalar>) -> Array2<Self::Scalar> {
        let activated_input = self.activate(input);
        activated_input.mapv(|v| {
            if v > T::zero() && v < T::one() {
                T::from(0.2).unwrap()
            } else {
                T::zero()
            }
        })
    }
}
