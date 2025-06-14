use std::{
    cmp::Ordering,
    fmt,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
};

use ndarray::ScalarOperand;
use rand_distr::num_traits::{One, Zero};

pub trait FromNumber {
    fn from_f64(num: f64) -> Self;
    fn from_f32(num: f32) -> Self;
}

pub trait Exponential {
    fn exp(self) -> Self;
    fn powi(self, n: i32) -> Self;
}

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct Float64 {
    pub value: f64,
}

impl Exponential for Float64 {
    fn exp(self) -> Float64 {
        Float64 {
            value: self.value.exp(),
        }
    }

    fn powi(self, n: i32) -> Float64 {
        Float64 {
            value: self.value.powi(n),
        }
    }
}

impl FromNumber for Float64 {
    fn from_f64(num: f64) -> Self {
        Float64 { value: num }
    }

    fn from_f32(num: f32) -> Self {
        Float64 { value: num as f64 }
    }
}

impl fmt::Display for Float64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Eq for Float64 {}

impl Ord for Float64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.value < other.value {
            Ordering::Less
        } else if self.value == other.value {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    }
}

impl Add for Float64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Float64 {
            value: self.value + rhs.value,
        }
    }
}

impl Sub for Float64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Float64 {
            value: self.value - rhs.value,
        }
    }
}

impl Neg for Float64 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Float64 { value: -self.value }
    }
}

impl Mul for Float64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Float64 {
            value: self.value * rhs.value,
        }
    }
}

impl Div for Float64 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Float64 {
            value: self.value / rhs.value,
        }
    }
}

impl AddAssign for Float64 {
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl SubAssign for Float64 {
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl Zero for Float64 {
    fn zero() -> Self {
        Float64 { value: 0.0 }
    }

    fn is_zero(&self) -> bool {
        self.value == 0.0
    }
}

impl One for Float64 {
    fn one() -> Self {
        Float64 { value: 1.0 }
    }
}

impl ScalarOperand for Float64 {}
