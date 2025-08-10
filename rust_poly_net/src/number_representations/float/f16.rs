use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign},
};

use crate::number_representations::core::{AIFloat, MlScalar};
use half::f16;
use ndarray::ScalarOperand;
use num_traits::{Num, NumCast, real::Real};

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MLf16(pub f16);

impl From<f16> for MLf16 {
    fn from(val: f16) -> Self {
        MLf16(val)
    }
}

impl From<MLf16> for f16 {
    fn from(val: MLf16) -> Self {
        val.0
    }
}

impl AIFloat for MLf16 {
    fn exp(self) -> Self {
        MLf16(self.0.exp()) // Unwrap, operate, and wrap back up
    }
    fn powi(self, n: i32) -> Self {
        MLf16(self.0.powi(n))
    }
    fn max(self, other: Self) -> Self {
        MLf16(self.0.max(other.0))
    }
    fn min(self, other: Self) -> Self {
        MLf16(self.0.min(other.0))
    }
}

impl ScalarOperand for MLf16 {}

impl MlScalar for MLf16 {}

impl Add for MLf16 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        MLf16(self.0 + rhs.0)
    }
}
impl Sub for MLf16 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        MLf16(self.0 - rhs.0)
    }
}
impl Mul for MLf16 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        MLf16(self.0 * rhs.0)
    }
}
impl Div for MLf16 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        MLf16(self.0 / rhs.0)
    }
}
impl Rem for MLf16 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        MLf16(self.0 % rhs.0)
    }
}
impl AddAssign for MLf16 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}
impl SubAssign for MLf16 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}
impl MulAssign for MLf16 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}
impl DivAssign for MLf16 {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}
impl Neg for MLf16 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        MLf16(-self.0)
    }
}

impl num_traits::Zero for MLf16 {
    fn zero() -> Self {
        MLf16(f16::ZERO)
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl num_traits::One for MLf16 {
    fn one() -> Self {
        MLf16(f16::ONE)
    }
}

// Finally, the `Num` trait itself
impl Num for MLf16 {
    type FromStrRadixErr = <f16 as Num>::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f16::from_str_radix(str, radix).map(MLf16)
    }
}

// --- Implement `NumCast` ---
impl NumCast for MLf16 {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        <half::f16 as NumCast>::from(n).map(MLf16)
    }
}

// You might also need `ToPrimitive` if other parts of your code require it
impl num_traits::ToPrimitive for MLf16 {
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }
    fn to_f32(&self) -> Option<f32> {
        Some(self.0.to_f32())
    }
    fn to_f64(&self) -> Option<f64> {
        Some(self.0.to_f64())
    }
}

impl Display for MLf16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
