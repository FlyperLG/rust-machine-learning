use ndarray::{LinalgScalar, ScalarOperand};
use num_traits::{Num, NumCast, ToPrimitive};
use rand_distr::num_traits::{One, Zero};
use std::{
    fmt::{self, Debug, Display},
    ops::{Add, AddAssign, Div, Mul, Neg, Rem, Sub, SubAssign},
};

pub trait AIFloat {
    fn exp(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
}

pub trait MlScalar:
    LinalgScalar
    + ScalarOperand
    + Num
    + NumCast
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + Display
    + Debug
    + AIFloat
{
}

impl AIFloat for f32 {
    fn exp(self) -> Self {
        self.exp()
    }
    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
}

impl AIFloat for f64 {
    fn exp(self) -> Self {
        self.exp()
    }
    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
}

impl MlScalar for f32 {}
impl MlScalar for f64 {}

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct Float64 {
    pub value: f64,
}

impl AIFloat for Float64 {
    fn exp(self) -> Self {
        Float64 {
            value: self.value.exp(),
        }
    }
    fn powi(self, n: i32) -> Self {
        Float64 {
            value: self.value.powi(n),
        }
    }
    fn max(self, other: Self) -> Self {
        Float64 {
            value: self.value.max(other.value),
        }
    }
    fn min(self, other: Self) -> Self {
        Float64 {
            value: self.value.min(other.value),
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

impl Rem for Float64 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        Float64 {
            value: self.value % rhs.value,
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

impl Display for Float64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl ScalarOperand for Float64 {}

impl Num for Float64 {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f64::from_str_radix(str, radix).map(|value| Float64 { value })
    }
}

impl ToPrimitive for Float64 {
    fn to_i64(&self) -> Option<i64> {
        self.value.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.value.to_u64()
    }
    fn to_f64(&self) -> Option<f64> {
        Some(self.value)
    }
}

impl NumCast for Float64 {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        n.to_f64().map(|value| Float64 { value })
    }
}

impl MlScalar for Float64 {}
