use ndarray::{LinalgScalar, ScalarOperand};
use num_traits::{Num, NumCast, ToPrimitive};
use rand_distr::num_traits::{One, Zero};
use softposit::{P32, P32E2};
use std::{
    f64,
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

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct MlPosit {
    pub value: P32,
}

// The AIFloat trait often requires converting to a float for the operation
// and then converting back, as these functions are not always native to posits.
impl AIFloat for MlPosit {
    fn exp(self) -> Self {
        let f64_val = <f64 as NumCast>::from(self.value).unwrap();
        MlPosit {
            value: <P32 as NumCast>::from(f64_val).unwrap(),
        }
    }
    fn powi(self, n: i32) -> Self {
        let f64_val = <f64 as NumCast>::from(self.value).unwrap();
        MlPosit {
            value: <P32 as NumCast>::from(f64_val.powi(n)).unwrap(),
        }
    }
    fn max(self, other: Self) -> Self {
        // Posits from the `softposit` crate already implement PartialOrd,
        // so we can compare them directly.
        if self.value > other.value {
            self
        } else {
            other
        }
    }
    fn min(self, other: Self) -> Self {
        if self.value < other.value {
            self
        } else {
            other
        }
    }
}

// --- Implement standard arithmetic operators by wrapping the inner posit's operations ---

impl Add for MlPosit {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        MlPosit {
            value: self.value + rhs.value,
        }
    }
}

impl Sub for MlPosit {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        MlPosit {
            value: self.value - rhs.value,
        }
    }
}

impl Neg for MlPosit {
    type Output = Self;
    fn neg(self) -> Self::Output {
        MlPosit { value: -self.value }
    }
}

impl Mul for MlPosit {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        MlPosit {
            value: self.value * rhs.value,
        }
    }
}

impl Div for MlPosit {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        MlPosit {
            value: self.value / rhs.value,
        }
    }
}

impl Rem for MlPosit {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        // The Remainder operation is typically done via f64 conversion.
        let self_f64 = <f64 as NumCast>::from(self.value).unwrap();
        let rhs_f64 = <f64 as NumCast>::from(rhs.value).unwrap();
        MlPosit {
            value: <P32 as NumCast>::from(self_f64 % rhs_f64).unwrap(),
        }
    }
}

impl AddAssign for MlPosit {
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl SubAssign for MlPosit {
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

// --- Implement traits from num-traits ---

impl Zero for MlPosit {
    fn zero() -> Self {
        MlPosit { value: P32::ZERO }
    }

    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
}

impl One for MlPosit {
    fn one() -> Self {
        MlPosit { value: P32::ONE }
    }
}

impl Display for MlPosit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display the inner posit value
        write!(f, "{}", self.value)
    }
}

// This is a marker trait indicating it can be used in ndarray expressions.
impl ScalarOperand for MlPosit {}

impl Num for MlPosit {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;

    // Posits don't have a standard string radix representation like integers.
    // The most practical approach is to parse as a float and convert.
    // We will support radix 10 for float-like strings.
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        // Defer to f64's parsing logic and then convert the result to a Posit.
        f64::from_str_radix(str, radix).map(|value| MlPosit {
            value: <P32 as NumCast>::from(value).unwrap(),
        })
    }
}

// --- Implement conversion traits ---

impl ToPrimitive for MlPosit {
    // Posits can be converted to floats. Conversion to integers can be lossy.
    fn to_i64(&self) -> Option<i64> {
        <f64 as NumCast>::from(self.value).unwrap().to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        <f64 as NumCast>::from(self.value).unwrap().to_u64()
    }
    fn to_f64(&self) -> Option<f64> {
        Some(<f64 as NumCast>::from(self.value).unwrap())
    }
    // You can add other conversions here if needed.
}

impl NumCast for MlPosit {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        // The most reliable way to cast from an unknown primitive is to
        // first convert it to f64, and then from f64 to our posit type.
        n.to_f64().map(|value| MlPosit {
            value: <P32 as NumCast>::from(value).unwrap(),
        })
    }
}

// Finally, after satisfying all trait bounds, we can implement our main trait.
impl MlScalar for MlPosit {}
