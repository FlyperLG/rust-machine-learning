use ndarray::ScalarOperand;
use num_traits::{Num, NumCast, One, ToPrimitive, Zero};
use softposit::P16E1;
use std::{
    fmt::{Debug, Display, Formatter},
    ops::{Add, AddAssign, Div, Mul, Neg, Rem, Sub, SubAssign},
    str::FromStr,
};

use crate::number_representations::core::{AIFloat, MlScalar};

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct Softposit16_1(pub P16E1);

impl ScalarOperand for Softposit16_1 {}

impl AIFloat for Softposit16_1 {
    fn exp(self) -> Self {
        // Convert to f64, perform the operation, and convert back.
        Softposit16_1(<P16E1 as From<f64>>::from(
            <f64 as From<P16E1>>::from(self.0).exp(),
        ))
    }

    fn powi(self, n: i32) -> Self {
        Softposit16_1(<P16E1 as From<f64>>::from(
            <f64 as From<P16E1>>::from(self.0).powi(n),
        ))
    }

    fn max(self, other: Self) -> Self {
        // We can use the PartialOrd we derived for the struct.
        if self > other { self } else { other }
    }

    fn min(self, other: Self) -> Self {
        if self < other { self } else { other }
    }
}

impl Neg for Softposit16_1 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Softposit16_1(-self.0)
    }
}

impl Add for Softposit16_1 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Softposit16_1(self.0 + rhs.0)
    }
}

impl Sub for Softposit16_1 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Softposit16_1(self.0 - rhs.0)
    }
}

impl Mul for Softposit16_1 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Softposit16_1(self.0 * rhs.0)
    }
}

impl Div for Softposit16_1 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Softposit16_1(self.0 / rhs.0)
    }
}

impl Rem for Softposit16_1 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        Softposit16_1(self.0 % rhs.0)
    }
}

impl AddAssign for Softposit16_1 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Softposit16_1 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Display for Softposit16_1 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Debug for Softposit16_1 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Softposit16_1({})", self.0)
    }
}

impl Zero for Softposit16_1 {
    fn zero() -> Self {
        Softposit16_1(P16E1::ZERO)
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for Softposit16_1 {
    fn one() -> Self {
        Softposit16_1(P16E1::ONE)
    }
}

impl Num for Softposit16_1 {
    type FromStrRadixErr = num_traits::ParseFloatError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        // The easiest path is to parse to a standard float first.
        let val = f64::from_str_radix(str, radix)?;
        Ok(Softposit16_1(<P16E1 as From<f64>>::from(val)))
    }
}

impl NumCast for Softposit16_1 {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        // Use f64 as a robust intermediate for casting.
        n.to_f64()
            .map(|f| Softposit16_1(<P16E1 as From<f64>>::from(f)))
    }
}

impl ToPrimitive for Softposit16_1 {
    fn to_i64(&self) -> Option<i64> {
        self.to_f64().and_then(|f| f.to_i64())
    }

    fn to_u64(&self) -> Option<u64> {
        self.to_f64().and_then(|f| f.to_u64())
    }

    // The most direct conversion is to a float
    fn to_f64(&self) -> Option<f64> {
        Some(<f64 as From<P16E1>>::from(self.0))
    }

    fn to_f32(&self) -> Option<f32> {
        Some(<f32 as From<P16E1>>::from(self.0))
    }
}

impl MlScalar for Softposit16_1 {}
