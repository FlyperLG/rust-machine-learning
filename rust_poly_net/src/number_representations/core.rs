use std::{
    fmt::{Debug, Display},
    ops::{AddAssign, Neg, SubAssign},
};

use ndarray::{LinalgScalar, ScalarOperand};
use num_traits::{Num, NumCast};

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
    + PartialOrd
{
}

pub trait AIFloat {
    fn exp(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
}
