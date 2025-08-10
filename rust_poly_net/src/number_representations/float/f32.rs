use crate::number_representations::core::{AIFloat, MlScalar};

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

impl MlScalar for f32 {}
