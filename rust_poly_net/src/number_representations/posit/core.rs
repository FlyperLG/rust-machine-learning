pub trait Posit<const N: usize, const ES: usize> {
    const N: usize = N;
    const ES: usize = ES;
    const USEED: usize = 1 << (1 << ES);
}

#[derive(Debug)]
pub(crate) struct DecodedPosit16 {
    pub is_nar: bool,
    pub is_zero: bool,
    pub sign: u16,
    pub scale: i16,
    // Use a large integer to hold the fraction with its implicit '1' bit
    // and provide headroom for shifting and rounding.
    pub mantissa: u32,
    pub frac_len: u16,
}

#[derive(Debug)]
pub struct UnpackedPosit16 {
    pub sign: u16,
    pub scale: i16,
    pub mantissa: i64, // Using i64 to hold the large intermediate result
}

#[derive(Debug)]
pub(crate) struct DecodedPosit32 {
    pub is_nar: bool,
    pub is_zero: bool,
    pub sign: u32,
    pub scale: i32,
    // Use a large integer to hold the fraction with its implicit '1' bit
    // and provide headroom for shifting and rounding.
    pub mantissa: u64,
    pub frac_len: u32,
}

#[derive(Debug)]
pub struct UnpackedPosit32 {
    pub sign: u32,
    pub scale: i32,
    pub mantissa: i128, // Using i128 to hold the large intermediate result
}
