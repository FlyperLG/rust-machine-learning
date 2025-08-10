use std::{
    cmp::Ordering,
    fmt,
    ops::{Add, AddAssign, Div, Mul, Neg, Rem, Sub, SubAssign},
};

use ndarray::ScalarOperand;
use num_traits::{Num, NumCast, One, ToPrimitive, Zero};

// --- MODIFIED: Using the specified structs ---
use crate::number_representations::{
    core::{AIFloat, MlScalar},
    posit::core::{DecodedPosit16, Posit, UnpackedPosit16},
};

#[derive(Debug, Clone, Copy)]
pub struct Posit16_2 {
    bits: u16,
}

impl MlScalar for Posit16_2 {}

impl Posit<16, 2> for Posit16_2 {}

impl Posit16_2 {
    pub const ZERO: Self = Posit16_2 { bits: 0 };
    pub const NAR: Self = Posit16_2 {
        bits: 1 << (Self::N - 1),
    };

    pub fn new(bits: u16) -> Self {
        Posit16_2 { bits }
    }

    // --- MODIFIED: Return type is now DecodedPosit16 ---
    fn decode(&self) -> DecodedPosit16 {
        if self.bits == Self::ZERO.bits {
            return DecodedPosit16 {
                is_nar: false,
                is_zero: true,
                sign: 0,
                scale: 0,
                mantissa: 0,
                frac_len: 0,
            };
        }
        if self.bits == Self::NAR.bits {
            return DecodedPosit16 {
                is_nar: true,
                is_zero: false,
                sign: 0,
                scale: 0,
                mantissa: 0,
                frac_len: 0,
            };
        }

        let sign = self.bits >> 15;
        let abs_bits = if sign == 1 {
            (-(self.bits as i16)) as u16
        } else {
            self.bits
        };

        let body = abs_bits << 1;
        let regime_msb = body >> 15;

        let k: i32;
        let regime_len: u32;

        if regime_msb == 1 {
            let num_ones = body.leading_ones();
            if num_ones >= 15 {
                // maxpos
                return DecodedPosit16 {
                    is_nar: false,
                    is_zero: false,
                    sign,
                    scale: (14 * (1 << Self::ES)) as i16,
                    mantissa: 1,
                    frac_len: 0,
                };
            }
            regime_len = num_ones + 1;
            k = num_ones as i32 - 1;
        } else {
            let num_zeros = body.leading_zeros();
            if num_zeros >= 14 {
                // minpos
                return DecodedPosit16 {
                    is_nar: false,
                    is_zero: false,
                    sign,
                    scale: (-14 * (1 << Self::ES)) as i16,
                    mantissa: 1,
                    frac_len: 0,
                };
            }
            regime_len = num_zeros + 1;
            k = -(num_zeros as i32);
        }

        let scale_base = k * (1 << Self::ES);
        let remaining_len = 15 - regime_len;

        let es_len = std::cmp::min(Self::ES as u32, remaining_len);
        let frac_len = remaining_len - es_len;

        let exp_frac_bits = (body << regime_len) >> (16 - remaining_len);

        let es_val = if es_len > 0 {
            (exp_frac_bits >> frac_len) as i32
        } else {
            0
        };

        let scale = scale_base + es_val;

        let frac_mask = if frac_len > 0 {
            (1u32 << frac_len) - 1
        } else {
            0
        };
        let frac_bits = (exp_frac_bits as u32) & frac_mask;
        let mantissa = (1u32 << frac_len) | frac_bits;

        // --- MODIFIED: Constructing DecodedPosit16 with correct types ---
        DecodedPosit16 {
            is_nar: false,
            is_zero: false,
            sign: sign as u16,
            scale: scale as i16,
            mantissa,
            frac_len: frac_len as u16,
        }
    }

    // --- MODIFIED: Parameter is now UnpackedPosit16 ---
    fn encode(unpacked: UnpackedPosit16) -> Self {
        if unpacked.mantissa == 0 {
            return Self::ZERO;
        }

        let result_sign = unpacked.sign;
        let mantissa = unpacked.mantissa;

        // STEP 1: NORMALIZE the unpacked mantissa
        // --- MODIFIED: Using 63 for i64 mantissa ---
        let msb_pos = 63 - mantissa.leading_zeros() as i32;

        let final_scale = unpacked.scale as i32 + msb_pos;

        // STEP 2: CALCULATE POSIT COMPONENTS
        let useed_power = (1 << Self::ES) as i32;
        let k = final_scale.div_euclid(useed_power);
        let es_val = (final_scale - k * useed_power) as u32;

        let regime_len = if k >= 0 { k as u32 + 2 } else { -k as u32 + 1 };

        if regime_len >= 15 {
            let body = if k >= 0 { 0x7FFF } else { 1 };
            let bits = if result_sign == 1 {
                (-(body as i16)) as u16
            } else {
                body
            };
            return Self::new(bits);
        }

        let regime_body = if k >= 0 {
            let num_ones = k as u32 + 1;
            (!0u16 << (16 - num_ones)) >> 1
        } else {
            1 << (15 - regime_len)
        };

        let available_len = 15 - regime_len;
        let es_len = std::cmp::min(Self::ES as u32, available_len);
        let frac_len = available_len - es_len;

        // STEP 3: PERFORM ROUNDING
        let shift_for_round = msb_pos - frac_len as i32;
        // --- MODIFIED: Using 64 for i64 mantissa ---
        if shift_for_round > 64 {
            return Self::ZERO;
        }

        let mut frac_plus_round = if shift_for_round > 0 {
            mantissa >> (shift_for_round - 1)
        } else {
            mantissa << (1 - shift_for_round)
        };

        if (frac_plus_round & 1) != 0 {
            if (frac_plus_round & 2) != 0 || (mantissa & ((1 << (shift_for_round - 1)) - 1)) != 0 {
                frac_plus_round += 2;
            }
        }

        // STEP 4: ASSEMBLE FINAL BITS
        let frac_bits = (frac_plus_round >> 1) as u16 & ((1 << frac_len) - 1);
        let es_bits = (es_val as u16) << frac_len;
        let payload_mask = if available_len >= 16 {
            u16::MAX
        } else {
            (1 << available_len) - 1
        };
        let body = regime_body | ((es_bits | frac_bits) & payload_mask);

        let final_bits = if result_sign == 1 {
            (-(body as i16)) as u16
        } else {
            body
        };
        Self::new(final_bits)
    }

    fn trunc(self) -> Self {
        let d = self.decode();
        if d.is_nar || d.is_zero {
            return self;
        }

        if d.scale < 0 {
            return Self::ZERO;
        }

        if d.scale >= d.frac_len as i16 {
            return self;
        }

        let bits_to_chop = d.frac_len as i16 - d.scale;

        let trunc_mantissa = (d.mantissa >> bits_to_chop) << bits_to_chop;

        // --- MODIFIED: Constructing UnpackedPosit16 with correct types ---
        let unpacked = UnpackedPosit16 {
            sign: d.sign,
            scale: d.scale - d.frac_len as i16,
            mantissa: trunc_mantissa as i64,
        };

        Self::encode(unpacked)
    }

    pub fn abs(self) -> Self {
        if self.bits == Self::NAR.bits {
            return Self::NAR;
        }

        let is_negative = (self.bits >> 15) == 1;
        if is_negative { -self } else { self }
    }

    pub fn round(self) -> Self {
        if self.bits == Self::NAR.bits {
            return Self::NAR;
        }

        let half = <Self as From<f32>>::from(0.5);
        // Check for negative, but don't include NaR (which is negative in 2's complement)
        let is_negative = (self.bits as i16) < 0;

        if is_negative {
            (self - half).trunc()
        } else {
            (self + half).trunc()
        }
    }
}

impl From<f32> for Posit16_2 {
    fn from(value: f32) -> Self {
        if value == 0.0 {
            return Self::ZERO;
        }
        if value.is_nan() || value.is_infinite() {
            return Self::NAR;
        }

        let input_bits = value.to_bits();
        let sign = (input_bits >> 31) as u16;

        let (exp, frac) = if value.is_subnormal() {
            let n = value.abs().to_bits().leading_zeros() - 8;
            (1 - 127 - (n as i32), (input_bits << n) & 0x7FFFFF)
        } else {
            (
                ((input_bits >> 23) & 0xFF) as i32 - 127,
                input_bits & 0x7FFFFF,
            )
        };

        let useed_power = (1 << Self::ES) as i32;
        let k = exp.div_euclid(useed_power);
        let es_val = (exp - k * useed_power) as u32;

        let regime_len = if k >= 0 { k as u32 + 2 } else { -k as u32 + 1 };

        let body: u16;

        if regime_len >= 15 {
            body = if k >= 0 { 0x7FFF } else { 1 };
        } else {
            let regime_body = if k >= 0 {
                let num_ones = k as u32 + 1;
                (!0u16 << (16 - num_ones)) >> 1
            } else {
                1 << (15 - regime_len)
            };

            let available_len = 15 - regime_len;
            let es_len = std::cmp::min(Self::ES as u32, available_len);
            let frac_len = available_len - es_len;

            let es_payload = (es_val as u16) << frac_len;
            let frac_payload: u16;

            let shift = 23 - (frac_len as i32);

            if shift >= 0 {
                let f32_frac_bits = frac as u64;
                let mut temp_payload = f32_frac_bits >> shift;

                let guard_mask = if shift > 0 { 1u64 << (shift - 1) } else { 0 };
                let guard_bit = (f32_frac_bits & guard_mask) != 0;
                let sticky_mask = guard_mask.wrapping_sub(1);
                let sticky_bit = (f32_frac_bits & sticky_mask) != 0;

                if guard_bit && (sticky_bit || (temp_payload & 1) == 1) {
                    temp_payload += 1;
                }
                frac_payload = temp_payload as u16;
            } else {
                frac_payload = (frac << -shift) as u16;
            }

            let frac_mask = if frac_len >= 16 {
                u16::MAX
            } else {
                (1 << frac_len) - 1
            };
            let payload = es_payload | (frac_payload & frac_mask);

            body = regime_body | payload;
        }

        let final_bits = if sign == 1 {
            (-(body as i16)) as u16
        } else {
            body
        };

        Self::new(final_bits)
    }
}

impl From<Posit16_2> for f32 {
    fn from(p: Posit16_2) -> Self {
        let decoded = p.decode();

        if decoded.is_zero {
            return 0.0;
        }
        if decoded.is_nar {
            return f32::NAN;
        }

        let final_val = (decoded.mantissa as f64 / (1u64 << decoded.frac_len) as f64)
            * 2.0f64.powi(decoded.scale as i32);

        if decoded.sign == 1 {
            -final_val as f32
        } else {
            final_val as f32
        }
    }
}

impl Add for Posit16_2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let p1 = self.decode();
        let p2 = rhs.decode();

        if p1.is_nar || p2.is_nar {
            return Self::NAR;
        }
        if p1.is_zero {
            return rhs;
        }
        if p2.is_zero {
            return self;
        }

        // --- MODIFIED: Using i64 for mantissa calculations ---
        let mut mant1 = if p1.sign == 1 {
            -(p1.mantissa as i64)
        } else {
            p1.mantissa as i64
        };
        let mut mant2 = if p2.sign == 1 {
            -(p2.mantissa as i64)
        } else {
            p2.mantissa as i64
        };

        // --- MODIFIED: Shift by 48 to fit in i64 ---
        const PRE_SHIFT: i16 = 48;

        mant1 <<= PRE_SHIFT - p1.frac_len as i16;
        mant2 <<= PRE_SHIFT - p2.frac_len as i16;

        let scale_diff = p1.scale as i32 - p2.scale as i32;
        let result_scale;
        if scale_diff > 0 {
            result_scale = p1.scale;
            mant2 >>= scale_diff.min(63);
        } else {
            result_scale = p2.scale;
            mant1 >>= (-scale_diff).min(63);
        }

        let result_mant = mant1 + mant2;

        if result_mant == 0 {
            return Self::ZERO;
        }

        let result_sign = if result_mant < 0 { 1 } else { 0 };

        // --- MODIFIED: Constructing UnpackedPosit16 ---
        let unpacked = UnpackedPosit16 {
            sign: result_sign,
            scale: result_scale - PRE_SHIFT,
            mantissa: result_mant.abs(),
        };

        Self::encode(unpacked)
    }
}

impl Mul for Posit16_2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let p1 = self.decode();
        let p2 = rhs.decode();

        if p1.is_nar || p2.is_nar {
            return Self::NAR;
        }
        if p1.is_zero || p2.is_zero {
            return Self::ZERO;
        }

        let result_scale =
            p1.scale as i32 + p2.scale as i32 - p1.frac_len as i32 - p2.frac_len as i32;
        // --- MODIFIED: Using i64 for mantissa calculations ---
        let result_mant = (p1.mantissa as i64) * (p2.mantissa as i64);

        // --- MODIFIED: Constructing UnpackedPosit16 ---
        let unpacked = UnpackedPosit16 {
            sign: p1.sign ^ p2.sign,
            scale: result_scale as i16,
            mantissa: result_mant,
        };

        Self::encode(unpacked)
    }
}

impl Neg for Posit16_2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.bits == Self::NAR.bits || self.bits == Self::ZERO.bits {
            return self;
        }
        Self::new((-(self.bits as i16)) as u16)
    }
}

impl Sub for Posit16_2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Div for Posit16_2 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let p1 = self.decode();
        let p2 = rhs.decode();

        if p1.is_nar || p2.is_nar || p2.is_zero {
            return Self::NAR;
        }
        if p1.is_zero {
            return Self::ZERO;
        }

        let result_scale =
            p1.scale as i32 - p2.scale as i32 - (p1.frac_len as i32 - p2.frac_len as i32);

        // --- MODIFIED: Shift by 48 to fit in i64 ---
        const PRE_SHIFT: i32 = 48;
        let result_mant = ((p1.mantissa as i64) << PRE_SHIFT) / (p2.mantissa as i64);

        // --- MODIFIED: Constructing UnpackedPosit16 ---
        let unpacked = UnpackedPosit16 {
            sign: p1.sign ^ p2.sign,
            scale: (result_scale - PRE_SHIFT) as i16,
            mantissa: result_mant,
        };

        Self::encode(unpacked)
    }
}

impl Rem for Posit16_2 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        if self.bits == Self::NAR.bits || rhs.bits == Self::NAR.bits || rhs.bits == Self::ZERO.bits
        {
            return Self::NAR;
        }

        if self.bits == Self::ZERO.bits {
            return Self::ZERO;
        }

        let division_result = self / rhs;
        let truncated_division = Self::trunc(division_result);
        let product = truncated_division * rhs;
        let remainder = self - product;
        remainder
    }
}

impl PartialEq for Posit16_2 {
    fn eq(&self, other: &Self) -> bool {
        if self.bits == Self::NAR.bits || other.bits == Self::NAR.bits {
            return false;
        }
        self.bits == other.bits
    }
}

impl PartialOrd for Posit16_2 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.bits == Self::NAR.bits || other.bits == Self::NAR.bits {
            return None;
        }
        (self.bits as i16).partial_cmp(&(other.bits as i16))
    }
}

impl fmt::Display for Posit16_2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val: f32 = (*self).into();
        write!(f, "{}", val)
    }
}

impl AddAssign for Posit16_2 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Posit16_2 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl NumCast for Posit16_2 {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        n.to_f32().map(<Posit16_2 as From<f32>>::from)
    }
}

impl Zero for Posit16_2 {
    fn zero() -> Self {
        Posit16_2::ZERO
    }
    fn is_zero(&self) -> bool {
        self.bits == Posit16_2::ZERO.bits
    }
}

impl One for Posit16_2 {
    fn one() -> Self {
        <Posit16_2 as From<f32>>::from(1.0)
    }
}

impl Num for Posit16_2 {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f32::from_str_radix(s, radix).map(<Posit16_2 as From<f32>>::from)
    }
}

impl ToPrimitive for Posit16_2 {
    fn to_i64(&self) -> Option<i64> {
        <f32 as From<Posit16_2>>::from(*self).to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        <f32 as From<Posit16_2>>::from(*self).to_u64()
    }
    fn to_f64(&self) -> Option<f64> {
        <f32 as From<Posit16_2>>::from(*self).to_f64()
    }
}

impl ScalarOperand for Posit16_2 {}

impl AIFloat for Posit16_2 {
    fn exp(self) -> Self {
        const LN2_F64: f64 = core::f64::consts::LN_2;
        // max arg is ln(maxpos) = ln(2^(14*4)) = ln(2^56) approx 38.8
        const MAX_EXP_ARG: Posit16_2 = Posit16_2 { bits: 0x72EE }; // from(38.8)
        // min arg is ln(minpos) approx -38.8
        const MIN_EXP_ARG: Posit16_2 = Posit16_2 { bits: 0x8D12 }; // from(-38.8)

        if self.bits == Self::NAR.bits {
            return Self::NAR;
        } else if self.is_zero() {
            return <Self as From<f32>>::from(1.0);
        } else if self > MAX_EXP_ARG {
            return Posit16_2::new(0x7FFF); // maxpos
        } else if self < MIN_EXP_ARG {
            return Self::ZERO;
        }

        let self_f64: f64 = <f32 as From<Posit16_2>>::from(self) as f64;
        let k = (self_f64 / LN2_F64).round() as i32;
        let r_f64 = self_f64 - (k as f64 * LN2_F64);
        let r: Posit16_2 = <Self as From<f32>>::from(r_f64 as f32);

        let mut exp_r = <Self as From<f32>>::from(1.0);
        let mut term = <Self as From<f32>>::from(1.0);
        for i in 1..=10 {
            let i_posit = <Self as From<f32>>::from(i as f32);
            term = term * r / i_posit;
            let prev_sum = exp_r;
            exp_r = exp_r + term;
            if exp_r.bits == prev_sum.bits {
                break;
            }
        }

        let scale_factor = <Self as From<f32>>::from(2.0).powi(k);
        exp_r * scale_factor
    }

    fn powi(self, n: i32) -> Self {
        if self.bits == Self::NAR.bits {
            return Self::NAR;
        }

        if n == 0 {
            return <Self as From<f32>>::from(1.0);
        } else if self.bits == <Self as From<f32>>::from(1.0).bits {
            return self;
        } else if n == 1 {
            return self;
        } else if self.is_zero() {
            return if n < 0 { Self::NAR } else { Self::ZERO };
        }

        let mut res = <Self as From<f32>>::from(1.0);
        let mut base = self;
        let mut exp = n;
        if exp < 0 {
            base = <Self as From<f32>>::from(1.0) / base;
            exp = -exp;
        }

        while exp > 0 {
            if exp % 2 == 1 {
                res = res * base;
            }
            base = base * base;
            exp /= 2;
        }

        res
    }

    fn max(self, other: Self) -> Self {
        if self.bits == Self::NAR.bits {
            return other;
        } else if other.bits == Self::NAR.bits {
            return self;
        }

        if (self.bits as i16) > (other.bits as i16) {
            self
        } else {
            other
        }
    }

    fn min(self, other: Self) -> Self {
        if self.bits == Self::NAR.bits {
            return other;
        } else if other.bits == Self::NAR.bits {
            return self;
        }

        if (self.bits as i16) < (other.bits as i16) {
            self
        } else {
            other
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Import a reference posit implementation if available, or use f32 for comparison
    // For this example, we'll compare against expected f32 results,
    // as a direct posit<16,2> reference might not be readily available in crates.
    // If you have a trusted reference implementation, use it for more accurate tests.

    /// Helper function to compare your f32 -> posit conversion against expected bit patterns.
    fn validate_f32_to_posit(value: f32, expected_bits: u16) {
        let my_posit = <Posit16_2 as From<f32>>::from(value);

        if my_posit.bits != expected_bits {
            println!("Input f32: {}", value);
            println!("My posit:    {:#018b} ({})", my_posit.bits, my_posit.bits);
            println!("Expected:    {:#018b} ({})", expected_bits, expected_bits);
        }

        assert_eq!(
            my_posit.bits, expected_bits,
            "f32 -> posit conversion mismatch for {}",
            value
        );
    }

    #[test]
    fn test_f32_to_posit() {
        // Test cases for posit<16, 2>
        // Use an online converter or reference implementation to get these expected values
        validate_f32_to_posit(0.0, 0x0000);
        validate_f32_to_posit(f32::NAN, 0x8000);
        validate_f32_to_posit(1.0, 0x4000);
        validate_f32_to_posit(-1.0, 0xC000);
        validate_f32_to_posit(256.0, 0x6000); // useed = 16, k=1, exp=4. 256 = 16^2
        validate_f32_to_posit(16.0, 0x5000); // useed = 16, k=1, exp=0
        validate_f32_to_posit(4.0, 0x4800); // useed = 16, k=0, exp=2
        validate_f32_to_posit(1.0 / 16.0, 0x3000);
        validate_f32_to_posit(65536.0, 0x7000); // 16^4
        validate_f32_to_posit(2.718, 0x4A83);
    }

    /// Helper function to compare your posit -> f32 conversion against the reference.
    fn validate_posit_to_f32(p_bits: u16, expected_f32: f32, tolerance: f32) {
        let my_posit = Posit16_2::new(p_bits);
        let my_f32: f32 = my_posit.into();

        if my_f32.is_nan() && expected_f32.is_nan() {
            return;
        }
        let diff = (my_f32 - expected_f32).abs();

        if diff > tolerance {
            println!("Input bits:  {:#018b} ({})", p_bits, p_bits);
            println!("My f32:      {} ({:#010x})", my_f32, my_f32.to_bits());
            println!(
                "Expected f32:{} ({:#010x})",
                expected_f32,
                expected_f32.to_bits()
            );
        }

        assert!(
            diff <= tolerance,
            "posit -> f32 conversion mismatch for bits {:#x}",
            p_bits
        );
    }

    #[test]
    fn test_posit_to_f32() {
        validate_posit_to_f32(0x0000, 0.0, 1e-9); // Zero
        validate_posit_to_f32(0x8000, f32::NAN, 1e-9); // NaR
        validate_posit_to_f32(0x4000, 1.0, 1e-9); // 1.0
        validate_posit_to_f32(0xC000, -1.0, 1e-9); // -1.0
        validate_posit_to_f32(0x7FFF, 2.8147498e17, 1e12); // maxpos (16^14)
        validate_posit_to_f32(0x0001, 3.5527137e-18, 1e-24); // minpos (16^-14)
        validate_posit_to_f32(0x5000, 16.0, 1e-9);
        validate_posit_to_f32(0x3000, 1.0 / 16.0, 1e-9);
    }

    fn validate_op(
        v1: f32,
        v2: f32,
        op: &str,
        my_res_p: Posit16_2,
        expected_f32: f32,
        tolerance: f32,
    ) {
        let my_res_f32: f32 = my_res_p.into();
        let diff = (my_res_f32 - expected_f32).abs();
        let expected_within_posit_precision = <Posit16_2 as From<f32>>::from(expected_f32);
        let my_res_bits = my_res_p.bits;
        let expected_bits = expected_within_posit_precision.bits;

        // Don't fail for tiny differences that are due to f32 vs posit precision
        if diff > tolerance && my_res_bits != expected_bits {
            println!("\n{} test failed for {} {} {}", op, v1, op, v2);
            println!("My Result (f32):       {}", my_res_f32);
            println!("Expected Result (f32): {}", expected_f32);
            println!("My Result (bits):      {:#018b}", my_res_bits);
            println!("Expected Result (bits):{:#018b}", expected_bits);
            println!("Difference:              {}", diff);
        }

        // Use bit-level comparison for higher fidelity check
        assert!(
            my_res_bits == expected_bits,
            "{} mismatch for {} {} {}. Got {}, expected {}",
            op,
            v1,
            op,
            v2,
            my_res_f32,
            expected_f32
        );
    }

    #[test]
    fn test_posit_arithmetic() {
        let pairs = [
            (1.0, 2.0),
            (16.0, 16.0),
            (100.0, -3.5),
            (0.125, 0.5),
            (2.75, 4.5),
            (-1000.0, -0.001),
            (2.81e17, 2.0), // near maxpos
        ];
        for &(v1, v2) in &pairs {
            validate_op(
                v1,
                v2,
                "+",
                <Posit16_2 as From<f32>>::from(v1) + <Posit16_2 as From<f32>>::from(v2),
                v1 + v2,
                1e-4,
            );
            validate_op(
                v1,
                v2,
                "-",
                <Posit16_2 as From<f32>>::from(v1) - <Posit16_2 as From<f32>>::from(v2),
                v1 - v2,
                1e-4,
            );
            validate_op(
                v1,
                v2,
                "*",
                <Posit16_2 as From<f32>>::from(v1) * <Posit16_2 as From<f32>>::from(v2),
                v1 * v2,
                1e-4,
            );
            if v2 != 0.0 {
                validate_op(
                    v1,
                    v2,
                    "/",
                    <Posit16_2 as From<f32>>::from(v1) / <Posit16_2 as From<f32>>::from(v2),
                    v1 / v2,
                    1e-4,
                );
            }
        }
    }

    // Other tests (neg, rem, ordering, trunc, round, etc.) from the posit16_1 implementation
    // can be adapted similarly. The key is to generate correct expected values for posit<16, 2>.
}
