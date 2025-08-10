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
pub struct Posit16_1 {
    bits: u16,
}

impl MlScalar for Posit16_1 {}

impl Posit<16, 1> for Posit16_1 {}

impl Posit16_1 {
    pub const ZERO: Self = Posit16_1 { bits: 0 };
    pub const NAR: Self = Posit16_1 {
        bits: 1 << (Self::N - 1),
    };

    pub fn new(bits: u16) -> Self {
        Posit16_1 { bits }
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

impl From<f32> for Posit16_1 {
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

impl From<Posit16_1> for f32 {
    fn from(p: Posit16_1) -> Self {
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

impl Add for Posit16_1 {
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

impl Mul for Posit16_1 {
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

impl Neg for Posit16_1 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.bits == Self::NAR.bits || self.bits == Self::ZERO.bits {
            return self;
        }
        Self::new((-(self.bits as i16)) as u16)
    }
}

impl Sub for Posit16_1 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Div for Posit16_1 {
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

impl Rem for Posit16_1 {
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

impl PartialEq for Posit16_1 {
    fn eq(&self, other: &Self) -> bool {
        if self.bits == Self::NAR.bits || other.bits == Self::NAR.bits {
            return false;
        }
        self.bits == other.bits
    }
}

impl PartialOrd for Posit16_1 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.bits == Self::NAR.bits || other.bits == Self::NAR.bits {
            return None;
        }
        (self.bits as i16).partial_cmp(&(other.bits as i16))
    }
}

impl fmt::Display for Posit16_1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val: f32 = (*self).into();
        write!(f, "{}", val)
    }
}

impl AddAssign for Posit16_1 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Posit16_1 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl NumCast for Posit16_1 {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        n.to_f32().map(<Posit16_1 as From<f32>>::from)
    }
}

impl Zero for Posit16_1 {
    fn zero() -> Self {
        Posit16_1::ZERO
    }
    fn is_zero(&self) -> bool {
        self.bits == Posit16_1::ZERO.bits
    }
}

impl One for Posit16_1 {
    fn one() -> Self {
        <Posit16_1 as From<f32>>::from(1.0)
    }
}

impl Num for Posit16_1 {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f32::from_str_radix(s, radix).map(<Posit16_1 as From<f32>>::from)
    }
}

impl ToPrimitive for Posit16_1 {
    fn to_i64(&self) -> Option<i64> {
        <f32 as From<Posit16_1>>::from(*self).to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        <f32 as From<Posit16_1>>::from(*self).to_u64()
    }
    fn to_f64(&self) -> Option<f64> {
        <f32 as From<Posit16_1>>::from(*self).to_f64()
    }
}

impl ScalarOperand for Posit16_1 {}

impl AIFloat for Posit16_1 {
    fn exp(self) -> Self {
        const LN2_F64: f64 = core::f64::consts::LN_2;
        // max arg is ln(maxpos) = ln(2^14) approx 9.7
        const MAX_EXP_ARG: Posit16_1 = Posit16_1 { bits: 0x6F5B }; // from(9.704)
        // min arg is ln(minpos) approx -9.7
        const MIN_EXP_ARG: Posit16_1 = Posit16_1 { bits: 0x90A5 }; // from(-9.704)

        if self.bits == Self::NAR.bits {
            return Self::NAR;
        } else if self.is_zero() {
            return <Self as From<f32>>::from(1.0);
        } else if self > MAX_EXP_ARG {
            return Posit16_1::new(0x7FFF); // maxpos
        } else if self < MIN_EXP_ARG {
            return Self::ZERO;
        }

        let self_f64: f64 = <f32 as From<Posit16_1>>::from(self) as f64;
        let k = (self_f64 / LN2_F64).round() as i32;
        let r_f64 = self_f64 - (k as f64 * LN2_F64);
        let r: Posit16_1 = <Self as From<f32>>::from(r_f64 as f32);

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
    // Import the 16-bit softposit implementation for reference
    use softposit::P16;

    /// Helper function to compare your f32 -> posit conversion against the reference.
    fn validate_f32_to_posit(value: f32) {
        let my_posit = <Posit16_1 as From<f32>>::from(value);
        let reference_posit = P16::from_f32(value);

        if my_posit.bits != reference_posit.to_bits() {
            println!("Input f32: {}", value);
            println!("My posit:    {:#018b} ({})", my_posit.bits, my_posit.bits);
            println!(
                "Ref posit:   {:#018b} ({})",
                reference_posit.to_bits(),
                reference_posit.to_bits()
            );
        }

        assert_eq!(
            my_posit.bits,
            reference_posit.to_bits(),
            "f32 -> posit conversion mismatch for {}",
            value
        );
    }

    #[test]
    fn test_f32_to_posit() {
        let test_cases = [
            // Special values
            0.0,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            // Powers of 2
            1.0,
            2.0,
            4.0,
            0.5,   // 2^-1
            0.25,  // 2^-2
            0.125, // 2^-3
            -1.0,
            -2.0,
            -0.5,
            // Numbers with fractions
            1.5,
            3.14159,
            -2.71828,
            123.456,
            // Large and small values within posit<16,1> range
            1.0e3,
            -1.0e3,
            1.0e-4,
            -1.0e-4,
            // f32 subnormal value
            f32::from_bits(0x00400001),
            // maxpos/minpos for posit<16,1>
            16384.0,       // maxpos (2^14)
            -16384.0,      // -maxpos
            6.1035156e-5,  // minpos (2^-14)
            -6.1035156e-5, // -minpos
        ];

        for &value in test_cases.iter() {
            validate_f32_to_posit(value);
        }
    }

    /// Helper function to compare your posit -> f32 conversion against the reference.
    fn validate_posit_to_f32(p_bits: u16) {
        let my_posit = Posit16_1::new(p_bits);
        let my_f32: f32 = my_posit.into();

        let reference_posit = P16::from_bits(p_bits);
        let reference_f32 = reference_posit.to_f32();

        if my_f32.is_nan() && reference_f32.is_nan() {
            return;
        }

        if my_f32.to_bits() != reference_f32.to_bits() {
            println!("Input bits:  {:#018b} ({})", p_bits, p_bits);
            println!("My f32:      {} ({:#010x})", my_f32, my_f32.to_bits());
            println!(
                "Ref f32:     {} ({:#010x})",
                reference_f32,
                reference_f32.to_bits()
            );
        }

        assert_eq!(
            my_f32.to_bits(),
            reference_f32.to_bits(),
            "posit -> f32 conversion mismatch for bits {:#x}",
            p_bits
        );
    }

    #[test]
    fn test_posit_to_f32() {
        let bit_patterns = [
            0x0000, // Zero
            0x8000, // NaR
            0x4000, // 1.0
            0xC000, // -1.0
            0x2C00, // 0.25
            0x7FFF, // maxpos
            0x0001, // minpos
            0xFFFF, // -minpos
            0x8001, // -maxpos
            0x5000, // 4.0
            0x2000, // 2^-4
        ];

        for &bits in bit_patterns.iter() {
            validate_posit_to_f32(bits);
        }
    }

    fn validate_add(v1: f32, v2: f32) {
        let p1 = <Posit16_1 as From<f32>>::from(v1);
        let p2 = <Posit16_1 as From<f32>>::from(v2);
        let my_sum = p1 + p2;

        let ref_p1 = P16::from_f32(v1);
        let ref_p2 = P16::from_f32(v2);
        let ref_sum = ref_p1 + ref_p2;

        if my_sum.bits != ref_sum.to_bits() {
            println!("\nAddition test failed for {} + {}", v1, v2);
            println!("My Posits:    {:#018b} + {:#018b}", p1.bits, p2.bits);
            println!("My Sum:       {:#018b}", my_sum.bits);
            println!("Ref Sum:      {:#018b}", ref_sum.to_bits());
        }

        assert_eq!(my_sum.bits, ref_sum.to_bits());
    }

    #[test]
    fn test_posit_addition() {
        validate_add(1.0, 2.0);
        validate_add(1.5, 0.5);
        validate_add(123.45, 1.0);
        validate_add(10.0, -3.0);
        validate_add(1.0, -10.0);
        validate_add(0.125, 0.125);
        validate_add(1.0, 1.0e-4);
        validate_add(1000.0, 1000.0);
        validate_add(-1.0, -5.0);
        validate_add(-4.0, 8.0);
    }

    fn validate_mul(v1: f32, v2: f32) {
        let p1 = <Posit16_1 as From<f32>>::from(v1);
        let p2 = <Posit16_1 as From<f32>>::from(v2);
        let my_product = p1 * p2;

        let ref_p1 = P16::from_f32(v1);
        let ref_p2 = P16::from_f32(v2);
        let ref_product = ref_p1 * ref_p2;

        if my_product.bits != ref_product.to_bits() {
            println!("\nMultiplication test failed for {} * {}", v1, v2);
            println!("My Posits:    {:#018b} * {:#018b}", p1.bits, p2.bits);
            println!("My Product:   {:#018b}", my_product.bits);
            println!("Ref Product:  {:#018b}", ref_product.to_bits());
        }

        assert_eq!(
            my_product.bits,
            ref_product.to_bits(),
            "Multiplication mismatch for {} * {}",
            v1,
            v2
        );
    }

    #[test]
    fn test_posit_multiplication() {
        validate_mul(2.0, 3.0);
        validate_mul(123.45, 0.0);
        validate_mul(123.45, 1.0);
        validate_mul(123.45, -1.0);
        validate_mul(4.0, 2.0);
        validate_mul(4.0, -2.0);
        validate_mul(-4.0, 2.0);
        validate_mul(-4.0, -2.0);
        validate_mul(0.5, 0.25);
        validate_mul(1.5, 1.5);
        validate_mul(16.0, 0.125);
        validate_mul(100.0, 100.0); // 10000
        validate_mul(200.0, 100.0); // Should overflow to maxpos
        validate_mul(1.0e-3, 1.0e-3); // Should underflow
        validate_mul(123.45, f32::NAN);
    }

    fn validate_sub(v1: f32, v2: f32) {
        let p1 = <Posit16_1 as From<f32>>::from(v1);
        let p2 = <Posit16_1 as From<f32>>::from(v2);
        let my_diff = p1 - p2;

        let ref_p1 = P16::from_f32(v1);
        let ref_p2 = P16::from_f32(v2);
        let ref_diff = ref_p1 - ref_p2;

        assert_eq!(
            my_diff.bits,
            ref_diff.to_bits(),
            "Subtraction mismatch for {} - {}",
            v1,
            v2
        );
    }

    #[test]
    fn test_posit_subtraction() {
        validate_sub(10.0, 3.0);
        validate_sub(3.0, 10.0);
        validate_sub(-10.0, 3.0);
        validate_sub(10.0, -3.0);
        validate_sub(123.45, 123.45);
    }

    fn validate_div(v1: f32, v2: f32) {
        let p1 = <Posit16_1 as From<f32>>::from(v1);
        let p2 = <Posit16_1 as From<f32>>::from(v2);
        let my_quot = p1 / p2;

        let ref_p1 = P16::from_f32(v1);
        let ref_p2 = P16::from_f32(v2);
        let ref_quot = ref_p1 / ref_p2;

        if my_quot.bits != ref_quot.to_bits() {
            println!("\nDivision test failed for {} / {}", v1, v2);
            println!("My Posits:    {:#018b} / {:#018b}", p1.bits, p2.bits);
            println!("My Quotient:  {:#018b}", my_quot.bits);
            println!("Ref Quotient: {:#018b}", ref_quot.to_bits());
        }

        assert_eq!(
            my_quot.bits,
            ref_quot.to_bits(),
            "Division mismatch for {} / {}",
            v1,
            v2
        );
    }

    #[test]
    fn test_posit_division() {
        validate_div(6.0, 3.0);
        validate_div(123.45, 1.0);
        validate_div(123.45, -1.0);
        validate_div(0.0, 123.45);
        validate_div(1.0, 4.0);
        validate_div(7.5, 2.5);
        validate_div(1000.0, 10.0);
        validate_div(1.0, 1000.0);
        validate_div(123.45, f32::NAN);
        validate_div(f32::NAN, 123.45);
        validate_div(123.45, 0.0);
    }

    fn validate_rem(v1: f32, v2: f32) {
        let p1 = <Posit16_1 as From<f32>>::from(v1);
        let p2 = <Posit16_1 as From<f32>>::from(v2);
        let my_rem = p1 % p2;

        // The definition of remainder used is: `a % b = a - trunc(a / b) * b`
        // We calculate the expected result using the same definition with posits.
        let div_res = p1 / p2;
        let trunc_div = div_res.trunc();
        let expected_rem = p1 - (trunc_div * p2);

        if my_rem.bits != expected_rem.bits {
            println!("\nRemainder test failed for {} % {}", v1, v2);
            println!(
                "My Rem:       {:#018b} ({})",
                my_rem.bits,
                <f32 as From<Posit16_1>>::from(my_rem)
            );
            println!(
                "Expected Rem: {:#018b} ({})",
                expected_rem.bits,
                <f32 as From<Posit16_1>>::from(expected_rem)
            );
        }

        assert_eq!(
            my_rem.bits, expected_rem.bits,
            "Remainder mismatch for {} % {}",
            v1, v2
        );
    }

    #[test]
    fn test_posit_remainder() {
        validate_rem(10.0, 3.0);
        validate_rem(10.0, -3.0);
        validate_rem(-10.0, 3.0);
        validate_rem(-10.0, -3.0);
        validate_rem(5.5, 2.0);
        validate_rem(5.5, 1.5);
        validate_rem(1.0, 0.6);
        validate_rem(123.45, 10.0);
        validate_rem(10.0, 10.0);
        validate_rem(3.0, 10.0);
        validate_rem(10.0, 1.0);

        // Special values
        let nar = Posit16_1::NAR;
        let zero = Posit16_1::ZERO;
        let p10 = <Posit16_1 as From<f32>>::from(10.0);

        assert_eq!((p10 % zero).bits, nar.bits, "10 % 0 should be NaR");
        assert_eq!((p10 % nar).bits, nar.bits, "10 % NaR should be NaR");
        assert_eq!((nar % p10).bits, nar.bits, "NaR % 10 should be NaR");
        assert_eq!((zero % p10).bits, zero.bits, "0 % 10 should be 0");
    }

    fn validate_neg(v: f32) {
        let p = <Posit16_1 as From<f32>>::from(v);
        let my_neg = -p;

        let ref_p = P16::from_f32(v);
        let ref_neg = -ref_p;

        if my_neg.bits != ref_neg.to_bits() {
            println!("\nNegation test failed for {}", v);
            println!("My Posit:    {:#018b}", p.bits);
            println!("My Negation: {:#018b}", my_neg.bits);
            println!("Ref Negation:{:#018b}", ref_neg.to_bits());
        }

        assert_eq!(
            my_neg.bits,
            ref_neg.to_bits(),
            "Negation mismatch for {}",
            v
        );
    }

    #[test]
    fn test_posit_negation() {
        validate_neg(3.0);
        validate_neg(-3.0);
        validate_neg(123.45);
        validate_neg(0.125);
        validate_neg(0.0);
        validate_neg(f32::NAN);
        validate_neg(16384.0); // maxpos
        validate_neg(6.1035156e-5); // minpos
    }

    fn validate_powi(v: f32, n: i32) {
        let p = <Posit16_1 as From<f32>>::from(v);
        let my_pow = p.powi(n);

        let expected_f32 = v.powi(n);
        // Handle cases where f32 calculation goes out of posit range
        let expected_pow = if expected_f32.is_nan() {
            Posit16_1::NAR
        } else {
            <Posit16_1 as From<f32>>::from(expected_f32)
        };

        if my_pow.bits != expected_pow.bits {
            println!("\n powi test failed for {}^{}", v, n);
            println!(
                "My pow:       {:#018b} ({})",
                my_pow.bits,
                <f32 as From<Posit16_1>>::from(my_pow)
            );
            println!("Expected pow: {:#018b} ({})", expected_pow.bits, v.powi(n));
        }

        assert_eq!(
            my_pow.bits, expected_pow.bits,
            "powi mismatch for {}^{}",
            v, n
        );
    }

    #[test]
    fn test_posit_powi() {
        validate_powi(2.0, 3);
        validate_powi(3.0, 2);
        validate_powi(4.0, 0);
        validate_powi(10.0, 1);
        validate_powi(4.0, -1);
        validate_powi(2.0, -3);
        validate_powi(-2.0, 3);
        validate_powi(-2.0, 2);
        validate_powi(0.5, 2);
        validate_powi(0.0, 2);
        validate_powi(0.0, -2); // Should be NaR
    }

    fn validate_max(v1: f32, v2: f32) {
        let p1 = <Posit16_1 as From<f32>>::from(v1);
        let p2 = <Posit16_1 as From<f32>>::from(v2);
        let my_max = AIFloat::max(p1, p2);

        // Handle NaN according to the function's documentation
        let expected_max = if v1.is_nan() {
            p2
        } else if v2.is_nan() {
            p1
        } else {
            <Posit16_1 as From<f32>>::from(v1.max(v2))
        };

        assert_eq!(
            my_max.bits, expected_max.bits,
            "max mismatch for max({}, {})",
            v1, v2
        );
    }

    #[test]
    fn test_posit_max() {
        validate_max(10.0, 3.0);
        validate_max(3.0, 10.0);
        validate_max(-10.0, -3.0);
        validate_max(-3.0, -10.0);
        validate_max(10.0, -3.0);
        validate_max(-10.0, 3.0);
        validate_max(123.45, 123.45);
        validate_max(123.0, f32::NAN);
        validate_max(f32::NAN, 123.0);
    }

    fn validate_min(v1: f32, v2: f32) {
        let p1 = <Posit16_1 as From<f32>>::from(v1);
        let p2 = <Posit16_1 as From<f32>>::from(v2);
        let my_min = AIFloat::min(p1, p2);

        let expected_min = if v1.is_nan() {
            p2
        } else if v2.is_nan() {
            p1
        } else {
            <Posit16_1 as From<f32>>::from(v1.min(v2))
        };

        assert_eq!(
            my_min.bits, expected_min.bits,
            "min mismatch for min({}, {})",
            v1, v2
        );
    }

    #[test]
    fn test_posit_min() {
        validate_min(10.0, 3.0);
        validate_min(3.0, 10.0);
        validate_min(-10.0, -3.0);
        validate_min(-3.0, -10.0);
        validate_min(10.0, -3.0);
        validate_min(-10.0, 3.0);
        validate_min(123.45, 123.45);
        validate_min(123.0, f32::NAN);
        validate_min(f32::NAN, 123.0);
    }

    fn validate_abs(v: f32) {
        let p = <Posit16_1 as From<f32>>::from(v);
        let my_abs = p.abs();
        let expected_abs = <Posit16_1 as From<f32>>::from(v.abs());

        if my_abs.bits != expected_abs.bits {
            println!("\nabs test failed for {}", v);
            println!(
                "My abs:       {:#018b} ({})",
                my_abs.bits,
                <f32 as From<Posit16_1>>::from(my_abs)
            );
            println!("Expected abs: {:#018b} ({})", expected_abs.bits, v.abs());
        }

        assert_eq!(my_abs.bits, expected_abs.bits, "abs mismatch for {}", v);
    }

    #[test]
    fn test_posit_abs() {
        validate_abs(1.0);
        validate_abs(123.45);
        validate_abs(-1.0);
        validate_abs(-123.45);
        validate_abs(0.0);
        validate_abs(-0.0);
        validate_abs(16384.0); // maxpos
        validate_abs(-16384.0); // -maxpos

        let nar = Posit16_1::NAR;
        assert_eq!(nar.abs().bits, nar.bits, "abs(NaR) should be NaR");
    }

    fn validate_round(v: f32) {
        let p = <Posit16_1 as From<f32>>::from(v);
        let my_round = p.round();

        // The old test was flawed. We must round the actual, low-precision
        // value that the posit represents, not the original f32.
        let p_as_f32 = <f32 as From<Posit16_1>>::from(p);
        let expected_f32 = p_as_f32.round();
        let expected_round = <Posit16_1 as From<f32>>::from(expected_f32 as f32);

        if my_round.bits != expected_round.bits {
            println!("\nround test failed for {}", v);
            println!(
                "My round:     {:#018b} ({})",
                my_round.bits,
                <f32 as From<Posit16_1>>::from(my_round)
            );
            println!(
                "Expected:     {:#018b} ({})",
                expected_round.bits,
                <f32 as From<Posit16_1>>::from(expected_round)
            );
        }
        assert_eq!(
            my_round.bits,
            expected_round.bits,
            "round mismatch for {}. Got {}, expected {}",
            v,
            <f32 as From<Posit16_1>>::from(my_round),
            <f32 as From<Posit16_1>>::from(expected_round)
        );
    }

    #[test]
    fn test_posit_round() {
        validate_round(3.2);
        validate_round(3.8);
        validate_round(3.5);
        validate_round(3.499);
        validate_round(-3.2);
        validate_round(-3.8);
        validate_round(-3.5);
        validate_round(-3.499);
        validate_round(0.4);
        validate_round(0.6);
        validate_round(-0.4);
        validate_round(-0.6);
        validate_round(5.0);
        validate_round(-5.0);
        validate_round(0.0);
        validate_round(123.45);
        validate_round(123.89);

        let nar = Posit16_1::NAR;
        assert_eq!(nar.round().bits, nar.bits, "round(NaR) should be NaR");
    }

    #[test]
    fn test_posit_ordering() {
        let p_minus_10 = <Posit16_1 as From<f32>>::from(-10.0);
        let p_minus_2 = <Posit16_1 as From<f32>>::from(-2.0);
        let p_zero = <Posit16_1 as From<f32>>::from(0.0);
        let p_one = <Posit16_1 as From<f32>>::from(1.0);
        let p_five = <Posit16_1 as From<f32>>::from(5.0);
        let nar = Posit16_1::NAR;

        assert!(p_five == <Posit16_1 as From<f32>>::from(5.0));
        assert!(p_five != p_one);
        assert!(!(nar == nar)); // NaR is not equal to itself
        assert!(p_five != nar);
        assert!(p_one < p_five);
        assert!(!(p_one > p_five));
        assert!(p_minus_2 > p_minus_10);
        assert!(!(p_five < nar));
        assert!(!(p_five > nar));

        let mut values = vec![p_five, p_minus_10, p_zero, p_one, p_minus_2];
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let expected_order = vec![p_minus_10, p_minus_2, p_zero, p_one, p_five];

        for i in 0..values.len() {
            assert_eq!(
                values[i].bits, expected_order[i].bits,
                "Sort order mismatch at index {}",
                i
            );
        }
    }

    fn validate_trunc(v: f32) {
        let p = <Posit16_1 as From<f32>>::from(v);
        let my_trunc = p.trunc();
        let expected_trunc = <Posit16_1 as From<f32>>::from(v.trunc());

        if my_trunc.bits != expected_trunc.bits {
            println!("\ntrunc test failed for {}", v);
            println!(
                "My trunc:     {:#018b} ({})",
                my_trunc.bits,
                <f32 as From<Posit16_1>>::from(my_trunc)
            );
            println!(
                "Expected:     {:#018b} ({})",
                expected_trunc.bits,
                v.trunc()
            );
        }

        assert_eq!(
            my_trunc.bits, expected_trunc.bits,
            "trunc mismatch for {}",
            v
        );
    }

    #[test]
    fn test_posit_trunc() {
        validate_trunc(3.2);
        validate_trunc(3.8);
        validate_trunc(3.5);
        validate_trunc(-3.2);
        validate_trunc(-3.8);
        validate_trunc(-3.5);
        validate_trunc(0.999);
        validate_trunc(0.1);
        validate_trunc(-0.999);
        validate_trunc(-0.1);
        validate_trunc(5.0);
        validate_trunc(-5.0);
        validate_trunc(0.0);

        let nar = Posit16_1::NAR;
        assert_eq!(nar.trunc().bits, nar.bits, "trunc(NaR) should be NaR");

        let maxpos_p = Posit16_1::new(0x7FFF);
        let minpos_p = Posit16_1::new(0x0001);
        assert_eq!(
            maxpos_p.trunc().bits,
            maxpos_p.bits,
            "trunc(maxpos) should be maxpos"
        );
        assert_eq!(
            minpos_p.trunc().bits,
            Posit16_1::ZERO.bits,
            "trunc(minpos) should be zero"
        );
    }
}
