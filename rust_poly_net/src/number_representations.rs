use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use ndarray::ScalarOperand;
use rust_poly_net::{AIFloat, MlScalar};

#[derive(Debug, Clone, Copy)]
pub struct PositV1<const N: usize, const ES: usize> {
    bits: u32,
}

#[derive(Debug)]
struct DecodedPositV1 {
    is_nar: bool,
    is_zero: bool,
    sign: u32,
    scale: i32,
    // Use a large integer to hold the fraction with its implicit '1' bit
    // and provide headroom for shifting and rounding.
    mantissa: u64,
    frac_len: u32,
}

#[derive(Debug)]
struct UnpackedPositV1 {
    sign: u32,
    scale: i32,
    mantissa: i128, // Using i128 to hold the large intermediate result
}

impl<const N: usize, const ES: usize> PositV1<N, ES> {
    // const BITS: usize = N;
    const ES: usize = ES;
    // const USEED: u32 = (1 << (1 << ES));

    pub const ZERO: Self = PositV1 { bits: 0 };
    pub const NAR: Self = PositV1 { bits: 1 << (N - 1) };

    pub fn new(bits: u32) -> Self {
        PositV1 { bits }
    }
}

pub type Posit32 = PositV1<32, 2>;

impl Posit32 {
    fn decode(&self) -> DecodedPositV1 {
        // *** FIX 1: Correctly check for ZERO, not NAR twice ***
        if self.bits == Self::ZERO.bits {
            return DecodedPositV1 {
                is_nar: false,
                is_zero: true,
                sign: 0,
                scale: 0,
                mantissa: 0,
                frac_len: 0,
            };
        }
        if self.bits == Self::NAR.bits {
            return DecodedPositV1 {
                is_nar: true,
                is_zero: false,
                sign: 0,
                scale: 0,
                mantissa: 0,
                frac_len: 0,
            };
        }

        let sign = self.bits >> 31;
        let abs_bits = if sign == 1 {
            (-(self.bits as i32)) as u32
        } else {
            self.bits
        };

        let body = abs_bits << 1;
        let regime_msb = body >> 31;

        let k: i32;
        let regime_len: u32;

        if regime_msb == 1 {
            let num_ones = body.leading_ones();
            if num_ones >= 31 {
                // maxpos
                return DecodedPositV1 {
                    is_nar: false,
                    is_zero: false,
                    sign,
                    scale: 120,
                    mantissa: 1,
                    frac_len: 0,
                };
            }
            regime_len = num_ones + 1;
            k = num_ones as i32 - 1;
        } else {
            let num_zeros = body.leading_zeros();
            if num_zeros >= 30 {
                // minpos
                return DecodedPositV1 {
                    is_nar: false,
                    is_zero: false,
                    sign,
                    scale: -120,
                    mantissa: 1,
                    frac_len: 0,
                };
            }
            regime_len = num_zeros + 1;
            k = -(num_zeros as i32);
        }

        let scale_base = k * (1 << Self::ES);
        let remaining_len = 31 - regime_len;

        let es_len = std::cmp::min(Self::ES as u32, remaining_len);
        let frac_len = remaining_len - es_len;

        let exp_frac_bits = (body << regime_len) >> (32 - remaining_len);

        let es_val = if es_len > 0 {
            (exp_frac_bits >> frac_len) as i32
        } else {
            0
        };

        let scale = scale_base + es_val;

        let frac_mask = if frac_len > 0 {
            (1u64 << frac_len) - 1
        } else {
            0
        };
        let frac_bits = (exp_frac_bits as u64) & frac_mask;
        let mantissa = (1u64 << frac_len) | frac_bits;

        DecodedPositV1 {
            is_nar: false,
            is_zero: false,
            sign,
            scale,
            mantissa,
            frac_len,
        }
    }

    fn encode(unpacked: UnpackedPositV1) -> Self {
        if unpacked.mantissa == 0 {
            return Self::ZERO;
        }

        let result_sign = unpacked.sign;
        let mantissa = unpacked.mantissa;

        // STEP 1: NORMALIZE the unpacked mantissa
        // Find the bit position of the most significant bit.
        let msb_pos = 127 - mantissa.leading_zeros() as i32;

        // The value is `mantissa * 2^scale`.
        // We can rewrite this as `(mantissa / 2^msb_pos) * 2^(scale + msb_pos)`.
        // The term `(mantissa / 2^msb_pos)` is our normalized `1.f` mantissa.
        // The term `scale + msb_pos` is our final, effective scale.
        let final_scale = unpacked.scale + msb_pos;

        // STEP 2: CALCULATE POSIT COMPONENTS
        let useed_power = (1 << Self::ES) as i32;
        let k = final_scale.div_euclid(useed_power);
        let es_val = (final_scale - k * useed_power) as u32;

        let regime_len = if k >= 0 { k as u32 + 2 } else { -k as u32 + 1 };

        if regime_len >= 31 {
            let body = if k >= 0 { 0x7FFFFFFF } else { 1 };
            let bits = if result_sign == 1 {
                (-(body as i32)) as u32
            } else {
                body
            };
            return Self::new(bits);
        }

        let regime_body = if k >= 0 {
            let num_ones = k as u32 + 1;
            (!0u32 << (32 - num_ones)) >> 1
        } else {
            1 << (31 - regime_len)
        };

        let available_len = 31 - regime_len;
        let es_len = std::cmp::min(Self::ES as u32, available_len);
        let frac_len = available_len - es_len;

        // STEP 3: PERFORM ROUNDING
        // Get the fractional part of the mantissa (all bits below the MSB).
        let shift_for_round = msb_pos - frac_len as i32;
        if shift_for_round > 128 {
            return Self::ZERO;
        }

        let mut frac_plus_round = if shift_for_round > 0 {
            mantissa >> (shift_for_round - 1)
        } else {
            mantissa << (1 - shift_for_round)
        };

        // Round-to-nearest-even
        if (frac_plus_round & 1) != 0 {
            if (frac_plus_round & 2) != 0
                || (mantissa & ((1i128 << (shift_for_round - 1)) - 1)) != 0
            {
                frac_plus_round += 2;
            }
        }

        // STEP 4: ASSEMBLE FINAL BITS
        let frac_bits = (frac_plus_round >> 1) as u32 & ((1 << frac_len) - 1);
        let es_bits = es_val << frac_len;
        let payload_mask = (1 << available_len) - 1;
        let body = regime_body | ((es_bits | frac_bits) & payload_mask);

        let final_bits = if result_sign == 1 {
            (-(body as i32)) as u32
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

        if d.scale >= d.frac_len as i32 {
            return self;
        }

        let bits_to_chop = d.frac_len as i32 - d.scale;

        let trunc_mantissa = (d.mantissa >> bits_to_chop) << bits_to_chop;

        let unpacked = UnpackedPositV1 {
            sign: d.sign,
            scale: d.scale - d.frac_len as i32,
            mantissa: trunc_mantissa as i128,
        };

        Self::encode(unpacked)
    }

    pub fn abs(self) -> Self {
        if self.bits == Self::NAR.bits {
            return Self::NAR;
        }

        let is_negative = (self.bits >> 31) == 1;
        if is_negative { -self } else { self }
    }

    pub fn round(self) -> Self {
        if self.bits == Self::NAR.bits {
            return Self::NAR;
        }

        // Use the definition round(x) = trunc(x + 0.5) for x>=0 and round(x) = trunc(x - 0.5) for x<0
        let half = <Self as From<f32>>::from(0.5);
        let is_negative = (self.bits >> 31) == 1 && self.bits != Self::ZERO.bits;

        if is_negative {
            (self - half).trunc()
        } else {
            (self + half).trunc()
        }
    }
}

impl From<f32> for Posit32 {
    fn from(value: f32) -> Self {
        if value == 0.0 {
            return Self::ZERO;
        }
        if value.is_nan() || value.is_infinite() {
            return Self::NAR;
        }

        let input_bits = value.to_bits();
        let sign = input_bits >> 31;

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

        let body: u32;

        if regime_len >= 31 {
            body = if k >= 0 { 0x7FFFFFFF } else { 1 };
        } else {
            let regime_body = if k >= 0 {
                let num_ones = k as u32 + 1;
                (!0u32 << (32 - num_ones)) >> 1
            } else {
                1 << (31 - regime_len)
            };

            let available_len = 31 - regime_len;
            let es_len = std::cmp::min(Self::ES as u32, available_len);
            let frac_len = available_len - es_len;

            let es_payload = (es_val as u32) << frac_len;
            let frac_payload: u32;

            let shift = 23 - (frac_len as i32);

            if shift >= 0 {
                // Right shift: posit has fewer/equal fraction bits, requires rounding.
                let f32_frac_bits = frac as u64;
                let mut temp_payload = f32_frac_bits >> shift;

                let guard_mask = if shift > 0 { 1u64 << (shift - 1) } else { 0 };
                let guard_bit = (f32_frac_bits & guard_mask) != 0;
                let sticky_mask = guard_mask.wrapping_sub(1);
                let sticky_bit = (f32_frac_bits & sticky_mask) != 0;

                if guard_bit && (sticky_bit || (temp_payload & 1) == 1) {
                    temp_payload += 1;
                }
                frac_payload = temp_payload as u32;
            } else {
                // Left shift: posit has more fraction bits, no rounding needed.
                frac_payload = frac << -shift;
            }

            // *** THIS IS THE MAIN FIX ***
            // The final payload combines the exponent and the (masked) fraction.
            // The mask prevents the fraction from corrupting the exponent bits.
            let frac_mask = if frac_len >= 32 {
                u32::MAX
            } else {
                (1 << frac_len) - 1
            };
            let payload = es_payload | (frac_payload & frac_mask);

            body = regime_body | payload;
        }

        let final_bits = if sign == 1 {
            (-(body as i32)) as u32
        } else {
            body
        };

        Self::new(final_bits)
    }
}

impl From<Posit32> for f32 {
    fn from(p: Posit32) -> Self {
        if p.bits == Posit32::ZERO.bits {
            return 0.0;
        }
        if p.bits == Posit32::NAR.bits {
            return f32::NAN;
        }

        let sign = p.bits >> 31;
        let abs_bits = if sign == 1 {
            (-(p.bits as i32)) as u32
        } else {
            p.bits
        };

        let body = abs_bits << 1;
        let regime_msb = body >> 31;

        let k: i32;
        let regime_len: u32;

        if regime_msb == 1 {
            let num_ones = body.leading_ones();
            if num_ones >= 31 {
                // maxpos (k=30). E = 30 * 4 = 120.
                // f32 biased exponent is 127 + 120 = 247.
                let bits = (sign << 31) | (247u32 << 23);
                return f32::from_bits(bits);
            }
            regime_len = num_ones + 1;
            k = num_ones as i32 - 1;
        } else {
            let num_zeros = body.leading_zeros();
            // minpos has a body of 0x00000002 (30 leading zeros for N=32).
            if num_zeros >= 30 {
                // *** THIS IS THE FIX ***
                // minpos (k=-30). E = -30 * 4 = -120.
                // f32 biased exponent is 127 - 120 = 7.
                let bits = (sign << 31) | (7u32 << 23);
                return f32::from_bits(bits);
            }
            regime_len = num_zeros + 1;
            k = -(num_zeros as i32);
        }

        let scale = k * (1 << Posit32::ES);
        let remaining_len = 32 - 1 - regime_len;

        let es_len = std::cmp::min(Posit32::ES as u32, remaining_len);
        let frac_len = remaining_len - es_len;

        // We are safe from the shift-overflow panic here because the minpos case (where remaining_len=0) was handled above.
        let exp_frac_bits = (body << regime_len) >> (32 - remaining_len);

        let exp_val = if es_len > 0 {
            let exp_bits = exp_frac_bits >> frac_len;
            scale + exp_bits as i32
        } else {
            scale
        };

        let fraction = if frac_len > 0 {
            let frac_mask = (1 << frac_len) - 1;
            let frac_bits = exp_frac_bits & frac_mask;
            1.0 + (frac_bits as f32 / (1u64 << frac_len) as f32)
        } else {
            1.0
        };

        let final_val = 2.0f64.powi(exp_val) * fraction as f64;

        if sign == 1 {
            -final_val as f32
        } else {
            final_val as f32
        }
    }
}

impl Add for Posit32 {
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

        let mut mant1 = if p1.sign == 1 {
            -(p1.mantissa as i128)
        } else {
            p1.mantissa as i128
        };
        let mut mant2 = if p2.sign == 1 {
            -(p2.mantissa as i128)
        } else {
            p2.mantissa as i128
        };

        mant1 <<= 96 - p1.frac_len;
        mant2 <<= 96 - p2.frac_len;

        let scale_diff = p1.scale - p2.scale;
        let result_scale;
        if scale_diff > 0 {
            result_scale = p1.scale;
            mant2 >>= scale_diff.min(127);
        } else {
            result_scale = p2.scale;
            mant1 >>= (-scale_diff).min(127);
        }

        let result_mant = mant1 + mant2;

        if result_mant == 0 {
            return Self::ZERO;
        }

        // --- THIS IS THE FINAL, CRUCIAL FIX ---
        // The sign of the result depends *only* on the sign of the final mantissa.
        // It has no dependency on the original p1.sign.
        let result_sign = if result_mant < 0 { 1 } else { 0 };

        let unpacked = UnpackedPositV1 {
            sign: result_sign,
            scale: result_scale - 96,
            mantissa: result_mant.abs(),
        };

        Self::encode(unpacked)
    }
}

impl Mul for Posit32 {
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

        // The value of a decoded posit is: mantissa * 2^(scale - frac_len).
        // The product's value is: (m1*m2) * 2^(s1-fl1 + s2-fl2)
        // Which is: (m1*m2) * 2^(s1+s2 - (fl1+fl2)).
        // We pass these components to our generic `encode` function.
        let result_scale = p1.scale + p2.scale - p1.frac_len as i32 - p2.frac_len as i32;
        let result_mant = (p1.mantissa as i128) * (p2.mantissa as i128);

        let unpacked = UnpackedPositV1 {
            sign: p1.sign ^ p2.sign,
            scale: result_scale,
            mantissa: result_mant,
        };

        Self::encode(unpacked)
    }
}

impl Neg for Posit32 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        // NaR and Zero are their own negations.
        if self.bits == Self::NAR.bits || self.bits == Self::ZERO.bits {
            return self;
        }
        // For all other values, negation is the two's complement of the bit pattern.
        Self::new((-(self.bits as i32)) as u32)
    }
}

impl Sub for Posit32 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        // a - b is equivalent to a + (-b)
        self + (-rhs)
    }
}

impl Div for Posit32 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let p1 = self.decode();
        let p2 = rhs.decode();

        // Handle special cases:
        // Anything / NaR = NaR
        // NaR / Anything = NaR
        if p1.is_nar || p2.is_nar {
            return Self::NAR;
        }
        // Anything / Zero = NaR
        if p2.is_zero {
            return Self::NAR;
        }
        // Zero / Anything = Zero
        if p1.is_zero {
            return Self::ZERO;
        }

        // The value of a decoded posit is: mantissa * 2^(scale - frac_len).
        // The quotient's value is: (m1/m2) * 2^(s1-fl1 - (s2-fl2))
        // Which is: (m1/m2) * 2^(s1-s2 - (fl1-fl2)).
        let result_scale = p1.scale - p2.scale - (p1.frac_len as i32 - p2.frac_len as i32);

        // To perform division with fractions, we pre-shift the numerator.
        // Shifting left by 96 bits gives us plenty of precision for the result.
        let result_mant = ((p1.mantissa as i128) << 96) / (p2.mantissa as i128);

        let unpacked = UnpackedPositV1 {
            sign: p1.sign ^ p2.sign,
            // The pre-shift of 96 introduces a scale factor of 2^96. We must subtract
            // this from the final scale to compensate.
            scale: result_scale - 96,
            mantissa: result_mant,
        };

        Self::encode(unpacked)
    }
}

impl Rem for Posit32 {
    type Output = Self;

    /// Follows the formula: `a % b = a - trunc(a / b) * b`
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

impl<const N: usize, const ES: usize> PartialEq for PositV1<N, ES> {
    fn eq(&self, other: &Self) -> bool {
        if self.bits == Self::NAR.bits || other.bits == Self::NAR.bits {
            return false;
        }

        self.bits == other.bits
    }
}

impl<const N: usize, const ES: usize> PartialOrd for PositV1<N, ES> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.bits == Self::NAR.bits || other.bits == Self::NAR.bits {
            return None;
        }

        Some((self.bits as i32).cmp(&(other.bits as i32)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Add this use statement to bring the softposit library into scope for tests
    use softposit::P32;

    /// Helper function to compare your f32->posit conversion against the reference implementation.
    fn validate_f32_to_posit(value: f32) {
        let my_posit = <Posit32 as From<f32>>::from(value);
        let reference_posit = P32::from_f32(value);

        if my_posit.bits != reference_posit.to_bits() {
            println!("Input f32: {}", value);
            println!("My posit:    {:#034b} ({})", my_posit.bits, my_posit.bits);
            println!(
                "Ref posit:   {:#034b} ({})",
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
            // Powers of 2 (simple regimes)
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
            1.5, // 1.1_2 * 2^0
            3.14159,
            -2.71828,
            4323.143, // The original failing case
            // Large and small values
            1.0e10,
            -1.0e10,
            1.0e-10,
            -1.0e-10,
            // f32 subnormal value
            f32::from_bits(0x00400001),
            // maxpos/minpos for posit<32,2>
            2.6815616e36,   // maxpos
            -2.6815616e36,  // -maxpos
            3.7252903e-37,  // minpos
            -3.7252903e-37, // -minpos
        ];

        for &value in test_cases.iter() {
            validate_f32_to_posit(value);
        }
    }

    /// Helper function to compare your posit->f32 conversion against the reference.
    fn validate_posit_to_f32(p_bits: u32) {
        let my_posit = Posit32::new(p_bits);
        let my_f32: f32 = my_posit.into();

        let reference_posit = P32::from_bits(p_bits);
        let reference_f32 = reference_posit.to_f32();

        // When comparing NaNs, their bit patterns can differ.
        // It's sufficient to check that both results are NaN.
        if my_f32.is_nan() && reference_f32.is_nan() {
            return;
        }

        if my_f32.to_bits() != reference_f32.to_bits() {
            println!("Input bits:  {:#034b} ({})", p_bits, p_bits);
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
        // Test a range of interesting bit patterns
        let bit_patterns = [
            0x00000000, // Zero
            0x80000000, // NaR
            0x40000000, // 1.0
            0xC0000000, // -1.0
            0x28000000, // 0.125
            0x780E324A, // 4323.143...
            0x7FFFFFFF, // maxpos
            0x00000001, // minpos
            0xFFFFFFFF, // -minpos
            0x80000001, // -maxpos
            0x60000000, // 4.0
            0x10000000, // 2^-8
        ];

        for &bits in bit_patterns.iter() {
            validate_posit_to_f32(bits);
        }
    }

    fn validate_add(v1: f32, v2: f32) {
        let p1 = <Posit32 as From<f32>>::from(v1);
        let p2 = <Posit32 as From<f32>>::from(v2);
        let my_sum = p1 + p2;

        let ref_p1 = P32::from_f32(v1);
        let ref_p2 = P32::from_f32(v2);
        let ref_sum = ref_p1 + ref_p2;

        if my_sum.bits != ref_sum.to_bits() {
            println!("\nAddition test failed for {} + {}", v1, v2);
            println!("My Posits:    {:#034b} + {:#034b}", p1.bits, p2.bits);
            println!("My Sum:       {:#034b}", my_sum.bits);
            println!("Ref Sum:      {:#034b}", ref_sum.to_bits());
        }

        assert_eq!(my_sum.bits, ref_sum.to_bits());
    }

    #[test]
    fn test_posit_addition() {
        validate_add(1.0, 2.0); // 3.0
        validate_add(1.5, 0.5); // 2.0
        validate_add(4323.143, 1.0);
        validate_add(10.0, -3.0); // 7.0 (subtraction)
        validate_add(1.0, -10.0); // -9.0 (subtraction with sign change)
        validate_add(0.125, 0.125); // 0.25
        validate_add(1.0, 1.0e-10); // Aligning vastly different scales
        validate_add(1.0e20, 1.0e20); // Large numbers
        validate_add(-1.0, -5.0);
        validate_add(-4.0, 8.0);
    }

    fn validate_mul(v1: f32, v2: f32) {
        let p1 = <Posit32 as From<f32>>::from(v1);
        let p2 = <Posit32 as From<f32>>::from(v2);
        let my_product = p1 * p2;

        let ref_p1 = P32::from_f32(v1);
        let ref_p2 = P32::from_f32(v2);
        let ref_product = ref_p1 * ref_p2;

        if my_product.bits != ref_product.to_bits() {
            println!("\nMultiplication test failed for {} * {}", v1, v2);
            println!("My Posits:    {:#034b} * {:#034b}", p1.bits, p2.bits);
            println!("My Product:   {:#034b}", my_product.bits);
            println!("Ref Product:  {:#034b}", ref_product.to_bits());
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
        // === Basic Properties ===
        validate_mul(2.0, 3.0); // 6.0
        validate_mul(4323.143, 0.0); // Multiply by zero
        validate_mul(4323.143, 1.0); // Multiply by one (identity)
        validate_mul(4323.143, -1.0); // Multiply by negative one (sign flip)

        // === Sign combinations ===
        validate_mul(4.0, 2.0); // pos * pos = pos
        validate_mul(4.0, -2.0); // pos * neg = neg
        validate_mul(-4.0, 2.0); // neg * pos = neg
        validate_mul(-4.0, -2.0); // neg * neg = pos

        // === Scale and Fraction Interaction ===
        validate_mul(0.5, 0.25); // 0.125 (multiplying fractions)
        validate_mul(1.5, 1.5); // 2.25 (multiplying mantissas)
        validate_mul(16.0, 0.125); // 2.0 (large scale * small scale)

        // === Edge Cases and Large/Small Numbers ===
        validate_mul(1.0e15, 1.0e15); // Large number multiplication
        validate_mul(1.0e-15, 1.0e-15); // Small number multiplication
        validate_mul(1.0e25, 1.0e25); // Should overflow to maxpos
        validate_mul(1.0e-25, 1.0e-25); // Should underflow towards zero
        validate_mul(f32::from_bits(0x7F7FFFFF), 2.0); // Near f32 max * 2

        // === Special Values ===
        validate_mul(123.45, f32::NAN); // Multiply by NaR
    }

    fn validate_sub(v1: f32, v2: f32) {
        let p1 = <Posit32 as From<f32>>::from(v1);
        let p2 = <Posit32 as From<f32>>::from(v2);
        let my_diff = p1 - p2;

        let ref_p1 = P32::from_f32(v1);
        let ref_p2 = P32::from_f32(v2);
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
        validate_sub(10.0, 3.0); // 7.0
        validate_sub(3.0, 10.0); // -7.0
        validate_sub(-10.0, 3.0); // -13.0
        validate_sub(10.0, -3.0); // 13.0 (This is 10 + 3)
        validate_sub(4323.143, 4323.143); // Should be zero
    }

    fn validate_div(v1: f32, v2: f32) {
        let p1 = <Posit32 as From<f32>>::from(v1);
        let p2 = <Posit32 as From<f32>>::from(v2);
        let my_quot = p1 / p2;

        let ref_p1 = P32::from_f32(v1);
        let ref_p2 = P32::from_f32(v2);
        let ref_quot = ref_p1 / ref_p2;

        if my_quot.bits != ref_quot.to_bits() {
            println!("\nDivision test failed for {} / {}", v1, v2);
            println!("My Posits:    {:#034b} / {:#034b}", p1.bits, p2.bits);
            println!("My Quotient:  {:#034b}", my_quot.bits);
            println!("Ref Quotient: {:#034b}", ref_quot.to_bits());
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
        // === Basic Properties ===
        validate_div(6.0, 3.0); // 2.0
        validate_div(4323.143, 1.0); // Divide by one (identity)
        validate_div(4323.143, -1.0); // Divide by negative one
        validate_div(0.0, 4323.143); // Zero divided by a number

        // === Scale and Fraction Interaction ===
        validate_div(1.0, 4.0); // 0.25
        validate_div(7.5, 2.5); // 3.0
        validate_div(1.0e15, 1.0e5); // Large number division

        // === Edge Cases ===
        validate_div(1.0, 1.0e25); // Should produce a very small number
        validate_div(1.0e25, 1.0e-25); // Should overflow to maxpos

        // === Special Values ===
        validate_div(123.45, f32::NAN); // Divide by NaR
        validate_div(f32::NAN, 123.45); // NaR divided by a number
        validate_div(123.45, 0.0); // Division by zero -> NaR
    }

    fn validate_rem(v1: f32, v2: f32) {
        let p1 = <Posit32 as From<f32>>::from(v1);
        let p2 = <Posit32 as From<f32>>::from(v2);
        let my_rem = p1 % p2; // Using the '%' operator

        let expected_f32 = v1 % v2;
        let ref_rem = P32::from_f32(expected_f32);

        if my_rem.bits != ref_rem.to_bits() {
            println!("\nRemainder test failed for {} % {}", v1, v2);
            println!("My Posits:    {:#034b} % {:#034b}", p1.bits, p2.bits);
            println!(
                "My Rem:       {:#034b} ({})",
                my_rem.bits,
                <f32 as From<Posit32>>::from(my_rem)
            );
            println!(
                "Expected Rem: {:#034b} ({})",
                ref_rem.to_bits(),
                expected_f32
            );
        }

        assert_eq!(
            my_rem.bits,
            ref_rem.to_bits(),
            "Remainder mismatch for {} % {}",
            v1,
            v2
        );
    }

    #[test]
    fn test_posit_remainder() {
        // Standard integer-like cases
        validate_rem(10.0, 3.0); // 1.0
        validate_rem(10.0, -3.0); // 1.0 (sign of result matches dividend)
        validate_rem(-10.0, 3.0); // -1.0
        validate_rem(-10.0, -3.0); // -1.0

        // Floating point cases
        validate_rem(5.5, 2.0); // 1.5
        validate_rem(5.5, 1.5); // 1.0
        validate_rem(1.0, 0.6); // 0.4
        validate_rem(4323.143, 100.0);

        // Edge cases
        validate_rem(10.0, 10.0); // 0.0
        validate_rem(3.0, 10.0); // 3.0
        validate_rem(10.0, 1.0); // 0.0

        // Special values
        let nar = <Posit32 as From<f32>>::from(f32::NAN);
        let zero = <Posit32 as From<f32>>::from(0.0);
        let p10 = <Posit32 as From<f32>>::from(10.0);

        assert_eq!((p10 % zero).bits, nar.bits, "10 % 0 should be NaR");
        assert_eq!((p10 % nar).bits, nar.bits, "10 % NaR should be NaR");
        assert_eq!((nar % p10).bits, nar.bits, "NaR % 10 should be NaR");
        assert_eq!((zero % p10).bits, zero.bits, "0 % 10 should be 0");
    }

    fn validate_neg(v: f32) {
        let p = <Posit32 as From<f32>>::from(v);
        let my_neg = -p;

        let ref_p = P32::from_f32(v);
        let ref_neg = -ref_p;

        if my_neg.bits != ref_neg.to_bits() {
            println!("\nNegation test failed for {}", v);
            println!("My Posit:    {:#034b}", p.bits);
            println!("My Negation: {:#034b}", my_neg.bits);
            println!("Ref Negation:{:#034b}", ref_neg.to_bits());
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
        validate_neg(-3.0); // Negating a negative should return the positive
        validate_neg(4323.143);
        validate_neg(0.125);

        // Special values
        validate_neg(0.0);
        validate_neg(f32::NAN);

        // maxpos and minpos
        validate_neg(f32::from_bits(0x7FFFFFFF)); // approx maxpos
        validate_neg(f32::from_bits(0x00000001)); // approx minpos
    }

    fn validate_exp(v: f32, expected: f32, tolerance: f32) {
        let p: PositV1<32, 2> = <Posit32 as From<f32>>::from(v);
        let my_exp_p = p.exp();
        let my_exp_f32: f32 = my_exp_p.into();

        let diff = (my_exp_f32 - expected).abs();

        if diff > tolerance {
            println!("\nexp_scratch test failed for exp({})", v);
            println!(
                "My result:   {} (posit bits: {:#x})",
                my_exp_f32, my_exp_p.bits
            );
            println!("Expected:    {}", expected);
            println!("Difference:  {}", diff);
        }

        assert!(
            diff <= tolerance,
            "exp_scratch mismatch for exp({}). Got {}, expected {}",
            v,
            my_exp_f32,
            expected
        );
    }

    #[test]
    fn test_posit_exp() {
        // Test key values
        validate_exp(0.0, 1.0, 1e-7);
        validate_exp(1.0, 2.7182818, 1e-6); // e^1
        validate_exp(-1.0, 0.3678794, 1e-6); // e^-1

        // Test reduction path
        validate_exp(3.14159, 23.14069, 1e-4);
        validate_exp(-2.5, 0.082085, 1e-6);

        // Test overflow/underflow
        let maxpos = <f32 as From<Posit32>>::from(Posit32::new(0x7FFFFFFF));
        validate_exp(88.0, maxpos, 1e-7); // Should clamp to maxpos
        validate_exp(-88.0, 0.0, 1e-7); // Should be 0

        // Test special values
        let nar = Posit32::NAR;
        let p_nar = <Posit32 as From<f32>>::from(f32::NAN);
        assert_eq!(p_nar.exp().bits, nar.bits);
    }

    fn validate_powi(v: f32, n: i32) {
        let p = <Posit32 as From<f32>>::from(v);
        let my_pow = p.powi(n);
        let expected_pow = <Posit32 as From<f32>>::from(v.powi(n));

        if my_pow.bits != expected_pow.bits {
            println!("\n powi test failed for {}^{}", v, n);
            println!(
                "My pow:       {:#034b} ({})",
                my_pow.bits,
                <f32 as From<Posit32>>::from(my_pow)
            );
            println!("Expected pow: {:#034b} ({})", expected_pow.bits, v.powi(n));
        }

        assert_eq!(
            my_pow.bits, expected_pow.bits,
            "powi mismatch for {}^{}",
            v, n
        );
    }

    #[test]
    fn test_posit_powi() {
        validate_powi(2.0, 3); // 8.0
        validate_powi(3.0, 2); // 9.0
        validate_powi(4.0, 0); // 1.0
        validate_powi(10.0, 1); // 10.0
        validate_powi(4.0, -1); // 0.25
        validate_powi(2.0, -3); // 0.125
        validate_powi(-2.0, 3); // -8.0
        validate_powi(-2.0, 2); // 4.0
        validate_powi(0.5, 2); // 0.25
    }

    fn validate_max(v1: f32, v2: f32) {
        let p1 = <Posit32 as From<f32>>::from(v1);
        let p2 = <Posit32 as From<f32>>::from(v2);
        let my_max = AIFloat::max(p1, p2);
        let expected_max = <Posit32 as From<f32>>::from(v1.max(v2));

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
        validate_max(4323.143, 4323.143);

        // NaR handling
        validate_max(123.0, f32::NAN); // Should be 123.0
        validate_max(f32::NAN, 123.0); // Should be 123.0
    }

    fn validate_min(v1: f32, v2: f32) {
        let p1 = <Posit32 as From<f32>>::from(v1);
        let p2 = <Posit32 as From<f32>>::from(v2);
        let my_min = AIFloat::min(p1, p2);
        let expected_min = <Posit32 as From<f32>>::from(v1.min(v2));

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
        validate_min(4323.143, 4323.143);

        // NaR handling
        validate_min(123.0, f32::NAN); // Should be 123.0
        validate_min(f32::NAN, 123.0); // Should be 123.0
    }

    fn validate_abs(v: f32) {
        let p = <Posit32 as From<f32>>::from(v);
        let my_abs = p.abs();

        // The expected result is the posit of the f32's absolute value.
        let expected_abs = <Posit32 as From<f32>>::from(v.abs());

        if my_abs.bits != expected_abs.bits {
            println!("\nabs test failed for {}", v);
            println!(
                "My abs:       {:#034b} ({})",
                my_abs.bits,
                <f32 as From<Posit32>>::from(my_abs)
            );
            println!("Expected abs: {:#034b} ({})", expected_abs.bits, v.abs());
        }

        assert_eq!(my_abs.bits, expected_abs.bits, "abs mismatch for {}", v);
    }

    #[test]
    fn test_posit_abs() {
        // Positive numbers
        validate_abs(1.0);
        validate_abs(123.456);

        // Negative numbers
        validate_abs(-1.0);
        validate_abs(-123.456);

        // Zero
        validate_abs(0.0);
        validate_abs(-0.0); // Should be the same as 0.0

        // Min and Max values
        validate_abs(2.6815616e36); // maxpos
        validate_abs(-2.6815616e36); // -maxpos

        // Special value NaR
        let nar = Posit32::NAR;
        assert_eq!(nar.abs().bits, nar.bits, "abs(NaR) should be NaR");
    }

    fn validate_round(v: f32) {
        let p = <Posit32 as From<f32>>::from(v);
        let my_round = p.round();

        // The expected result is the posit of the f32's rounded value.
        let expected_round = <Posit32 as From<f32>>::from(v.round());

        if my_round.bits != expected_round.bits {
            println!("\nround test failed for {}", v);
            println!(
                "My round:     {:#034b} ({})",
                my_round.bits,
                <f32 as From<Posit32>>::from(my_round)
            );
            println!(
                "Expected:     {:#034b} ({})",
                expected_round.bits,
                v.round()
            );
        }

        assert_eq!(
            my_round.bits, expected_round.bits,
            "round mismatch for {}",
            v
        );
    }

    #[test]
    fn test_posit_round() {
        // --- Positive Numbers ---
        validate_round(3.2); // -> 3.0
        validate_round(3.8); // -> 4.0
        validate_round(3.5); // -> 4.0 (round half away from zero)
        validate_round(3.499); // -> 3.0

        // --- Negative Numbers ---
        validate_round(-3.2); // -> -3.0
        validate_round(-3.8); // -> -4.0
        validate_round(-3.5); // -> -4.0 (round half away from zero)
        validate_round(-3.499); // -> -3.0

        // --- Numbers close to zero ---
        validate_round(0.4); // -> 0.0
        validate_round(0.6); // -> 1.0
        validate_round(-0.4); // -> 0.0
        validate_round(-0.6); // -> -1.0

        // --- Integers (should not change) ---
        validate_round(5.0);
        validate_round(-5.0);
        validate_round(0.0);

        // --- Large Numbers ---
        validate_round(4323.143);
        validate_round(4323.89);

        // Special value NaR
        let nar = Posit32::NAR;
        assert_eq!(nar.round().bits, nar.bits, "round(NaR) should be NaR");
    }

    #[test]
    fn test_posit_ordering() {
        let p_minus_10 = <Posit32 as From<f32>>::from(-10.0);
        let p_minus_2 = <Posit32 as From<f32>>::from(-2.0);
        let p_zero = <Posit32 as From<f32>>::from(0.0);
        let p_one = <Posit32 as From<f32>>::from(1.0);
        let p_five = <Posit32 as From<f32>>::from(5.0);
        let nar = Posit32::NAR;

        // --- Test PartialEq (this part is correct) ---
        assert!(p_five == <Posit32 as From<f32>>::from(5.0));
        assert!(p_five != p_one);
        assert!(nar != nar);
        assert!(p_five != nar);

        // --- Test PartialOrd (`<`, `>`) (this part is correct) ---
        assert!(p_one < p_five);
        assert!(!(p_one < nar)); // Correctly returns false for NaR

        // --- Test Sorting using .sort_by() ---
        let mut values = vec![p_five, nar, p_minus_10, p_zero, p_one, p_minus_2];

        // THIS IS THE FIX: Use .sort_by() with a custom comparator
        values.sort_by(|a, b| {
            // This is the total ordering logic from your old `Ord` trait.
            // We define NaR to be the greatest value for sorting purposes.
            match (a.bits == Posit32::NAR.bits, b.bits == Posit32::NAR.bits) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => (a.bits as i32).cmp(&(b.bits as i32)),
            }
        });

        let expected_order = vec![p_minus_10, p_minus_2, p_zero, p_one, p_five, nar];

        assert_eq!(
            values.len(),
            expected_order.len(),
            "Sorted vec has wrong length"
        );

        for i in 0..values.len() {
            assert_eq!(
                values[i].bits, expected_order[i].bits,
                "Sort order mismatch at index {}. Got {:?}, expected {:?}",
                i, values, expected_order
            );
        }
    }

    fn validate_trunc(v: f32) {
        let p = <Posit32 as From<f32>>::from(v);
        let my_trunc = p.trunc();

        // The expected result is the posit of the f32's truncated value.
        let expected_trunc = <Posit32 as From<f32>>::from(v.trunc());

        if my_trunc.bits != expected_trunc.bits {
            println!("\ntrunc test failed for {}", v);
            println!(
                "My trunc:     {:#034b} ({})",
                my_trunc.bits,
                <f32 as From<Posit32>>::from(my_trunc)
            );
            println!(
                "Expected:     {:#034b} ({})",
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
        // === Positive Numbers with Fractions ===
        // Should always round towards zero.
        validate_trunc(3.2); // -> 3.0
        validate_trunc(3.8); // -> 3.0
        validate_trunc(3.5); // -> 3.0 (crucially different from round())

        // === Negative Numbers with Fractions ===
        // Should always round towards zero.
        validate_trunc(-3.2); // -> -3.0
        validate_trunc(-3.8); // -> -3.0
        validate_trunc(-3.5); // -> -3.0

        // === The CRITICAL Buggy Case: Numbers between -1 and 1 ===
        // This validates the fix we implemented.
        validate_trunc(0.999); // -> 0.0
        validate_trunc(0.1); // -> 0.0
        validate_trunc(-0.999); // -> -0.0 (which is 0.0)
        validate_trunc(-0.1); // -> -0.0 (which is 0.0)

        // === Integers (should not change) ===
        validate_trunc(5.0);
        validate_trunc(-5.0);
        validate_trunc(0.0);

        // === Special Values ===
        let nar = Posit32::NAR;
        assert_eq!(nar.trunc().bits, nar.bits, "trunc(NaR) should be NaR");

        // === Boundary Values ===
        let maxpos = <f32 as From<Posit32>>::from(Posit32::new(0x7FFFFFFF));
        let minpos = <f32 as From<Posit32>>::from(Posit32::new(0x00000001));
        validate_trunc(maxpos); // maxpos is an integer, should not change.
        validate_trunc(minpos); // minpos is between 0 and 1, should become 0.
    }
}

use num_traits::{Num, NumCast, One, ToPrimitive, Zero};
use std::ops::{AddAssign, SubAssign};
use std::{fmt, result};

// Implement Display for PositV1<32, 2>
impl fmt::Display for PositV1<32, 2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display as f32 for readability
        let val: f32 = (*self).into();
        write!(f, "{}", val)
    }
}

// Implement AddAssign
impl AddAssign for Posit32 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

// Implement SubAssign
impl SubAssign for Posit32 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

// Implement NumCast
impl NumCast for Posit32 {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        n.to_f32().map(<Posit32 as From<f32>>::from)
    }
}

// Implement Zero and One for Num
impl Zero for Posit32 {
    fn zero() -> Self {
        Posit32::ZERO
    }
    fn is_zero(&self) -> bool {
        self.bits == Posit32::ZERO.bits
    }
}

impl One for Posit32 {
    fn one() -> Self {
        <Posit32 as From<f32>>::from(1.0)
    }
}

// Implement Num
impl Num for Posit32 {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f32::from_str_radix(s, radix).map(<Posit32 as From<f32>>::from)
    }
}

impl ToPrimitive for Posit32 {
    fn to_i64(&self) -> Option<i64> {
        <f32 as From<Posit32>>::from(*self).to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        <f32 as From<Posit32>>::from(*self).to_u64()
    }
    fn to_f64(&self) -> Option<f64> {
        <f32 as From<Posit32>>::from(*self).to_f64()
    }
}

// Implement ScalarOperand
impl ScalarOperand for Posit32 {}

// Implement AIFloat (dummy, you may need to fill in methods if required)
impl AIFloat for Posit32 {
    fn exp(self) -> Self {
        const LN2: Posit32 = Posit32 { bits: 1729683968 }; // from(0.6931472)
        const MAX_EXP_ARG: Posit32 = Posit32 { bits: 1765804032 }; // from(84.0)
        const MIN_EXP_ARG: Posit32 = Posit32 { bits: 2528997376 }; // from(-84.0)

        if self.bits == Self::NAR.bits {
            return Self::NAR;
        } else if self.bits == Self::ZERO.bits {
            return <Self as From<f32>>::from(1.0);
        } else if self > MAX_EXP_ARG {
            return Posit32::new(0x7FFF_FFFF);
        } else if self < MIN_EXP_ARG {
            return Self::ZERO;
        }

        let y = self / LN2;
        let k_posit = y.round();
        let k = <f32 as From<Posit32>>::from(k_posit) as i32;
        let r = self - (k_posit * LN2);

        let mut exp_r = <Self as From<f32>>::from(1.0);
        let mut term = <Self as From<f32>>::from(1.0);
        for i in 1..=15 {
            let i_posit = <Self as From<f32>>::from(i as f32);
            term = term * r / i_posit;
            let prev_sum = exp_r;
            exp_r = exp_r + term;
            if exp_r.bits == prev_sum.bits {
                break;
            }
        }

        let scale_factor = <Self as From<f32>>::from(2.0).powi(k);
        let result = exp_r * scale_factor;
        result
    }

    /*fn exp(self) -> Self {
        if self.bits == Self::NAR.bits {
            return Self::NAR;
        }

        let val_f32: f32 = self.into();
        let result_f32 = val_f32.exp();
        <Self as From<f32>>::from(result_f32)
    }*/

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
        } else if self.bits == Self::ZERO.bits {
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

        if (self.bits as i32) > (other.bits as i32) {
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

        if (self.bits as i32) < (other.bits as i32) {
            self
        } else {
            other
        }
    }
}

// Implement MlScalar
impl MlScalar for Posit32 {}
