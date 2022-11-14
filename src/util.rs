use num_traits::{FromPrimitive, PrimInt};
use v_frame::{
    frame::Frame,
    prelude::{ChromaSampling, Pixel},
};

#[inline]
pub fn get_chroma_sampling<T: Pixel>(input: &Frame<T>) -> ChromaSampling {
    if input.planes[1].cfg.width == 0 {
        ChromaSampling::Cs400
    } else {
        match input.planes[1].cfg.xdec + input.planes[1].cfg.ydec {
            2 => ChromaSampling::Cs420,
            1 => ChromaSampling::Cs422,
            0 => ChromaSampling::Cs444,
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
pub fn round_halfup(x: f64) -> f64 {
    // When rounding on the pixel grid, the invariant
    //   round(x - 1) == round(x) - 1
    // must be preserved. This precludes the use of modes such as
    // half-to-even and half-away-from-zero.
    if x < 0.0_f64 {
        (x + 0.5).floor()
    } else {
        (x + 0.499_999_999_999_999_94).floor()
    }
}

/// Round up the argument x to the nearest multiple of n.
/// x must be non-negative and n must be positive and power-of-2.
#[inline(always)]
pub fn ceil_n<T: PrimInt + FromPrimitive>(x: T, n: usize) -> T {
    let n_minus_1 = T::from_usize(n - 1).unwrap();
    (x + n_minus_1) & !n_minus_1
}
