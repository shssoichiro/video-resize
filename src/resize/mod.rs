pub mod algorithms;

use anyhow::Result;
use std::{
    mem::align_of,
    num::{NonZeroU8, NonZeroUsize},
};
use v_frame::{
    chroma::ChromaSubsampling,
    frame::{Frame, FrameBuilder},
    pixel::Pixel,
};

use crate::util::{ceil_n, round_halfup};

pub fn should_resize_horiz_first(width_ratio: f32, height_ratio: f32) -> bool {
    let horiz_first_cost = width_ratio
        .max(1.0)
        .mul_add(2.0, width_ratio * height_ratio.max(1.0));
    let vert_first_cost = (height_ratio * width_ratio.max(1.0)).mul_add(2.0, height_ratio.max(1.0));

    horiz_first_cost < vert_first_cost
}

pub fn resize_horizontal<T: Pixel, F: ResizeAlgorithm>(
    input: &Frame<T>,
    dest_width: NonZeroUsize,
    bit_depth: NonZeroU8,
) -> Result<Frame<T>> {
    let pixel_max = (1i32 << bit_depth.get()) - 1_i32;

    let mut output: Frame<T> = FrameBuilder::new(
        dest_width,
        input.y_plane.height(),
        input.subsampling,
        bit_depth,
    )
    .build()?;
    for p in 0..(if input.subsampling == ChromaSubsampling::Monochrome {
        1
    } else {
        3
    }) {
        let input_plane = input.plane(p).expect("has plane");
        let output_plane = output.plane_mut(p).expect("has plane");
        let src_width = input_plane.width();
        let dest_width = output_plane.width();
        let filter = compute_filter::<F>(src_width, dest_width, src_width);

        for (in_row, out_row) in input_plane.rows().zip(output_plane.rows_mut()) {
            // SAFETY: We control the size and bounds
            unsafe {
                #[allow(clippy::needless_range_loop)]
                for j in 0..dest_width.get() {
                    let top = *filter.left.get_unchecked(j);
                    let mut accum = 0i32;

                    for k in 0..filter.filter_width {
                        let coeff =
                            i32::from(*filter.data_i16.get_unchecked(j * filter.stride_i16 + k));
                        let x = unpack_pixel_u16(in_row.get_unchecked(top + k).to_u16().unwrap());
                        accum += coeff * x;
                    }

                    *out_row.get_unchecked_mut(j) = match size_of::<T>() {
                        1 => T::from(pack_pixel_u16(accum, pixel_max) as u8).expect("T is u8"),
                        2 => T::from(pack_pixel_u16(accum, pixel_max)).expect("T is u16"),
                        _ => unreachable!(),
                    };
                }
            }
        }
    }
    Ok(output)
}

pub fn resize_vertical<T: Pixel, F: ResizeAlgorithm>(
    input: &Frame<T>,
    dest_height: NonZeroUsize,
    bit_depth: NonZeroU8,
) -> Result<Frame<T>> {
    let pixel_max = (1i32 << bit_depth.get()) - 1_i32;

    let mut output: Frame<T> = FrameBuilder::new(
        input.y_plane.width(),
        dest_height,
        input.subsampling,
        bit_depth,
    )
    .build()?;
    for p in 0..(if input.subsampling == ChromaSubsampling::Monochrome {
        1
    } else {
        3
    }) {
        let input_plane = input.plane(p).expect("plane exists");
        let output_plane = output.plane_mut(p).expect("plane exists");
        let src_height = input_plane.height();
        let dest_height = output_plane.height();
        let src_width = input_plane.width();
        let src_stride = input_plane.geometry().stride;
        let dest_stride = output_plane.geometry().stride;
        let input_data = &input_plane.data()[input_plane.data_origin()..];
        let out_origin = output_plane.data_origin();
        let output_data = &mut output_plane.data_mut()[out_origin..];
        let filter = compute_filter::<F>(src_height, dest_height, src_height);

        for i in 0..dest_height.get() {
            // SAFETY: We control the size and bounds
            unsafe {
                let filter_coeffs = filter.data_i16.as_ptr().add(i * filter.stride_i16);
                let top = *filter.left.get_unchecked(i);

                for j in 0..src_width.get() {
                    let mut accum = 0i32;

                    for k in 0..filter.filter_width {
                        let coeff = i32::from(*filter_coeffs.add(k));
                        let x = unpack_pixel_u16(
                            input_data
                                .get_unchecked((top + k) * src_stride.get() + j)
                                .to_u16()
                                .unwrap(),
                        );
                        accum += coeff * x;
                    }

                    *output_data.get_unchecked_mut(i * dest_stride.get() + j) = match size_of::<T>()
                    {
                        1 => T::from(pack_pixel_u16(accum, pixel_max) as u8).expect("T is u8"),
                        2 => T::from(pack_pixel_u16(accum, pixel_max)).expect("T is u16"),
                        _ => unreachable!(),
                    };
                }
            }
        }
    }
    Ok(output)
}

#[inline(always)]
fn unpack_pixel_u16(x: u16) -> i32 {
    i32::from(x) + i32::from(i16::MIN)
}

#[inline(always)]
fn pack_pixel_u16(x: i32, pixel_max: i32) -> u16 {
    let x = ((x + (1_i32 << 13usize)) >> 14usize) - i32::from(i16::MIN);
    let x = x.min(pixel_max).max(0_i32);

    x as u16
}

/// Specifies the target resolution for the resized image.
#[derive(Debug, Clone, Copy)]
pub struct ResizeDimensions {
    pub width: NonZeroUsize,
    pub height: NonZeroUsize,
}

pub trait ResizeAlgorithm {
    fn support() -> u32;
    fn new() -> Self;
    fn process(&self, x: f64) -> f64;
}

struct FilterContext {
    filter_width: usize,
    // TODO: Enable these fields if v_frame ever supports f32 types
    // filter_rows: usize,
    // input_width: usize,
    // stride: usize,
    // data: Box<[f32]>,
    stride_i16: usize,
    data_i16: Box<[i16]>,
    left: Box<[usize]>,
}

fn compute_filter<F: ResizeAlgorithm>(
    src_dim: NonZeroUsize,
    dest_dim: NonZeroUsize,
    width: NonZeroUsize,
) -> FilterContext {
    let scale = dest_dim.get() as f64 / width.get() as f64;
    let step = scale.min(1.0);
    let support = f64::from(F::support()) / step;
    let filter_size = (support.ceil() as usize * 2).max(1);
    let f = F::new();
    // This represents a row-major matrix with dest_dim rows and src_dim cols
    //
    // TODO: We should be able to represent this as a compressed sparse matrix
    // to reduce memory usage.
    let mut m: Vec<f64> = vec![0.0_f64; dest_dim.get() * src_dim.get()];

    let src_dim_f = src_dim.get() as f64;
    for i in 0..dest_dim.get() {
        // Position of output sample on input grid.
        let pos = (i as f64 + 0.5_f64) / scale;
        let begin_pos = round_halfup((filter_size as f64).mul_add(-0.5, pos)) + 0.5_f64;

        let mut total = 0.0_f64;
        for j in 0..filter_size {
            let xpos = begin_pos + j as f64;
            total += f.process((xpos - pos) * step);
        }

        let mut left = usize::MAX;

        for j in 0..filter_size {
            let xpos = begin_pos + j as f64;

            // Mirror the position if it goes beyond image bounds.
            let real_pos = if xpos < 0.0_f64 {
                -xpos
            } else if xpos >= src_dim_f {
                2.0f64.mul_add(src_dim_f, -xpos)
            } else {
                xpos
            };

            // Clamp the position if it is still out of bounds.
            let real_pos = real_pos.max(0.0);

            let idx = (real_pos.floor() as usize).min(src_dim.get() - 1);
            // SAFETY: We control the size and bounds
            unsafe {
                *m.get_unchecked_mut(i * src_dim.get() + idx) +=
                    f.process((xpos - pos) * step) / total;
            }
            left = left.min(idx);
        }
    }

    matrix_to_filter(&m, src_dim)
}

fn matrix_to_filter(m: &[f64], input_width: NonZeroUsize) -> FilterContext {
    assert!(!m.is_empty());

    let height = m.len() / input_width;
    let width = m.chunks_exact(input_width.get()).fold(0, |max, row| {
        let mut first = None;
        let mut last = None;
        for (idx, val) in row.iter().enumerate() {
            // We want to find the first and last index that have a non-zero value.
            if first.is_none() {
                if *val == 0.0_f64 {
                    continue;
                }
                first = Some(idx);
            }
            if *val == 0.0_f64 {
                // This is the end of the non-sparse values.
                break;
            }
            last = Some(idx);
        }
        let width = last.unwrap() + 1 - first.unwrap();
        max.max(width)
    });
    // TODO: Enable this code if v_frame ever supports f32 types
    // let stride = ceil_n(width, align_of::<f32>());
    let stride_i16 = ceil_n(width, align_of::<u16>());
    let mut e = FilterContext {
        filter_width: width,
        // TODO: Enable these fields if v_frame ever supports f32 types
        // filter_rows: height,
        // input_width: m.ncols(),
        // stride,
        // data: vec![0.0; stride * height].into_boxed_slice(),
        stride_i16,
        data_i16: vec![0; stride_i16 * height].into_boxed_slice(),
        left: vec![0; height].into_boxed_slice(),
    };

    for (i, row) in m.chunks_exact(input_width.get()).enumerate() {
        let left = row
            .iter()
            .position(|val| *val != 0.0_f64)
            .unwrap()
            .min(row.len() - width);
        let mut f32_err = 0.0_f64;
        let mut i16_err = 0.0_f64;
        let mut f32_sum = 0.0_f64;
        let mut i16_sum = 0_i16;
        let mut i16_greatest = 0_i16;
        let mut i16_greatest_idx = 0usize;

        // Dither filter coefficients when rounding them to their storage format.
        // This minimizes accumulation of error and ensures that the filter
        // continues to sum as close to 1.0 as possible after rounding.
        for j in 0..width {
            // SAFETY: We control the size and bounds
            unsafe {
                let coeff = *row.get_unchecked(left + j);

                let coeff_expected_f32 = coeff - f32_err;
                let coeff_expected_i16 = coeff.mul_add(f64::from(1i16 << 14usize), -i16_err);

                let coeff_f32 = coeff_expected_f32 as f32;
                let coeff_i16 = coeff_expected_i16.round() as i16;

                #[allow(clippy::unnecessary_cast)]
                {
                    f32_err = coeff_expected_f32 as f64 - coeff_expected_f32;
                    i16_err = coeff_expected_i16 as f64 - coeff_expected_i16;
                }

                if coeff_i16.abs() > i16_greatest {
                    i16_greatest = coeff_i16;
                    i16_greatest_idx = j;
                }

                f32_sum += f64::from(coeff_f32);
                i16_sum += coeff_i16;

                // TODO: Enable this code if v_frame ever supports f32 types
                // *e.data.get_unchecked_mut(i * stride + j) = coeff_f32;
                *e.data_i16.get_unchecked_mut(i * stride_i16 + j) = coeff_i16;
            }
        }

        /* The final sum may still be off by a few ULP. This can not be fixed for
         * floating point data, since the error is dependent on summation order,
         * but for integer data, the error can be added to the greatest coefficient.
         */
        debug_assert!(
            1.0_f64 - f32_sum <= f64::from(f32::EPSILON),
            "error too great"
        );
        debug_assert!((1i16 << 14usize) - i16_sum <= 1, "error too great");

        // SAFETY: We control the size and bounds
        unsafe {
            *e.data_i16
                .get_unchecked_mut(i * e.stride_i16 + i16_greatest_idx) +=
                (1i16 << 14usize) - i16_sum;
            *e.left.get_unchecked_mut(i) = left;
        }
    }

    e
}
