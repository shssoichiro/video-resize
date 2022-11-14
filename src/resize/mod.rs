pub mod algorithms;

use nalgebra::DMatrix;
use std::mem::align_of;
use v_frame::{
    frame::Frame,
    prelude::{ChromaSampling, Pixel},
};

use crate::util::{ceil_n, get_chroma_sampling, round_halfup};

pub fn should_resize_horiz_first(width_ratio: f32, height_ratio: f32) -> bool {
    let horiz_first_cost = width_ratio
        .max(1.0)
        .mul_add(2.0, width_ratio * height_ratio.max(1.0));
    let vert_first_cost = (height_ratio * width_ratio.max(1.0)).mul_add(2.0, height_ratio.max(1.0));

    horiz_first_cost < vert_first_cost
}

pub fn resize_horizontal<T: Pixel, F: ResizeAlgorithm>(
    input: &Frame<T>,
    dest_width: usize,
    bit_depth: usize,
) -> Frame<T> {
    let chroma_sampling = get_chroma_sampling(input);
    let src_height = input.planes[0].cfg.height;
    let pixel_max = (1i32 << bit_depth) - 1_i32;

    let mut output: Frame<T> = Frame::new_with_padding(dest_width, src_height, chroma_sampling, 0);
    for p in 0..(if chroma_sampling == ChromaSampling::Cs400 {
        1
    } else {
        3
    }) {
        let src_width = input.planes[p].cfg.width;
        let dest_width = output.planes[p].cfg.width;
        let filter = compute_filter::<F>(src_width, dest_width, 0.0, src_width as f64);

        for (in_row, out_row) in input.planes[p]
            .rows_iter()
            .zip(output.planes[p].rows_iter_mut())
        {
            // SAFETY: We know the bounds of the frame, so we are not worried about exceeding it
            unsafe {
                for j in 0..dest_width {
                    let top = *filter.left.get_unchecked(j);
                    let mut accum = 0i32;

                    for k in 0..filter.filter_width {
                        let coeff =
                            i32::from(*filter.data_i16.get_unchecked(j * filter.stride_i16 + k));
                        let x = unpack_pixel_u16(in_row.get_unchecked(top + k).to_u16().unwrap());
                        accum += coeff * x;
                    }

                    *out_row.get_unchecked_mut(j) = T::cast_from(pack_pixel_u16(accum, pixel_max));
                }
            }
        }
    }
    output
}

pub fn resize_vertical<T: Pixel, F: ResizeAlgorithm>(
    input: &Frame<T>,
    dest_height: usize,
    bit_depth: usize,
) -> Frame<T> {
    let chroma_sampling = get_chroma_sampling(input);
    let src_width = input.planes[0].cfg.width;
    let pixel_max = (1i32 << bit_depth) - 1_i32;

    let mut output: Frame<T> = Frame::new_with_padding(src_width, dest_height, chroma_sampling, 0);
    for p in 0..(if chroma_sampling == ChromaSampling::Cs400 {
        1
    } else {
        3
    }) {
        let src_height = input.planes[p].cfg.height;
        let dest_height = output.planes[p].cfg.height;
        let src_stride = input.planes[p].cfg.stride;
        let dest_stride = output.planes[p].cfg.stride;
        let filter = compute_filter::<F>(src_height, dest_height, 0.0, src_height as f64);

        for (i, (in_row, out_row)) in input.planes[p]
            .rows_iter()
            .zip(output.planes[p].rows_iter_mut())
            .enumerate()
        {
            // SAFETY: We know the bounds of the frame, so we are not worried about exceeding it
            unsafe {
                let filter_coeffs = &filter.data_i16[(i * filter.stride_i16)..];
                let top = *filter.left.get_unchecked(i);

                for j in 0..src_width {
                    let mut accum = 0i32;

                    for k in 0..filter.filter_width {
                        let coeff = i32::from(*filter_coeffs.get_unchecked(k));
                        let x = unpack_pixel_u16(
                            in_row
                                .get_unchecked((top + k) * src_stride + j)
                                .to_u16()
                                .unwrap(),
                        );
                        accum += coeff * x;
                    }

                    *out_row.get_unchecked_mut(i * dest_stride + j) =
                        T::cast_from(pack_pixel_u16(accum, pixel_max));
                }
            }
        }
    }
    output
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
    pub width: usize,
    pub height: usize,
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
    src_dim: usize,
    dest_dim: usize,
    shift: f64,
    width: f64,
) -> FilterContext {
    let scale = dest_dim as f64 / width;
    let step = scale.min(1.0);
    let support = f64::from(F::support()) / step;
    let filter_size = (support.ceil() as usize * 2).max(1);
    let f = F::new();
    let mut m: DMatrix<f64> = DMatrix::zeros(dest_dim, src_dim);

    let src_dim_f = src_dim as f64;
    for i in 0..dest_dim {
        // Position of output sample on input grid.
        let pos = (i as f64 + 0.5_f64) / scale + shift;
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

            let idx = (real_pos.floor() as usize).min(src_dim - 1);
            // SAFETY: We know the bounds of this matrix and will not exceed it
            unsafe {
                *m.get_unchecked_mut((i, idx)) += f.process((xpos - pos) * step) / total;
            }
            left = left.min(idx);
        }
    }

    matrix_to_filter(&m)
}

fn matrix_to_filter(m: &DMatrix<f64>) -> FilterContext {
    assert!(!m.is_empty());

    let width = m.row_iter().fold(0, |max, row| {
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
        // filter_rows: m.nrows(),
        // input_width: m.ncols(),
        // stride,
        // data: vec![0.0; stride * m.nrows()].into_boxed_slice(),
        stride_i16,
        data_i16: vec![0; stride_i16 * m.nrows()].into_boxed_slice(),
        left: vec![0; m.nrows()].into_boxed_slice(),
    };

    for (i, row) in m.row_iter().enumerate() {
        let left = row
            .iter()
            .position(|val| *val != 0.0_f64)
            .unwrap()
            .min(row.ncols() - width);
        let mut f32_err = 0.0_f64;
        let mut i16_err = 0.0_f64;
        let mut f32_sum = 0.0_f64;
        let mut i16_sum = 0_i16;
        let mut i16_greatest = 0_i16;
        let mut i16_greatest_idx = 0usize;

        // SAFETY: We know the bounds of the data structures we are dealing with,
        // and we do not go outside them.
        unsafe {
            // Dither filter coefficients when rounding them to their storage format.
            // This minimizes accumulation of error and ensures that the filter
            // continues to sum as close to 1.0 as possible after rounding.
            for j in 0..width {
                let coeff = *row.get_unchecked(left + j);

                let coeff_expected_f32 = coeff - f32_err;
                let coeff_expected_i16 = coeff.mul_add(f64::from(1i16 << 14usize), -i16_err);

                let coeff_f32 = coeff_expected_f32 as f32;
                let coeff_i16 = coeff_expected_i16.round() as i16;

                f32_err = coeff_expected_f32 as f64 - coeff_expected_f32;
                i16_err = coeff_expected_i16 as f64 - coeff_expected_i16;

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

            /* The final sum may still be off by a few ULP. This can not be fixed for
             * floating point data, since the error is dependent on summation order,
             * but for integer data, the error can be added to the greatest coefficient.
             */
            debug_assert!(
                1.0_f64 - f32_sum <= f64::from(f32::EPSILON),
                "error too great"
            );
            debug_assert!((1i16 << 14usize) - i16_sum <= 1, "error too great");

            *e.data_i16
                .get_unchecked_mut(i * e.stride_i16 + i16_greatest_idx) +=
                (1i16 << 14usize) - i16_sum;
            *e.left.get_unchecked_mut(i) = left;
        }
    }

    e
}
