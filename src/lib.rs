#![deny(clippy::all)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::inconsistent_struct_constructor)]
#![allow(clippy::inline_always)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::similar_names)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::use_self)]
#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::create_dir)]
#![warn(clippy::dbg_macro)]
#![warn(clippy::default_numeric_fallback)]
#![warn(clippy::exit)]
#![warn(clippy::filetype_is_file)]
#![warn(clippy::float_cmp_const)]
#![warn(clippy::if_then_some_else_none)]
#![warn(clippy::lossy_float_literal)]
#![warn(clippy::map_err_ignore)]
#![warn(clippy::mem_forget)]
#![warn(clippy::multiple_inherent_impl)]
#![warn(clippy::pattern_type_mismatch)]
#![warn(clippy::rc_buffer)]
#![warn(clippy::rc_mutex)]
#![warn(clippy::rest_pat_in_fully_bound_structs)]
#![warn(clippy::same_name_method)]
#![warn(clippy::str_to_string)]
#![warn(clippy::string_to_string)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::unnecessary_self_imports)]
#![warn(clippy::unneeded_field_pattern)]
#![warn(clippy::use_debug)]
#![warn(clippy::verbose_file_reads)]

use anyhow::{ensure, Result};
use v_frame::{
    frame::Frame,
    prelude::{ChromaSampling, Pixel},
};

/// Specifies the number of pixels to crop off of each side of the image.
#[derive(Debug, Clone, Copy)]
pub struct CropDimensions {
    pub top: usize,
    pub bottom: usize,
    pub left: usize,
    pub right: usize,
}

/// Crops a video based on the given crop dimensions.
///
/// # Errors
///
/// - If the crop dimensions are not even
pub fn crop<T: Pixel>(input: &Frame<T>, dimensions: CropDimensions) -> Result<Frame<T>> {
    ensure!(
        dimensions.top % 2 == 0
            && dimensions.bottom % 2 == 0
            && dimensions.left % 2 == 0
            && dimensions.right % 2 == 0,
        "Crop dimensions must be a multiple of 2"
    );
    ensure!(
        dimensions.left + dimensions.right <= input.planes[0].cfg.width + 4,
        "Resulting width must be at least 4 pixels"
    );
    ensure!(
        dimensions.top + dimensions.bottom <= input.planes[0].cfg.height + 4,
        "Resulting height must be at least 4 pixels"
    );

    let new_w = input.planes[0].cfg.width - dimensions.left - dimensions.right;
    let new_h = input.planes[0].cfg.height - dimensions.top - dimensions.bottom;
    let chroma_sampling = if input.planes[1].cfg.width == 0 {
        ChromaSampling::Cs400
    } else {
        match input.planes[1].cfg.xdec + input.planes[1].cfg.ydec {
            2 => ChromaSampling::Cs420,
            1 => ChromaSampling::Cs422,
            0 => ChromaSampling::Cs444,
            _ => unreachable!(),
        }
    };
    let mut output: Frame<T> = Frame::new_with_padding(new_w, new_h, chroma_sampling, 0);
    for p in 0..(if chroma_sampling == ChromaSampling::Cs400 {
        1
    } else {
        3
    }) {
        let plane_cfg = &input.planes[p].cfg;
        let new_w = new_w >> plane_cfg.xdec;
        let new_h = new_h >> plane_cfg.ydec;
        let left = dimensions.left >> plane_cfg.xdec;
        let top = dimensions.top >> plane_cfg.ydec;

        for (out_row, in_row) in output.planes[p]
            .rows_iter_mut()
            .zip(input.planes[p].rows_iter().skip(top).take(new_h))
        {
            // SAFETY: `Frame` ensures that certain variants are upheld.
            // Given that we have verified the crop dimensions do not exceed
            // the original size of the frame, this is safe and avoids some bounds checks.
            unsafe {
                out_row
                    .as_mut_ptr()
                    .copy_from_nonoverlapping(in_row.as_ptr().add(left), new_w);
            }
        }
    }

    Ok(output)
}

#[cfg(feature = "devel")]
// This function exists for benchmarking and ASM inspection.
pub fn crop_u8(input: &Frame<u8>, dimensions: CropDimensions) -> Result<Frame<u8>> {
    crop::<u8>(input, dimensions)
}

#[cfg(feature = "devel")]
// This function exists for benchmarking and ASM inspection.
pub fn crop_u16(input: &Frame<u16>, dimensions: CropDimensions) -> Result<Frame<u16>> {
    crop::<u16>(input, dimensions)
}

/// Specifies the target resolution for the resized image.
#[derive(Debug, Clone, Copy)]
pub struct ResizeDimensions {
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum ResizeAlgorithm {
    BicubicCatmullRom,
}

/// Resizes a video to the given target dimensions.
///
/// # Errors
///
/// - If the resize dimensions are not even
pub fn resize<T: Pixel>(
    input: &Frame<T>,
    dimensions: ResizeDimensions,
    algorithm: ResizeAlgorithm,
) -> Result<Frame<T>> {
    ensure!(
        dimensions.width % 2 == 0 && dimensions.height % 2 == 0,
        "Resize dimensions must be a multiple of 2"
    );

    todo!()
}

/// Resamples a video to the given bit depth.
///
/// # Errors
///
/// - If an unsupported target bit depth is chosen.
///   - Currently supported bit depths are 8, 10, 12, and 16.
pub fn resample_bit_depth<T: Pixel, U: Pixel>(
    input: &Frame<T>,
    target_bit_depth: usize,
    dither: bool,
) -> Result<Frame<U>> {
    ensure!(
        [8, 10, 12, 16].contains(&target_bit_depth),
        "Currently supported bit depths are 8, 10, 12, and 16"
    );

    todo!()
}

/// Resamples a video to the given chroma subsampling.
#[must_use]
pub fn resample_chroma_sampling<T: Pixel>(
    input: &Frame<T>,
    target_chroma_sampling: ChromaSampling,
    algorithm: ResizeAlgorithm,
) -> Frame<T> {
    todo!()
}
