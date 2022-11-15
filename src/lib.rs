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

mod resize;
mod util;

use std::mem::size_of;

pub use crate::resize::{algorithms, ResizeAlgorithm, ResizeDimensions};
use crate::{
    resize::{resize_horizontal, resize_vertical, should_resize_horiz_first},
    util::get_chroma_sampling,
};
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
    let chroma_sampling = get_chroma_sampling(input);
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

/// Resizes a video to the given target dimensions.
///
/// # Errors
///
/// - If the resize dimensions are not even
/// - If the specified input bit depth does not match the size of `T`
pub fn resize<T: Pixel, F: ResizeAlgorithm>(
    input: &Frame<T>,
    dimensions: ResizeDimensions,
    input_bit_depth: usize,
) -> Result<Frame<T>> {
    if size_of::<T>() == 1 {
        ensure!(
            input_bit_depth == 8,
            "input bit depth must be 8 for 8-bit pixel type"
        );
    } else if size_of::<T>() == 2 {
        ensure!(
            input_bit_depth > 8 && input_bit_depth <= 16,
            "input bit depth must be between 9-16 for 8-bit pixel type"
        );
    } else {
        unreachable!("32-bit types not implemented in v_frame");
    }
    ensure!(
        dimensions.width % 2 == 0 && dimensions.height % 2 == 0,
        "Resize dimensions must be a multiple of 2"
    );
    ensure!(
        dimensions.width >= 4 && dimensions.height >= 4,
        "Resulting image must be at least 4x4 pixels"
    );

    let src_w = input.planes[0].cfg.width;
    let src_h = input.planes[0].cfg.height;
    let resize_horiz = src_w != dimensions.width;
    let resize_vert = src_h != dimensions.height;
    if !resize_horiz {
        return Ok(resize_vertical::<T, F>(
            input,
            dimensions.height,
            input_bit_depth,
        ));
    }

    if !resize_vert {
        return Ok(resize_horizontal::<T, F>(
            input,
            dimensions.width,
            input_bit_depth,
        ));
    }

    let horiz_first = should_resize_horiz_first(
        dimensions.width as f32 / src_w as f32,
        dimensions.height as f32 / src_h as f32,
    );
    if horiz_first {
        let temp = resize_horizontal::<T, F>(input, dimensions.width, input_bit_depth);
        return Ok(resize_vertical::<T, F>(
            &temp,
            dimensions.height,
            input_bit_depth,
        ));
    }
    let temp = resize_vertical::<T, F>(input, dimensions.height, input_bit_depth);
    Ok(resize_horizontal::<T, F>(
        &temp,
        dimensions.width,
        input_bit_depth,
    ))
}

#[cfg(feature = "devel")]
pub fn resize_horizontal_u8_bicubic(input: &Frame<u8>, dest_width: usize) -> Frame<u8> {
    use resize::algorithms::BicubicMitchell;

    resize_horizontal::<u8, BicubicMitchell>(input, dest_width, 8)
}

#[cfg(feature = "devel")]
pub fn resize_horizontal_u16_bicubic(
    input: &Frame<u16>,
    dest_width: usize,
    input_bit_depth: usize,
) -> Frame<u16> {
    use resize::algorithms::BicubicMitchell;

    resize_horizontal::<u16, BicubicMitchell>(input, dest_width, input_bit_depth)
}

#[cfg(feature = "devel")]
pub fn resize_vertical_u8_bicubic(input: &Frame<u8>, dest_height: usize) -> Frame<u8> {
    use resize::algorithms::BicubicMitchell;

    resize_vertical::<u8, BicubicMitchell>(input, dest_height, 8)
}

#[cfg(feature = "devel")]
pub fn resize_vertical_u16_bicubic(
    input: &Frame<u16>,
    dest_height: usize,
    input_bit_depth: usize,
) -> Frame<u16> {
    use resize::algorithms::BicubicMitchell;

    resize_vertical::<u16, BicubicMitchell>(input, dest_height, input_bit_depth)
}

#[cfg(feature = "devel")]
pub fn resize_horizontal_u8_lanczos3(input: &Frame<u8>, dest_width: usize) -> Frame<u8> {
    use resize::algorithms::Lanczos3;

    resize_horizontal::<u8, Lanczos3>(input, dest_width, 8)
}

#[cfg(feature = "devel")]
pub fn resize_horizontal_u16_lanczos3(
    input: &Frame<u16>,
    dest_width: usize,
    input_bit_depth: usize,
) -> Frame<u16> {
    use resize::algorithms::Lanczos3;

    resize_horizontal::<u16, Lanczos3>(input, dest_width, input_bit_depth)
}

#[cfg(feature = "devel")]
pub fn resize_vertical_u8_lanczos3(input: &Frame<u8>, dest_height: usize) -> Frame<u8> {
    use resize::algorithms::Lanczos3;

    resize_vertical::<u8, Lanczos3>(input, dest_height, 8)
}

#[cfg(feature = "devel")]
pub fn resize_vertical_u16_lanczos3(
    input: &Frame<u16>,
    dest_height: usize,
    input_bit_depth: usize,
) -> Frame<u16> {
    use resize::algorithms::Lanczos3;

    resize_vertical::<u16, Lanczos3>(input, dest_height, input_bit_depth)
}

#[cfg(feature = "devel")]
pub fn resize_horizontal_u8_spline36(input: &Frame<u8>, dest_width: usize) -> Frame<u8> {
    use resize::algorithms::Spline36;

    resize_horizontal::<u8, Spline36>(input, dest_width, 8)
}

#[cfg(feature = "devel")]
pub fn resize_horizontal_u16_spline36(
    input: &Frame<u16>,
    dest_width: usize,
    input_bit_depth: usize,
) -> Frame<u16> {
    use resize::algorithms::Spline36;

    resize_horizontal::<u16, Spline36>(input, dest_width, input_bit_depth)
}

#[cfg(feature = "devel")]
pub fn resize_vertical_u8_spline36(input: &Frame<u8>, dest_height: usize) -> Frame<u8> {
    use resize::algorithms::Spline36;

    resize_vertical::<u8, Spline36>(input, dest_height, 8)
}

#[cfg(feature = "devel")]
pub fn resize_vertical_u16_spline36(
    input: &Frame<u16>,
    dest_height: usize,
    input_bit_depth: usize,
) -> Frame<u16> {
    use resize::algorithms::Spline36;

    resize_vertical::<u16, Spline36>(input, dest_height, input_bit_depth)
}

/// Resamples a video to the given bit depth.
///
/// # Errors
///
/// - If an unsupported target bit depth is chosen.
///   - Currently supported bit depths are 8, 10, 12, and 16.
/// - If the specified input bit depth does not match the size of `T`
/// - If the specified target bit depth does not match the size of `U`
///
/// # Panics
///
/// - Not yet implemented
#[doc(hidden)]
pub fn resample_bit_depth<T: Pixel, U: Pixel>(
    _input: &Frame<T>,
    input_bit_depth: usize,
    target_bit_depth: usize,
    _dither: bool,
) -> Result<Frame<U>> {
    if size_of::<T>() == 1 {
        ensure!(
            input_bit_depth == 8,
            "input bit depth must be 8 for 8-bit pixel type"
        );
    } else if size_of::<T>() == 2 {
        ensure!(
            input_bit_depth > 8 && input_bit_depth <= 16,
            "input bit depth must be between 9-16 for 8-bit pixel type"
        );
    } else {
        unreachable!("32-bit types not implemented in v_frame");
    }
    if size_of::<U>() == 1 {
        ensure!(
            target_bit_depth == 8,
            "target bit depth must be 8 for 8-bit pixel type"
        );
    } else if size_of::<U>() == 2 {
        ensure!(
            target_bit_depth > 8 && target_bit_depth <= 16,
            "target bit depth must be between 9-16 for 8-bit pixel type"
        );
    } else {
        unreachable!("32-bit types not implemented in v_frame");
    }
    ensure!(
        [8, 10, 12, 16].contains(&target_bit_depth),
        "Currently supported bit depths are 8, 10, 12, and 16"
    );

    todo!()
}

/// Resamples a video to the given chroma subsampling.
///
/// # Errors
///
/// - If the specified input bit depth does not match the size of `T`
///
/// # Panics
///
/// - Not yet implemented
#[doc(hidden)]
pub fn resample_chroma_sampling<T: Pixel, F: ResizeAlgorithm>(
    _input: &Frame<T>,
    input_bit_depth: usize,
    _target_chroma_sampling: ChromaSampling,
) -> Result<Frame<T>> {
    if size_of::<T>() == 1 {
        ensure!(
            input_bit_depth == 8,
            "input bit depth must be 8 for 8-bit pixel type"
        );
    } else if size_of::<T>() == 2 {
        ensure!(
            input_bit_depth > 8 && input_bit_depth <= 16,
            "input bit depth must be between 9-16 for 8-bit pixel type"
        );
    } else {
        unreachable!("32-bit types not implemented in v_frame");
    }

    todo!()
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use image::{DynamicImage, Rgb32FImage};
    use yuvxyb::{
        ColorPrimaries, Frame, MatrixCoefficients, Rgb, TransferCharacteristic, Yuv, YuvConfig,
    };

    use crate::{
        algorithms::{BicubicCatmullRom, Bilinear, Lanczos3, Point, Spline64},
        crop, resize, CropDimensions, ResizeDimensions,
    };

    fn get_u8_test_image() -> Frame<u8> {
        let i = image::open("./test_files/ducks_take_off.png")
            .unwrap()
            .to_rgb32f();
        let rgb = Rgb::new(
            i.pixels().map(|p| [p[0], p[1], p[2]]).collect(),
            i.width() as usize,
            i.height() as usize,
            TransferCharacteristic::BT1886,
            ColorPrimaries::BT709,
        )
        .unwrap();
        let yuv: Yuv<u8> = (
            rgb,
            YuvConfig {
                bit_depth: 8,
                subsampling_x: 1,
                subsampling_y: 1,
                full_range: false,
                matrix_coefficients: MatrixCoefficients::BT709,
                transfer_characteristics: TransferCharacteristic::BT1886,
                color_primaries: ColorPrimaries::BT709,
            },
        )
            .try_into()
            .unwrap();
        let data = yuv.data();
        Frame {
            planes: [data[0].clone(), data[1].clone(), data[2].clone()],
        }
    }

    fn get_u16_test_image() -> Frame<u16> {
        let i = image::open("./test_files/ducks_take_off.png")
            .unwrap()
            .to_rgb32f();
        let rgb = Rgb::new(
            i.pixels().map(|p| [p[0], p[1], p[2]]).collect(),
            i.width() as usize,
            i.height() as usize,
            TransferCharacteristic::BT1886,
            ColorPrimaries::BT709,
        )
        .unwrap();
        let yuv: Yuv<u16> = (
            rgb,
            YuvConfig {
                bit_depth: 10,
                subsampling_x: 1,
                subsampling_y: 1,
                full_range: false,
                matrix_coefficients: MatrixCoefficients::BT709,
                transfer_characteristics: TransferCharacteristic::BT1886,
                color_primaries: ColorPrimaries::BT709,
            },
        )
            .try_into()
            .unwrap();
        let data = yuv.data();
        Frame {
            planes: [data[0].clone(), data[1].clone(), data[2].clone()],
        }
    }

    fn output_image_from_u8(image: Frame<u8>, filename: &str) {
        let width = image.planes[0].cfg.width;
        let height = image.planes[0].cfg.height;
        let yuv: Yuv<u8> = Yuv::new(
            image,
            YuvConfig {
                bit_depth: 8,
                subsampling_x: 1,
                subsampling_y: 1,
                full_range: false,
                matrix_coefficients: MatrixCoefficients::BT709,
                transfer_characteristics: TransferCharacteristic::BT1886,
                color_primaries: ColorPrimaries::BT709,
            },
        )
        .unwrap();
        let rgb: Rgb = yuv.try_into().unwrap();
        let i = DynamicImage::ImageRgb32F(
            Rgb32FImage::from_vec(
                width as u32,
                height as u32,
                rgb.data().iter().copied().flatten().collect(),
            )
            .unwrap(),
        )
        .to_rgb8();
        if !Path::new("/tmp/video-resize-tests").is_dir() {
            fs::create_dir_all("/tmp/video-resize-tests").unwrap();
        }
        i.save(filename).unwrap();
    }

    fn output_image_from_u16(image: Frame<u16>, filename: &str) {
        let width = image.planes[0].cfg.width;
        let height = image.planes[0].cfg.height;
        let yuv: Yuv<u16> = Yuv::new(
            image,
            YuvConfig {
                bit_depth: 10,
                subsampling_x: 1,
                subsampling_y: 1,
                full_range: false,
                matrix_coefficients: MatrixCoefficients::BT709,
                transfer_characteristics: TransferCharacteristic::BT1886,
                color_primaries: ColorPrimaries::BT709,
            },
        )
        .unwrap();
        let rgb: Rgb = yuv.try_into().unwrap();
        let i = DynamicImage::ImageRgb32F(
            Rgb32FImage::from_vec(
                width as u32,
                height as u32,
                rgb.data().iter().copied().flatten().collect(),
            )
            .unwrap(),
        )
        .to_rgb16();
        if !Path::new("/tmp/video-resize-tests").is_dir() {
            fs::create_dir_all("/tmp/video-resize-tests").unwrap();
        }
        i.save(filename).unwrap();
    }

    #[test]
    fn crop_u8() {
        let input = get_u8_test_image();
        let output = crop::<u8>(
            &input,
            CropDimensions {
                top: 40,
                bottom: 40,
                left: 20,
                right: 20,
            },
        )
        .unwrap();
        output_image_from_u8(output, "/tmp/video-resize-tests/crop_u8.png");
    }

    #[test]
    fn crop_u16() {
        let input = get_u16_test_image();
        let output = crop::<u16>(
            &input,
            CropDimensions {
                top: 40,
                bottom: 40,
                left: 20,
                right: 20,
            },
        )
        .unwrap();
        output_image_from_u16(output, "/tmp/video-resize-tests/crop_u16.png");
    }

    #[test]
    fn resize_point_u8_down() {
        let input = get_u8_test_image();
        let output = resize::<u8, Point>(
            &input,
            ResizeDimensions {
                width: 1280,
                height: 720,
            },
            8,
        )
        .unwrap();
        output_image_from_u8(output, "/tmp/video-resize-tests/resize_point_u8_down.png");
    }

    #[test]
    fn resize_point_u16_down() {
        let input = get_u16_test_image();
        let output = resize::<u16, Point>(
            &input,
            ResizeDimensions {
                width: 1280,
                height: 720,
            },
            10,
        )
        .unwrap();
        output_image_from_u16(output, "/tmp/video-resize-tests/resize_point_u16_down.png");
    }

    #[test]
    fn resize_point_u8_up() {
        let input = get_u8_test_image();
        let output = resize::<u8, Point>(
            &input,
            ResizeDimensions {
                width: 2560,
                height: 1440,
            },
            8,
        )
        .unwrap();
        output_image_from_u8(output, "/tmp/video-resize-tests/resize_point_u8_up.png");
    }

    #[test]
    fn resize_point_u16_up() {
        let input = get_u16_test_image();
        let output = resize::<u16, Point>(
            &input,
            ResizeDimensions {
                width: 2560,
                height: 1440,
            },
            10,
        )
        .unwrap();
        output_image_from_u16(output, "/tmp/video-resize-tests/resize_point_u16_up.png");
    }

    #[test]
    fn resize_bilinear_u8_down() {
        let input = get_u8_test_image();
        let output = resize::<u8, Bilinear>(
            &input,
            ResizeDimensions {
                width: 1280,
                height: 720,
            },
            8,
        )
        .unwrap();
        output_image_from_u8(
            output,
            "/tmp/video-resize-tests/resize_bilinear_u8_down.png",
        );
    }

    #[test]
    fn resize_bilinear_u16_down() {
        let input = get_u16_test_image();
        let output = resize::<u16, Bilinear>(
            &input,
            ResizeDimensions {
                width: 1280,
                height: 720,
            },
            10,
        )
        .unwrap();
        output_image_from_u16(
            output,
            "/tmp/video-resize-tests/resize_bilinear_u16_down.png",
        );
    }

    #[test]
    fn resize_bilinear_u8_up() {
        let input = get_u8_test_image();
        let output = resize::<u8, Bilinear>(
            &input,
            ResizeDimensions {
                width: 2560,
                height: 1440,
            },
            8,
        )
        .unwrap();
        output_image_from_u8(output, "/tmp/video-resize-tests/resize_bilinear_u8_up.png");
    }

    #[test]
    fn resize_bilinear_u16_up() {
        let input = get_u16_test_image();
        let output = resize::<u16, Bilinear>(
            &input,
            ResizeDimensions {
                width: 2560,
                height: 1440,
            },
            10,
        )
        .unwrap();
        output_image_from_u16(output, "/tmp/video-resize-tests/resize_bilinear_u16_up.png");
    }

    #[test]
    fn resize_bicubic_u8_down() {
        let input = get_u8_test_image();
        let output = resize::<u8, BicubicCatmullRom>(
            &input,
            ResizeDimensions {
                width: 1280,
                height: 720,
            },
            8,
        )
        .unwrap();
        output_image_from_u8(output, "/tmp/video-resize-tests/resize_bicubic_u8_down.png");
    }

    #[test]
    fn resize_bicubic_u16_down() {
        let input = get_u16_test_image();
        let output = resize::<u16, BicubicCatmullRom>(
            &input,
            ResizeDimensions {
                width: 1280,
                height: 720,
            },
            10,
        )
        .unwrap();
        output_image_from_u16(
            output,
            "/tmp/video-resize-tests/resize_bicubic_u16_down.png",
        );
    }

    #[test]
    fn resize_bicubic_u8_up() {
        let input = get_u8_test_image();
        let output = resize::<u8, BicubicCatmullRom>(
            &input,
            ResizeDimensions {
                width: 2560,
                height: 1440,
            },
            8,
        )
        .unwrap();
        output_image_from_u8(output, "/tmp/video-resize-tests/resize_bicubic_u8_up.png");
    }

    #[test]
    fn resize_bicubic_u16_up() {
        let input = get_u16_test_image();
        let output = resize::<u16, BicubicCatmullRom>(
            &input,
            ResizeDimensions {
                width: 2560,
                height: 1440,
            },
            10,
        )
        .unwrap();
        output_image_from_u16(output, "/tmp/video-resize-tests/resize_bicubic_u16_up.png");
    }

    #[test]
    fn resize_lanczos_u8_down() {
        let input = get_u8_test_image();
        let output = resize::<u8, Lanczos3>(
            &input,
            ResizeDimensions {
                width: 1280,
                height: 720,
            },
            8,
        )
        .unwrap();
        output_image_from_u8(output, "/tmp/video-resize-tests/resize_lanczos_u8_down.png");
    }

    #[test]
    fn resize_lanczos_u16_down() {
        let input = get_u16_test_image();
        let output = resize::<u16, Lanczos3>(
            &input,
            ResizeDimensions {
                width: 1280,
                height: 720,
            },
            10,
        )
        .unwrap();
        output_image_from_u16(
            output,
            "/tmp/video-resize-tests/resize_lanczos_u16_down.png",
        );
    }

    #[test]
    fn resize_lanczos_u8_up() {
        let input = get_u8_test_image();
        let output = resize::<u8, Lanczos3>(
            &input,
            ResizeDimensions {
                width: 2560,
                height: 1440,
            },
            8,
        )
        .unwrap();
        output_image_from_u8(output, "/tmp/video-resize-tests/resize_lanczos_u8_up.png");
    }

    #[test]
    fn resize_lanczos_u16_up() {
        let input = get_u16_test_image();
        let output = resize::<u16, Lanczos3>(
            &input,
            ResizeDimensions {
                width: 2560,
                height: 1440,
            },
            10,
        )
        .unwrap();
        output_image_from_u16(output, "/tmp/video-resize-tests/resize_lanczos_u16_up.png");
    }

    #[test]
    fn resize_spline64_u8_down() {
        let input = get_u8_test_image();
        let output = resize::<u8, Spline64>(
            &input,
            ResizeDimensions {
                width: 1280,
                height: 720,
            },
            8,
        )
        .unwrap();
        output_image_from_u8(
            output,
            "/tmp/video-resize-tests/resize_spline64_u8_down.png",
        );
    }

    #[test]
    fn resize_spline64_u16_down() {
        let input = get_u16_test_image();
        let output = resize::<u16, Spline64>(
            &input,
            ResizeDimensions {
                width: 1280,
                height: 720,
            },
            10,
        )
        .unwrap();
        output_image_from_u16(
            output,
            "/tmp/video-resize-tests/resize_spline64_u16_down.png",
        );
    }

    #[test]
    fn resize_spline64_u8_up() {
        let input = get_u8_test_image();
        let output = resize::<u8, Spline64>(
            &input,
            ResizeDimensions {
                width: 2560,
                height: 1440,
            },
            8,
        )
        .unwrap();
        output_image_from_u8(output, "/tmp/video-resize-tests/resize_spline64_u8_up.png");
    }

    #[test]
    fn resize_spline64_u16_up() {
        let input = get_u16_test_image();
        let output = resize::<u16, Spline64>(
            &input,
            ResizeDimensions {
                width: 2560,
                height: 1440,
            },
            10,
        )
        .unwrap();
        output_image_from_u16(output, "/tmp/video-resize-tests/resize_spline64_u16_up.png");
    }
}
