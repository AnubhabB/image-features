// use image::{ImageBuff};

use std::f32::consts::PI;

use anyhow::Result;
use image::{save_buffer_with_format, ImageBuffer, Luma, Pixel};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

pub type ImageF32 = ImageBuffer<Luma<f32>, Vec<f32>>;

pub struct ImageOpsFilter<'a> {
    kernel: Box<dyn Fn(f32) -> f32 + 'a>,
    support: f32,
}

pub trait ImageOps {
    fn blur(&self, sigma: f32) -> ImageF32;
    fn draw(&self, path: &str) -> Result<()>;
    fn resize(&self, new_width: u32, new_height: u32) -> ImageF32;
    fn subtract(&self, rhs: &ImageF32) -> ImageF32;

    fn horizontal_sample(&self, new_width: u32, s: &mut ImageOpsFilter) -> ImageF32;
    fn vertical_sample(&self, new_height: u32, s: &mut ImageOpsFilter) -> ImageF32;
}

impl ImageOps for ImageF32 {
    fn blur(&self, sigma: f32) -> ImageF32 {
        let sigma = if sigma <= 0.0 { 1.0 } else { sigma };

        let mut f = ImageOpsFilter {
            kernel: Box::new(|x| gaussian(x, sigma)),
            support: 2. * sigma,
        };

        let out = self.vertical_sample(self.height(), &mut f);

        out.horizontal_sample(self.width(), &mut f)
    }

    fn resize(&self, new_width: u32, new_height: u32) -> ImageF32 {
        let mut f = ImageOpsFilter {
            kernel: Box::new(|_| 1.),
            support: 0.,
        };

        let out = self.vertical_sample(new_height, &mut f);

        out.horizontal_sample(new_width, &mut f)
    }

    fn draw(&self, path: &str) -> Result<()> {
        let p: Vec<u8> = self.to_vec().iter().map(|p| p.round() as u8).collect();
        save_buffer_with_format(
            path,
            &p[..],
            self.width() as u32,
            self.height() as u32,
            image::ColorType::L8,
            image::ImageFormat::Png,
        )?;
        Ok(())
    }

    fn horizontal_sample(&self, new_width: u32, s: &mut ImageOpsFilter) -> ImageF32 {
        let (width, height) = self.dimensions();
        let mut out = ImageF32::new(new_width, height);
        let mut ws = Vec::new();

        let ratio = width as f32 / new_width as f32;
        let sratio = if ratio < 1. { 1. } else { ratio };

        let src_support = s.support * sratio;

        for outx in 0..new_width {
            let inputx = (outx as f32 + 0.5) * ratio;

            let left = (inputx - src_support).floor() as i64;
            let left = clamp(left, 0, <i64 as From<_>>::from(width) - 1) as u32;

            let right = (inputx + src_support).ceil() as i64;
            let right = clamp(
                right,
                <i64 as From<_>>::from(left) + 1,
                <i64 as From<_>>::from(width),
            ) as u32;

            let inputx = inputx - 0.5;
            ws.clear();

            let mut sum = 0.0;
            for i in left..right {
                let w = (s.kernel)((i as f32 - inputx) / sratio);
                ws.push(w);
                sum += w;
            }
            ws.iter_mut().for_each(|w| *w /= sum);

            for y in 0..height {
                let mut t = [0.];

                for (i, w) in ws.iter().enumerate() {
                    let p = self.get_pixel(left + i as u32, y).channels();

                    t[0] += p[0] * w;
                }

                out.put_pixel(outx, y, Luma::<f32>::from(t));
            }
        }

        out
    }

    fn vertical_sample(&self, new_height: u32, s: &mut ImageOpsFilter) -> ImageF32 {
        let (width, height) = self.dimensions();
        let mut out = ImageF32::new(width, new_height);
        let mut ws = Vec::new();

        let ratio = height as f32 / new_height as f32;
        let sratio = if ratio < 1. { 1. } else { ratio };
        let src_support = s.support * sratio;

        for outy in 0..new_height {
            let inputy = (outy as f32 + 0.5) * ratio;
            let left = (inputy - src_support).floor() as i64;
            let left = clamp(left, 0, <i64 as From<_>>::from(height) - 1) as u32;

            let right = (inputy + src_support).ceil() as i64;
            let right = clamp(
                right,
                <i64 as From<_>>::from(left) + 1,
                <i64 as From<_>>::from(height),
            ) as u32;

            let inputy = inputy - 0.5;

            ws.clear();
            let mut sum = 0.0;
            for i in left..right {
                let w = (s.kernel)((i as f32 - inputy) / sratio);
                ws.push(w);
                sum += w;
            }
            ws.iter_mut().for_each(|w| *w /= sum);

            for x in 0..width {
                let mut t = [0.];

                for (i, w) in ws.iter().enumerate() {
                    let p = self.get_pixel(x, left + i as u32).channels();

                    // #[allow(deprecated)]
                    // let (k1, k2, k3, k4) = p.channels4();
                    // let vec: (f32, f32, f32, f32) = (
                    //     NumCast::from(k1).unwrap(),
                    //     NumCast::from(k2).unwrap(),
                    //     NumCast::from(k3).unwrap(),
                    //     NumCast::from(k4).unwrap(),
                    // );

                    t[0] += p[0] * w;
                }

                // #[allow(deprecated)]
                // This is not necessarily Rgba.
                // let t = Pixel::from_channels(t.0, t.1, t.2, t.3);

                out.put_pixel(x, outy, Luma::<f32>::from(t));
            }
        }

        out
    }

    fn subtract(&self, other: &ImageF32) -> ImageF32 {
        let mut img = ImageF32::new(self.width(), self.height());

        img.par_iter_mut().enumerate().for_each(|(i, p)| {
            *p = self.get(i).unwrap() - other.get(i).unwrap();
        });

        img
    }
}

// pub fn blur_f32( image_f32: &ImageF32, sigma: f32 ) -> ImageF32 {
//     let sigma = if sigma <= 0.0 { 1.0 } else { sigma };
//     // let out = vertical_sample_blur_f32(image_f32, sigma);

//     // horizontal_sample_blur_f32(&out, sigma)
// }

// Sample the columns of the supplied image using the provided filter.
// The width of the image remains unchanged.
// ```new_height``` is the desired height of the new image
// ```filter``` is the filter to use for sampling.
// The return value is not necessarily Rgba, the underlying order of channels in ```image``` is
// preserved.
// fn vertical_sample(image: &I, new_height: u32, filter: &mut Filter) -> Rgba32FImage {
//     // let (width, height) = image.dimensions();
//     // let mut out = ImageBuffer::new(width, new_height);
//     // let mut ws = Vec::new();

//     // let ratio = height as f32 / new_height as f32;
//     // let sratio = if ratio < 1.0 { 1.0 } else { ratio };
//     // let src_support = filter.support * sratio;

//     for outy in 0..new_height {
//         // For an explanation of this algorithm, see the comments
//         // in horizontal_sample.
//         // let inputy = (outy as f32 + 0.5) * ratio;

//         // let left = (inputy - src_support).floor() as i64;
//         // let left = clamp(left, 0, <i64 as From<_>>::from(height) - 1) as u32;

//         // let right = (inputy + src_support).ceil() as i64;
//         // let right = clamp(
//         //     right,
//         //     <i64 as From<_>>::from(left) + 1,
//         //     <i64 as From<_>>::from(height),
//         // ) as u32;

//         // let inputy = inputy - 0.5;

//         // ws.clear();
//         // let mut sum = 0.0;
//         // for i in left..right {
//         //     let w = (filter.kernel)((i as f32 - inputy) / sratio);
//         //     ws.push(w);
//         //     sum += w;
//         // }
//         // ws.iter_mut().for_each(|w| *w /= sum);

//         for x in 0..width {
//             let mut t = (0.0, 0.0, 0.0, 0.0);

//             for (i, w) in ws.iter().enumerate() {
//                 let p = image.get_pixel(x, left + i as u32);

//                 #[allow(deprecated)]
//                 let (k1, k2, k3, k4) = p.channels4();
//                 let vec: (f32, f32, f32, f32) = (
//                     NumCast::from(k1).unwrap(),
//                     NumCast::from(k2).unwrap(),
//                     NumCast::from(k3).unwrap(),
//                     NumCast::from(k4).unwrap(),
//                 );

//                 t.0 += vec.0 * w;
//                 t.1 += vec.1 * w;
//                 t.2 += vec.2 * w;
//                 t.3 += vec.3 * w;
//             }

//             #[allow(deprecated)]
//             // This is not necessarily Rgba.
//             let t = Pixel::from_channels(t.0, t.1, t.2, t.3);

//             out.put_pixel(x, outy, t);
//         }
//     }

//     out
// }

// fn vertical_sample_blur_f32(image_f32: &ImageBuffer<Luma<f32>, Vec<f32>>, sigma: f32) -> ImageBuffer<Luma<f32>, Vec<f32>> {
//     let (width, height) = image_f32.dimensions();
//     let mut out = ImageBuffer::<Luma<f32>, Vec<f32>>::new(width, height);
//     let mut ws = Vec::new();

//     // let ratio = 1.;
//     // let sratio = if ratio < 1.0 { 1.0 } else { ratio };
//     let src_support = 2. * sigma;

//     let kernel = Box::new(|x| gaussian(x, sigma));

//     for outy in 0..height {
//         // For an explanation of this algorithm, see the comments
//         // in horizontal_sample.
//         let inputy = outy as f32 + 0.5;

//         let left = (inputy - src_support).floor() as i64;
//         let left = clamp(left, 0, <i64 as From<_>>::from(height) - 1) as u32;

//         let right = (inputy + src_support).ceil() as i64;
//         let right = clamp(
//             right,
//             <i64 as From<_>>::from(left) + 1,
//             <i64 as From<_>>::from(height),
//         ) as u32;

//         let inputy = inputy - 0.5;

//         ws.clear();
//         let mut sum = 0.0;
//         for i in left..right {
//             let w = (kernel)(i as f32 - inputy);
//             ws.push(w);
//             sum += w;
//         }
//         ws.iter_mut().for_each(|w| *w /= sum);

//         for x in 0..width {
//             let mut t = [0.0];

//             ws
//             .iter()
//             .enumerate()
//             .for_each(|(i, w)| {
//                 let p = image_f32.get_pixel(x, left + i as u32).channels();
//                 // let mut k = t.write();
//                 t[0] += p[0] * w;
//             });

//             // let t = t.into_inner();
//             out.put_pixel(x, outy, Luma::<f32>::from(t));
//         }
//     }

//     out
// }

// fn horizontal_sample_blur_f32( image_f32: &ImageBuffer<Luma<f32>, Vec<f32>>, sigma: f32) -> ImageBuffer<Luma<f32>, Vec<f32>> {
//     let (width, height) = image_f32.dimensions();
//     let mut out = ImageBuffer::new(width, height);
//     let mut ws = Vec::new();

//     // let max: f32 = 1.;
//     // let min: f32 = 0.;
//     let ratio = 1.;
//     let sratio = if ratio < 1.0 { 1.0 } else { ratio };
//     let src_support = 2. * sigma;

//     let kernel = Box::new(|x| gaussian(x, sigma));

//     for outx in 0..width {
//         // Find the point in the input image corresponding to the centre
//         // of the current pixel in the output image.
//         let inputx = (outx as f32 + 0.5) * ratio;

//         // Left and right are slice bounds for the input pixels relevant
//         // to the output pixel we are calculating.  Pixel x is relevant
//         // if and only if (x >= left) && (x < right).

//         // Invariant: 0 <= left < right <= width

//         let left = (inputx - src_support).floor() as i64;
//         let left = clamp(left, 0, <i64 as From<_>>::from(width) - 1) as u32;

//         let right = (inputx + src_support).ceil() as i64;
//         let right = clamp(
//             right,
//             <i64 as From<_>>::from(left) + 1,
//             <i64 as From<_>>::from(width),
//         ) as u32;

//         // Go back to left boundary of pixel, to properly compare with i
//         // below, as the kernel treats the centre of a pixel as 0.
//         let inputx = inputx - 0.5;

//         ws.clear();
//         let mut sum = 0.0;
//         for i in left..right {
//             let w = (kernel)((i as f32 - inputx) / sratio);
//             ws.push(w);
//             sum += w;
//         }
//         ws.iter_mut().for_each(|w| *w /= sum);

//         for y in 0..height {
//             let mut t = [0.0];

//             ws
//             .iter()
//             .enumerate()
//             .for_each(|(i, w)| {
//                 let p = image_f32.get_pixel(left + i as u32, y).channels();
//                 // let mut k = t.write();
//                 t[0] += p[0] * w;
//             });

//             // let t = t.into_inner();
//             out.put_pixel(outx, y, Luma(t));
//         }
//     }

//     out
// }

fn clamp<N>(a: N, min: N, max: N) -> N
where
    N: PartialOrd,
{
    if a < min {
        min
    } else if a > max {
        max
    } else {
        a
    }
}
fn gaussian(x: f32, r: f32) -> f32 {
    ((2.0 * PI).sqrt() * r).recip() * (-x.powi(2) / (2.0 * r.powi(2))).exp()
}

#[cfg(test)]
mod tests {
    // use anyhow::Result;
    // use image::{GenericImageView, imageops};

    // #[test]
    // fn blur() -> Result<()> {
    // let im = image::open("data/1_small.png").unwrap();
    // let dim = im.dimensions();
    // let im = im.grayscale().resize((dim.0 * 2) as u32, (dim.1 * 2) as u32, imageops ::FilterType::Nearest);

    // let sigma = 1.6_f32.powf(2.) - (2. * 0.5_f32).powf(2.).sqrt();

    // let im1 = blur_f32(&im.to_luma32f(), sigma);
    // let im2 = im.blur(sigma).to_luma8();

    // println!("{:?}", &im1.to_vec()[0..1000]);
    // println!("{:?}", &im2.to_vec()[0..1000]);

    //     Ok(())
    // }

    // #[test]
    // fn lstsq() -> Result<()> {
    //     use nalgebra::{OMatrix, OVector, U3};
    //     // So: [0.5, 0.0, 0.0] [-1.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.25, 0.0] -> -(lstsq(hess, grad)[0]) -> [ 0.5, -0. , -0. ]

    //     let hess = OMatrix::<f32, U3, U3>::from_row_slice(&[
    //         -1.0, 0.0, 0.0,
    //         0.0, 0.0, 0.25,
    //         0.0, 0.25, 0.0
    //     ]);

    //     let grad = OVector::<f32, U3>::from_row_slice(&[0.5, 0.0, 0.0]);

    //     let lst = -(lstsq::lstsq(&hess, &grad, EPSILON).unwrap().solution);

    //     println!("{:?}", &lst);
    //     Ok(())
    // }
}
