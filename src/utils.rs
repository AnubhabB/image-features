// use image::{ImageBuff};

use std::f32::consts::PI;

use image::{ImageBuffer, Luma, Pixel};

pub fn blur_f32( image_f32: &ImageBuffer<Luma<f32>, Vec<f32>>, sigma: f32 ) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let sigma = if sigma <= 0.0 { 1.0 } else { sigma };
    let out = vertical_sample_blur_f32(image_f32, sigma);

    horizontal_sample_blur_f32(&out, sigma)
}

fn vertical_sample_blur_f32(image_f32: &ImageBuffer<Luma<f32>, Vec<f32>>, sigma: f32) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let (width, height) = image_f32.dimensions();
    let mut out = ImageBuffer::<Luma<f32>, Vec<f32>>::new(width, height);
    let mut ws = Vec::new();

    // let ratio = 1.;
    // let sratio = if ratio < 1.0 { 1.0 } else { ratio };
    let src_support = 2. * sigma;

    let kernel = Box::new(|x| gaussian(x, sigma));

    for outy in 0..height {
        // For an explanation of this algorithm, see the comments
        // in horizontal_sample.
        let inputy = outy as f32 + 0.5;

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
            let w = (kernel)(i as f32 - inputy);
            ws.push(w);
            sum += w;
        }
        ws.iter_mut().for_each(|w| *w /= sum);

        for x in 0..width {
            let mut t = [0.0];

            ws
            .iter()
            .enumerate()
            .for_each(|(i, w)| {
                let p = image_f32.get_pixel(x, left + i as u32).channels();
                // let mut k = t.write();
                t[0] += p[0] * w;
            });

            // let t = t.into_inner();
            out.put_pixel(x, outy, Luma(t));
        }
    }

    out
}

fn horizontal_sample_blur_f32( image_f32: &ImageBuffer<Luma<f32>, Vec<f32>>, sigma: f32) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let (width, height) = image_f32.dimensions();
    let mut out = ImageBuffer::new(width, height);
    let mut ws = Vec::new();

    // let max: f32 = 1.;
    // let min: f32 = 0.;
    let ratio = 1.;
    let sratio = if ratio < 1.0 { 1.0 } else { ratio };
    let src_support = 2. * sigma;

    let kernel = Box::new(|x| gaussian(x, sigma));

    for outx in 0..width {
        // Find the point in the input image corresponding to the centre
        // of the current pixel in the output image.
        let inputx = (outx as f32 + 0.5) * ratio;

        // Left and right are slice bounds for the input pixels relevant
        // to the output pixel we are calculating.  Pixel x is relevant
        // if and only if (x >= left) && (x < right).

        // Invariant: 0 <= left < right <= width

        let left = (inputx - src_support).floor() as i64;
        let left = clamp(left, 0, <i64 as From<_>>::from(width) - 1) as u32;

        let right = (inputx + src_support).ceil() as i64;
        let right = clamp(
            right,
            <i64 as From<_>>::from(left) + 1,
            <i64 as From<_>>::from(width),
        ) as u32;

        // Go back to left boundary of pixel, to properly compare with i
        // below, as the kernel treats the centre of a pixel as 0.
        let inputx = inputx - 0.5;

        ws.clear();
        let mut sum = 0.0;
        for i in left..right {
            let w = (kernel)((i as f32 - inputx) / sratio);
            ws.push(w);
            sum += w;
        }
        ws.iter_mut().for_each(|w| *w /= sum);

        for y in 0..height {
            let mut t = [0.0];

            ws
            .iter()
            .enumerate()
            .for_each(|(i, w)| {
                let p = image_f32.get_pixel(left + i as u32, y).channels();
                // let mut k = t.write();
                t[0] += p[0] * w;
            });
            
            // let t = t.into_inner();
            out.put_pixel(outx, y, Luma(t));
        }
    }

    out
}

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
    use super::blur_f32;
    use anyhow::Result;
    use image::{GenericImageView, imageops};

    #[test]
    fn blur() -> Result<()> {
        let im = image::open("data/1_small.png").unwrap();
        let dim = im.dimensions();
        let im = im.grayscale().resize((dim.0 * 2) as u32, (dim.1 * 2) as u32, imageops ::FilterType::Nearest);


        let sigma = 1.6_f32.powf(2.) - (2. * 0.5_f32).powf(2.).sqrt();

        let im1 = blur_f32(&im.to_luma32f(), sigma);
        let im2 = im.blur(sigma).to_luma8();
        println!("{:?}", &im1.to_vec()[0..1000]);
        println!("{:?}", &im2.to_vec()[0..1000]);
        Ok(())
    }

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