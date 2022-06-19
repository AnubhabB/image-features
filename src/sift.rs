use std::f32::{consts::E, EPSILON};
use std::iter::zip;
use std::ops::Div;
use std::sync::{Arc};

use image::imageops::resize;
use image::{DynamicImage, GenericImageView, save_buffer_with_format, ColorType, imageops, ImageBuffer, Luma};
use anyhow::{Result};
use nalgebra::{OMatrix, U3, U1, DMatrix, Matrix3, Vector3};
use parking_lot::RwLock;
use rayon::{scope, iter::{IntoParallelRefMutIterator, IndexedParallelIterator, ParallelIterator, IntoParallelRefIterator}};

use crate::utils::blur_f32;


const BLUR: f32 = 0.5;
const SIGMA: f32 = 1.6;
const INTERVAL: f32 = 3.;
const CONTRAST_THESHOLD: f32 = 0.04; // from OpenCV implementation
const EIGENVALUE_RATIO: f32 = 10.; // from OpenCV implementation default

#[derive(Default)]
pub struct Sift {
    img: DynamicImage,
    viz: bool,
    dogs: Vec<Vec<Vec<f32>>>,
    images: Vec<Vec<OctaveImage>>,
    octaves: u32,

}

#[derive(Debug, Clone, Default)]
struct OctaveImage {
    w: u32,
    h: u32,
    v: Option<DMatrix<f32>>
}


#[derive(Debug, Clone, Default, PartialEq)]
struct KeyPoint {
    prc: (usize, usize),
    oct: usize,
    size: f32,
    res: f32,
    angle: f32
}

impl Sift {
    pub fn new(input_path: &str) -> Result<Self> {
        // Image { input_path }
        let img = image::open(input_path)?;

        Ok(Self {
            img,
            ..Default::default()
        })
    }

    pub fn visualize(&mut self) {
        self.viz = true;
    }

    /// Scale Invarient Feature Transform
    /// 
    pub fn generate(&mut self) -> Result<()> {
        let gaussian_kernels = Self::gaussian_kernels(SIGMA, INTERVAL);

        self.octaves = self.compute_octaves();
        self.generate_images(&gaussian_kernels[..])?;
        self.dogs()?;

        let keypoints = self.scale_space_extrema()?;
        let desciptors = self.generate_descriptors(&keypoints[..])?;
        
        // println!("{:?}", keypoints);

        Ok(())
    }

    fn sift_base_image(&self, sigma: f32, b: f32) -> ImageBuffer<Luma<f32>, Vec<f32>> {
        let sigma_diff = sigma.powf(2.) - (2. * b).powf(2.).sqrt();
        let dim = self.img.dimensions();

        let img = &self.to_grey()
            .resize(dim.0 * 2, dim.1 * 2, imageops::FilterType::Nearest)
            .to_luma32f();
            
        blur_f32(img, sigma_diff)
    }

    fn compute_octaves(&self) -> u32 {
        let dim = self.img.dimensions();
        let min = if dim.0 < dim.1 {
            dim.0 as f32
        } else {
            dim.1 as f32
        } * 2.; // multiplying by this because the base image is going to change

        ((min.log(E) / 2_f32.log(E)) - 1.).floor() as u32
    }

    fn generate_images(&mut self, kernels: &[f32]) -> Result<()> {
        let kernel_len = kernels.len();

        let mut bimg = self.sift_base_image(SIGMA, BLUR);
        // let imgs = Arc::new(RwLock::new(
        //     vec![OctaveImage::default(); (self.octaves * kernel_len) as usize]));
        let imgs = Arc::new(
            RwLock::new(
                vec![
                    vec![
                        OctaveImage::default();
                        kernel_len
                    ];
                    self.octaves as usize
                ]
            )
        );
        
        let mid_p = (kernel_len - 3) as usize;
        // generating base images for each octave has to happen in sequence because each octave is dependent on the last image of the previous octave
        for i in 0 .. self.octaves as usize {
            {
                // let first = (i * kernel_len) as usize;
                let mut l = imgs.write();
                let mut b_vec = bimg.to_vec();
                b_vec.par_iter_mut().for_each(|p| *p *= 255.);

                let w = bimg.width() as usize;
                let h = bimg.height() as usize;

                let k = DMatrix::from_row_slice(h, w, &b_vec[..]);

                l[i][0] = OctaveImage {
                    w: bimg.width(),
                    h: bimg.height(),
                    v: Some(k)
                };
                
                if self.viz {
                    // this is a sample visualization
                    // save_buffer_with_format(
                    //     format!("data/sift/{i}_0.png").as_str(),
                    //     bimg.as_bytes(),
                    //     bimg.dimensions().0,
                    //     bimg.dimensions().1,
                    //     ColorType::L8,
                    //     image::ImageFormat::Png,
                    // ).unwrap();
                }
            }

            // let timg = RwLock::new(DynamicImage::new_luma8(0, 0));
            let timg = RwLock::new(ImageBuffer::new(0,0));

            kernels
            .par_iter()
            .skip(1)
            .enumerate()
            .for_each(|(ki, k)| {
                let ti = ki + 1;
                // let idx = (i * kernel_len) + ti as u32;

                // let gimg = bimg.blur(*k);
                let gimg = blur_f32(&bimg, *k);
                let mut g_data = gimg.to_vec();
                g_data.par_iter_mut().for_each(|p| *p *= 255.);

                let w = gimg.width() as usize;
                let h = gimg.height() as usize;

                let k = DMatrix::from_row_slice(h, w, &g_data[..]);

                {
                    let mut w = imgs.write();
                    w[i][ti] = OctaveImage {
                        w: gimg.width(),
                        h: gimg.height(),
                        v: Some(k)
                    };
                }

                if self.viz {
                    // let dims = gimg.dimensions();
                    // this is a sample visualization
                    // save_buffer_with_format(
                    //     format!("data/sift/{i}_{ki}.png").as_str(),
                    //     g_bytes,
                    //     dims.0,
                    //     dims.1,
                    //     ColorType::L8,
                    //     image::ImageFormat::Png,
                    // ).unwrap();
                }

                if ti == mid_p {
                    let mut w = timg.write();
                    *w = gimg;
                }
            });
            let timg = timg.into_inner();
            bimg = resize(&timg, timg.width()/2, timg.height()/2, imageops::FilterType::Nearest);
        }

        let b = imgs.read();
        self.images = b.clone();

        Ok(())
    }

    // Difference-of-gradients
    fn dogs(&mut self) -> Result<()> {
        let img_per_oct = self.images[0].len() - 1;
        // let dogs = Arc::new(RwLock::new(vec![ );
        let dogs = Arc::new(
            RwLock::new(
                vec![
                    vec![
                        Vec::<f32>::new();
                        img_per_oct
                    ];
                    self.octaves as usize
                ]
            )
        );

        // let images = &self.images[..];
        let viz = self.viz;

        scope(|s| {
            for o in 0 .. self.octaves as usize {
                let images = &self.images[o][..];

                for (idx, oimg) in images.iter().take(images.len() - 1).enumerate() {
                    let dogs = dogs.clone();
                    s.spawn(move |_| {
                        // let im0 = &oimg.v.unwrap();
                        // let im1 = &images.get(idx + 1).unwrap().v.unwrap();
                        let (im0, im1) = if let (Some(im0), Some(im1)) = (&oimg.v, &images[idx + 1].v) {
                            (im0, im1)
                        } else {
                            return;
                        };

                        let mut v = vec![0.;im0.len()];
                        
                        v.par_iter_mut()
                            .enumerate()
                            .for_each(|(k, v)| {
                                let d = im1[k] as f32 - im0[k] as f32;
                                *v = d;
                            });
                        
                        if viz {
                            let h: f32;
                            let l: f32;

                            let buf = RwLock::new(vec![0; v.len()]);

                            {
                                let mut k = v.clone();
                                k.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

                                l = k[0];
                                h = k[k.len() - 1];
                            }
                            
                            v.par_iter()
                                .enumerate()
                                .for_each(|(i, d)| {
                                    let val = (((*d - l)/ (h - l)) * 255.).floor() as u8;
                                    let mut r = buf.write();
                                    r[i] = val;
                                });
                            
                            let g = &images[idx];
                            save_buffer_with_format(
                                format!("data/sift/dog_{}_{}.png", o, idx).as_str(),
                                &buf.into_inner(),
                                g.w,
                                g.h,
                                ColorType::L8,
                                image::ImageFormat::Png,
                            ).unwrap();
                        }
                        let mut l = dogs.write();
                        
                        l[o][idx] = v;
                    });
                }
            }
        });
        
        self.dogs = dogs.read().clone();

        Ok(())
    }

    // Find pixel positions of all scale-space extrema in the image pyramid
    fn scale_space_extrema(&self) -> Result<Vec<KeyPoint>> {
        let threshold = (0.5 * CONTRAST_THESHOLD / INTERVAL * 255.).floor(); // from OpenCV implementation
        let keypoints = Arc::new(RwLock::new(Vec::<KeyPoint>::new()));

        scope(|s| {
            for (octave, oct) in self.dogs.iter().enumerate() {
                let (w, h) = (self.images[octave][0].w as usize, self.images[octave][0].h as usize);
    
                for (img_idx, (img0, (img1, img2))) in zip(oct, zip(&oct[1..], &oct[2..])).enumerate() {
                    let img0 = Arc::new(DMatrix::from_row_slice(h as usize, w as usize, &img0[..]));
                    let img1 = Arc::new(DMatrix::from_row_slice(h as usize, w as usize, &img1[..]));
                    let img2 = Arc::new(DMatrix::from_row_slice(h as usize, w as usize, &img2[..]));
    
                    for c in 1 .. w - 2 {
                        for r in 1 .. h - 2 {
                            let img0 = img0.clone();
                            let img1 = img1.clone();
                            let img2 = img2.clone();

                            let keypoints = keypoints.clone();

                            s.spawn(move |_| {
                                if !Self::is_pixel_extremum(
                                    img0.index((r-1 .. r+2, c-1 .. c+2)).into(),
                                    img1.index((r-1 .. r+2, c-1 .. c+2)).into(),
                                    img2.index((r-1 .. r+2, c-1 .. c+2)).into(),
                                    threshold
                                ) {
                                    return;
                                }

                                let localize = self.localize_extremum(img_idx + 1, octave, (r, c), w, h);
                                if localize.is_none() {
                                    return;
                                }
                                
                                let (kp, local_img_idx) = localize.unwrap();
                                let mut kp_with_orientations = self.compute_keypoint_orientation(&kp, octave, w, h, local_img_idx);
                                if !kp_with_orientations.is_empty() {
                                    let mut l = keypoints.write();
                                    l.append(&mut kp_with_orientations);
                                }
                            });
                        }
                    }
                }
            }
        });
        
        // Sort keypoints and remove duplicates if any
        {
            let mut w = keypoints.write();
            w.sort_unstable_by(|k1, k2| {
                if k1.prc.0 != k2.prc.0 {
                    return k1.prc.0.cmp(&k2.prc.0);
                }
    
                if k1.prc.1 != k2.prc.1 {
                    return k2.prc.1.cmp(&k2.prc.1);
                }
    
                if k1.size != k2.size && let Some(o) = k1.size.partial_cmp(&k2.size) {
                    return o;
                }
    
                if k1.angle != k2.angle && let Some(o) = k1.angle.partial_cmp(&k2.angle) {
                    return o;
                }
    
                if k1.res != k2.res && let Some(o) = k1.res.partial_cmp(&k2.res) {
                    return o;
                }
    
                k1.oct.cmp(&k2.oct)
            });
            w.dedup();

            // rescaling keypoints as per https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-2-c4350274be2b
            // not sure of the `WHY` for the octave though
            w.iter_mut()
            .for_each(|k| {
                k.prc = (k.prc.0/ 2, k.prc.1/2);
                k.size *= 0.5;
                k.oct = (k.oct & !255) | ((k.oct - 1) & 255);
            });

        }


        let kp = keypoints.read().clone();

        Ok(kp)
    }

    fn localize_extremum(&self, im1_idx: usize, octave_idx: usize, px_idx: (usize, usize), oct_w: usize, oct_h: usize) -> Option<(KeyPoint, usize)> {
        let divisor: f32 = 255.;
        let mut converge = false;

        let mut r = px_idx.0;
        let mut c = px_idx.1;
        let mut imidx = im1_idx;

        let mut grad: Vector3<f32> = Vector3::default();
        let mut hess: Matrix3<f32> = Matrix3::default();
        let mut lstq: OMatrix<f32, U3, U1> = OMatrix::default();

        let mut pxstack: [Matrix3<f32>; 3] = [Matrix3::<f32>::default(); 3];

        for _ in 0 .. 5 {
            let cnt = (r * oct_w) + c;
            let top = (r - 1) * oct_w;
            let bot = (r + 1) * oct_w;

            pxstack = [
                Matrix3::from_row_slice(
                    &[
                        self.dogs[octave_idx][imidx - 1][top - 1], self.dogs[octave_idx][imidx - 1][top], self.dogs[octave_idx][imidx - 1][top + 1],
                        self.dogs[octave_idx][imidx - 1][cnt - 1], self.dogs[octave_idx][imidx - 1][cnt], self.dogs[octave_idx][imidx - 1][cnt + 1],
                        self.dogs[octave_idx][imidx - 1][bot - 1], self.dogs[octave_idx][imidx - 1][bot], self.dogs[octave_idx][imidx - 1][bot + 1]
                    ]
                ).div(divisor),
                Matrix3::from_row_slice(
                    &[
                        self.dogs[octave_idx][imidx][top - 1], self.dogs[octave_idx][imidx][top], self.dogs[octave_idx][imidx][top + 1],
                        self.dogs[octave_idx][imidx][cnt - 1], self.dogs[octave_idx][imidx][cnt], self.dogs[octave_idx][imidx][cnt + 1],
                        self.dogs[octave_idx][imidx][bot - 1], self.dogs[octave_idx][imidx][bot], self.dogs[octave_idx][imidx][bot + 1]
                    ]
                ).div(divisor),
                Matrix3::from_row_slice(
                    &[
                        self.dogs[octave_idx][imidx + 1][top - 1], self.dogs[octave_idx][imidx + 1][top], self.dogs[octave_idx][imidx + 1][top + 1],
                        self.dogs[octave_idx][imidx + 1][cnt - 1], self.dogs[octave_idx][imidx + 1][cnt], self.dogs[octave_idx][imidx + 1][cnt + 1],
                        self.dogs[octave_idx][imidx + 1][bot - 1], self.dogs[octave_idx][imidx + 1][bot], self.dogs[octave_idx][imidx + 1][bot + 1]
                    ]
                ).div(divisor),
            ];

            grad = Self::compute_gradient(pxstack);
            hess = Self::compute_hessian(pxstack);
            lstq = -lstsq::lstsq(&hess, &grad, EPSILON).unwrap().solution;
            
            if lstq[0].abs() < 0.5 && lstq[1].abs() < 0.5 && lstq[2].abs() < 0.5 {
                converge = true;
                break;
            }

            r += lstq[0].round() as usize;
            c += lstq[1].round() as usize;
            imidx += lstq[2].round() as usize;

            if r < 1 || c < 1 || r >= oct_h - 1 || c >= oct_w - 1 || imidx < 1 || imidx > INTERVAL as usize {
                break;
            }
        }

        if !converge {
            return None;
        }

        let val_at_extremum = pxstack[1].index((1, 1)) + 0.5 * grad.dot(&lstq);
        
        if val_at_extremum.abs() * INTERVAL < CONTRAST_THESHOLD {
            return None;
        }

        let xy_hess = hess.index((..2, ..2));
        let xy_hess_trace = xy_hess.trace();
        let xy_hess_det = xy_hess.determinant();

        // // if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:

        if xy_hess_det <= 0. || EIGENVALUE_RATIO * xy_hess_trace.powf(2.) >= ((EIGENVALUE_RATIO + 1.).powf(2.)) * xy_hess_det {
            return None;
        }

        Some((KeyPoint{
            prc: (
                ((r as f32 + lstq[0]) * 2_f32.powf(octave_idx as f32)) as usize,
                ((c as f32 + lstq[1]) * 2_f32.powf(octave_idx as f32)) as usize
            ),
            oct: octave_idx + imidx * 2_u32.pow(8) as usize + (((lstq[2] + 0.5) * 255.).round() * 2_f32.powf(16.)) as usize,
            size: SIGMA * (
                2_f32.powf(
                    (imidx as f32 + lstq[2]) / INTERVAL
                ) *
                2_f32.powf(octave_idx as f32 + 1.)
            ),
            res: val_at_extremum.abs(),
            angle: 0.
        }, imidx))
    }

    fn gaussian_kernels(sigma: f32, intervals: f32) -> Vec<f32> {
        let img_per_octave = intervals + 3.;
        let b: f32 = 2.;

        let k = b.powf(1./ intervals);

        let mut gk = vec![0.; img_per_octave as usize];
        gk[0] = sigma;

        gk.par_iter_mut()
            .enumerate()
            .skip(1)
            .for_each(|(i, u)| {
                let prev = k.powf(i as f32 - 1.) * sigma;
                let total = k * prev;

                *u = (total.powf(2.) - prev.powf(2.)).sqrt();
            });

        gk
    }

    fn to_grey(&self) -> DynamicImage {
        self.img.grayscale()
    }

    fn is_pixel_extremum(im0: DMatrix<f32>, im1: DMatrix<f32>, im2: DMatrix<f32>, threshold: f32) -> bool {
        let px = im1.index((1, 1));
        if px.abs() <= threshold {
            return false;
        }
        
        if px > &0. {
            for (p0, p2) in zip(im0.iter(), im2.iter()) {
                if px < p0 || px < p2 {
                    return false
                }
            }

            for (i, p) in im1.iter().enumerate() {
                // this is the center pixel, move on
                if i == 4 {
                    continue;
                }

                if px < p {
                    return false;
                }
            }

            true
        } else if px < &0. {
            for (p0, p2) in zip(im0.iter(), im2.iter()) {
                if px > p0 || px > p2 {
                    return false
                }
            }

            for (i, p) in im1.iter().enumerate() {
                // this is the center pixel, move on
                if i == 4 {
                    continue;
                }

                if px > p {
                    return false;
                }
            }

            true
        } else {
            false
        }

    }

    // Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    // https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-1-306a99b50aa5

    // With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
    // Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
    // NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    fn compute_gradient(px_cube:  [Matrix3<f32>; 3]) -> Vector3<f32> {
        // dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
        // dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
        // ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
        let dx = 0.5 * (px_cube[1].index((1, 2)) - px_cube[1].index((1, 0)));
        let dy = 0.5 * (px_cube[1].index((2, 1)) - px_cube[1].index((0, 1)));
        let ds = 0.5 * (px_cube[2].index((1, 1)) - px_cube[0].index((1, 1)));
        
        
        Vector3::from([dx, dy, ds])
    }

    // Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    // https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-1-306a99b50aa5

    // With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    // Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    // With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    // Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    // NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    fn compute_hessian(px_cube:  [Matrix3<f32>; 3]) -> Matrix3<f32> {
        // dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
        let px_cent = px_cube[1].index((1,1));

        let dxx = px_cube[1].index((1, 2)) - 2. * px_cent + px_cube[1].index((1, 0));
        let dyy = px_cube[1].index((2, 1)) - 2. * px_cent + px_cube[1].index((0, 1));
        let dss = px_cube[2].index((1, 1)) - 2. * px_cent + px_cube[0].index((1, 1));
        let dxy = 0.25 * (px_cube[1].index((2, 2)) - px_cube[1].index((2, 0)) - px_cube[1].index((0, 2)) + px_cube[1].index((0, 0)));
        let dxs = 0.25 * (px_cube[2].index((1, 2)) - px_cube[2].index((1, 0)) - px_cube[0].index((1, 2)) + px_cube[0].index((1, 0)));
        let dys = 0.25 * (px_cube[2].index((2, 1)) - px_cube[2].index((0, 1)) - px_cube[0].index((2, 1)) + px_cube[0].index((0, 1)));

        Matrix3::from_row_slice(&[
            dxx, dxy, dxs,
            dxy, dyy, dys,
            dxs, dys, dss
        ])
    }


    // Compute orientations for each keypoint
    fn compute_keypoint_orientation(&self, kp: &KeyPoint, octave_idx: usize, oct_w: usize, oct_h: usize, img_idx: usize) -> Vec<KeyPoint> {
        const NUM_BINS: usize = 36;

        let scale = 1.5 * kp.size / 2_f32.powf(octave_idx as f32 + 1.);
        let radius = (3. * scale).round() as i32;
        let weight_f = -0.5 / scale.powf(2.);

        let mut histogram_raw = [0.; NUM_BINS];
        let mut histogram_smooth = [0.; NUM_BINS];

        let mut keypoint_with_orient = Vec::<KeyPoint>::new();

        // let gaus_in_oct = self.images.len() / self.octaves as usize;
        // let img_idx = (gaus_in_oct * octave_idx) + kp.idx_oct;
        let img = self.images[octave_idx][img_idx].v.as_ref().unwrap();
        // let img_data = &img.v[..];

        for i in -radius .. radius + 1 {
            let region_y = (kp.prc.0 as f32 / 2_f32.powf(octave_idx as f32)).round() as i32 + i;
            if region_y <= 0 || region_y >= oct_h as i32 - 1 {
                continue;
            }

            for j in -radius .. radius + 1 {
                let region_x = (kp.prc.1 as f32 / 2_f32.powf(octave_idx as f32)).round() as i32 + j;
                if region_x <= 0 || region_x >= oct_w as i32 - 1 {
                    continue;
                }

                let dx = img.index((region_y as usize, (region_x + 1) as usize)) - img.index((region_y as usize, (region_x - 1) as usize));
                let dy = img.index(((region_y - 1) as usize, region_x as usize)) - img.index(((region_y + 1) as usize, region_x as usize));

                let grad_mag = (dx.powf(2.) + dy.powf(2.)).sqrt();
                let grad_deg = dy.atan2(dx).to_degrees();

                let weight = (weight_f * ((i as f32).powf(2.) + (j as f32).powf(2.))).exp();

                if !weight.is_finite() {
                    continue;
                }

                let hist_idx = (grad_deg * NUM_BINS as f32 / 360.).round() as usize % NUM_BINS;
                
                histogram_raw[hist_idx] += weight * grad_mag;
            }
        }

        let mut max = 0.;
        // smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
        for n in 0 .. NUM_BINS {
            let n_1 = if n < 1 {
                NUM_BINS - 1
            } else {
                n - 1
            };

            let n_2 = if n < 2 {
                NUM_BINS + (n - 2)
            } else {
                n - 2
            };

            histogram_smooth[n] = (6. * histogram_raw[n] + 4. * (histogram_raw[n_1] + histogram_raw[(n + 1) % NUM_BINS]) + histogram_raw[n_2] + histogram_raw[(n + 2) % NUM_BINS]) / 16.;
            if histogram_smooth[n] > max {
                max = histogram_smooth[n];
            }
        }

        let mut peaks: Vec<usize> = Vec::new();
        {
            let mut hl = histogram_smooth;
            let mut hr = histogram_smooth;

            hl.rotate_left(1);
            hr.rotate_right(1);

            for (i, (h, (e2, e3))) in zip(histogram_smooth, zip(hr, hl)).enumerate() {
                if h > e2 && h < e3 {
                    peaks.push(i);
                }
            }
        }

        for p in peaks.iter() {
            let pv = histogram_smooth[*p];
            if pv < 0.8 * max {
                continue;
            }

            // Quadratic peak interpolation
            // The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
            let left = histogram_smooth[(*p - 1) % NUM_BINS];
            let right = histogram_smooth[(*p + 1) % NUM_BINS];
            
            let peak_interpolated = (*p as f32 + 0.5 * (left - right) / (left - 2. * pv + right)) % NUM_BINS as f32;
            let mut orient = 360. - peak_interpolated * 360. / NUM_BINS as f32;
            
            if (orient -360.).abs() < 1e-7 {
                orient = 0.;
            }

            let mut kp = kp.clone();
            kp.angle = orient;

            keypoint_with_orient.push(kp);
        }
        
        keypoint_with_orient
    }

    fn generate_descriptors(&self, kp: &[KeyPoint]) -> Result<()> {
        const NUM_BINS: f32 = 8.;
        const WINDOW_WIDTH: f32 = 4.;
        const SCALE_MULT: f32 = 3.;

        let imgs = &self.images[..];
        let octlen = imgs.len() as i32;

        kp
        .iter()
        .for_each(|k| {
            let (mut octave, layer, scale) = Self::unpack_octave(k);
            if octave < 0 {
                octave = octlen - octave;
            }
            
            if octave >= octlen {
                octave %= octlen;
            }
            
            let img = &imgs[octave as usize][layer];
            let (num_r, num_c) = (img.h, img.w);
            let point = ((k.prc.0 as f32 * scale).round() as usize, (k.prc.1 as f32 * scale).round() as usize);
            let bins_per_deg = NUM_BINS/ 360.;
            let angle = 360. - k.angle;
            let angle_rad = angle.to_radians();

            let cos_angle = angle_rad.cos();
            let sin_angle = angle_rad.sin();

            let weight_mul = -0.5/ ((0.5 * WINDOW_WIDTH).powf(2.));
            let mut row_bin_list = Vec::new();
            let mut col_bin_list = Vec::new();
            let mut mag_list = Vec::new();
            let mut orient_bin_list = Vec::new();

            // Descriptor window size (described by half_width) follows OpenCV convention
            let hist_width = SCALE_MULT * scale * 0.5 * k.size;
            let half_width = ((hist_width * 2_f32.sqrt() * (WINDOW_WIDTH + 1.) * 0.5).round() as i32).min(
                ((num_r.pow(2) + num_c.pow(2)) as f32).sqrt().round() as i32
            );

            for r in -half_width .. half_width + 1 {
                for c in -half_width .. half_width + 1 {
                    let (r, c) = (r as f32, c as f32);
                    let r_rot = c * sin_angle + r * cos_angle;
                    let c_rot = c * cos_angle + r * sin_angle;
                    let r_bin = (r_rot / hist_width) + 0.5 * WINDOW_WIDTH - 0.5;
                    let c_bin = (c_rot / hist_width) + 0.5 * WINDOW_WIDTH - 0.5;

                    if r_bin <= -1. || r_bin >= WINDOW_WIDTH || c_bin <= -1. || c_bin >= WINDOW_WIDTH {
                        continue;
                    }

                    let window_r = (point.1 as f32 + r).round() as u32;
                    let window_c = (point.0 as f32 + c).round() as u32;

                    if window_r == 0 || window_r >= num_r - 1 || window_c == 0 || window_c >= num_c - 1 {
                        continue;
                    } 
                    
                    let (grad_mag, grad_orient) = if let Some(img) = &img.v {
                        let dx = img.index((window_r as usize, window_c as usize + 1)) - img.index((window_r as usize, window_c as usize + 1));
                        let dy = img.index((window_r as usize - 1, window_c as usize)) - img.index((window_r as usize + 1, window_c as usize));
                        ((dx.powf(2.) + dy.powf(2.)).sqrt(), dy.atan2(dx).to_degrees() % 360.)
                    } else {
                        continue;
                    };

                    let weight = (weight_mul * ((r_rot / hist_width).powf(2.) + (c_rot / hist_width).powf(2.))).exp();

                    row_bin_list.push(r_bin);
                    col_bin_list.push(c_bin);
                    mag_list.push(weight * grad_mag);
                    orient_bin_list.push((grad_orient - angle) * bins_per_deg);
                }
            }
        });

        Ok(())
    }

    fn unpack_octave(kp: &KeyPoint) -> (i32, usize, f32) {
        let mut octave = kp.oct as i32 & 255;
        let layer = (kp.oct >> 8) & 255;
        if octave >= 128 {
            octave |= -128;
        }

        // scale = 1 / float32(1 << octave) if octave >= 0 else float32(1 << -octave)
        let scale = if octave >= 0 {
            1. / (1 << octave as i32) as f32
        } else {
            (1 << -octave as i32)  as f32
        };

        (octave, layer, scale)
    }
}

#[cfg(test)]
mod tests {
    // use std::f32::EPSILON;

    use super::Sift;

    use anyhow::{Result, Ok};

    #[test]
    fn sift() -> Result<()> {
        let mut im = Sift::new("data/1_small.png")?;
        // im.visualize();

        im.generate().unwrap();

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