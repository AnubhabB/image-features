use std::f32::{consts::E, EPSILON};
use std::iter::zip;
use std::ops::{Div, Index, Mul};

use image::{GenericImageView, Luma};
use anyhow::{Result};
use nalgebra::{OMatrix, U3, U1, Matrix3, Vector3, OVector, Dynamic};
use parking_lot::RwLock;
use rayon::{iter::{IntoParallelRefMutIterator, IndexedParallelIterator, ParallelIterator, IntoParallelRefIterator}};

use crate::utils::{ImageF32, ImageOps};


const BLUR: f32 = 0.5;
const SIGMA: f32 = 1.6;
const INTERVAL: f32 = 3.;
const CONTRAST_THESHOLD: f32 = 0.04; // from OpenCV implementation
const EIGENVALUE_RATIO: f32 = 10.; // from OpenCV implementation default

#[derive(Default)]
pub struct Sift {
    img: ImageF32,
    viz: bool,
    dogs: Vec<Vec<ImageF32>>,
    images: Vec<Vec<ImageF32>>,
    octaves: u32,
    octave_dims: Vec<(u32, u32)>
}

#[derive(Debug, Clone, Default, PartialEq)]
struct KeyPoint {
    // column, row
    pcr: (f32, f32),
    oct: usize,
    size: f32,
    res: f32,
    angle: f32
}

impl Sift {
    pub fn new(input_path: &str) -> Result<Self> {
        let opn = image::open(input_path)?;
        let dim = opn.dimensions();

        let mut img = ImageF32::new(dim.0, dim.1);

        // the default greyscale implementation of 
        for c in 0 .. dim.0 - 1 {
            for r in 0 .. dim.1 - 1 {
                let px = opn.get_pixel(c, r).0;
                let g = (0.299 * px[0] as f32) + (0.587 * px[1] as f32) + (0.114 * px[2] as f32);
                // img.put_pixel(c, r, Rgba([g, g, g, 255]));  
                let pxl = Luma::<f32>::from([g]);
                img.put_pixel(c, r, pxl);
            }
        }

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
        self.compute_octaves();
        self.generate_images(&gaussian_kernels[..])?;
        self.dogs()?;

        let keypoints = self.scale_space_extrema()?;
        let desciptors = self.generate_descriptors(&keypoints[..])?;
        
        println!("{:?} {}", keypoints.len(), desciptors.len());

        Ok(())
    }

    fn sift_base_image(&self, sigma: f32, b: f32) -> ImageF32 {
        let sigma_diff = (sigma.powf(2.) - (2. * b).powf(2.)).sqrt();

        self.img.clone()
            .resize(self.img.width() * 2, self.img.height() * 2)
            .blur(sigma_diff)
    }

    fn compute_octaves(&mut self) {
        let dim = self.img.dimensions();
        let min = if dim.0 < dim.1 {
            dim.0 as f32
        } else {
            dim.1 as f32
        } * 2.; // multiplying by this because the base image is going to change

        self.octaves = ((min.log(E) / 2_f32.log(E)) - 1.).floor() as u32;
    }

    fn generate_images(&mut self, kernels: &[f32]) -> Result<()> {
        let gaussian_images = RwLock::new(vec![
            vec![
                ImageF32::new(0, 0);
                kernels.len()
            ];
            self.octaves as usize
        ]);

        let mut octave_dims = vec![(0, 0); self.octaves as usize];
        
        let mut img = self.sift_base_image(SIGMA, BLUR);
        for i in 0 .. self.octaves as usize {
            {
                let mut gi = gaussian_images.write();
                gi[i][0] = img.clone();
            }
            octave_dims[i] = (img.width(), img.height());

            kernels.par_iter().skip(1).enumerate().for_each(|(j, k)| {
                let m = img.blur(*k);
                let mut gi = gaussian_images.write();
                gi[i][j + 1] = m;
            });
            
            
            let tgimg = {
                let read = gaussian_images.read();
                read[i][kernels.len() - 3].clone()
            };
            
            img = tgimg.resize( tgimg.width() / 2, tgimg.height() / 2);
        }

        self.images = gaussian_images.into_inner();
        self.octave_dims = octave_dims;

        Ok(())
    }

    // Difference-of-gradients
    fn dogs(&mut self) -> Result<()> {
        let mut dogs = vec![
            vec![
                ImageF32::new(0, 0);
                self.images[0].len() - 1
            ];
            self.octaves as usize
        ];
        for (o, oct_img) in self.images.iter().enumerate() {
            for (i, (first, second)) in zip(oct_img, &oct_img[1..]).enumerate() {
                // 
                let sub = second.subtract(first);
                // for r in 2 .. 4 {
                //     for c in 10 .. 24 {
                //         println!("{} {} {}", sub.get_pixel(c, r).0[0], second.get_pixel(c, r).0[0], first.get_pixel(c, r).0[0]);
                //     }
                // }
                dogs[o][i] = sub;
            }

            // let l = dogs[o].len();
            // println!("{l}");
            // for r in 2 .. 4 {
            //     for c in 10 .. 24 {
            //         println!("{}", dogs[o][0].get_pixel(c, r).0[0])
            //     }
            // }
        }

        self.dogs = dogs;

        Ok(())
    }

    // Find pixel positions of all scale-space extrema in the image pyramid
    fn scale_space_extrema(&self) -> Result<Vec<KeyPoint>> {
        let threshold = (0.5 * CONTRAST_THESHOLD / INTERVAL * 255.).floor(); // from OpenCV implementation
        let mut keypoints = Vec::new();

        for (o, oct) in self.dogs.iter().enumerate() {
            let (octw, octh) = self.octave_dims[o];

            for (img_idx, (img0, (img1, img2))) in zip(oct, zip(&oct[1..], &oct[2..])).enumerate() {
                for r in 1 .. octh - 2 {
                    for c in 1 .. octw - 2 {
                        let m0 = Matrix3::from_column_slice(&[
                            img0.index((c - 1, r - 1)).0[0], img0.index((c, r - 1)).0[0], img0.index((c + 1, r - 1)).0[0],
                            img0.index((c - 1, r)).0[0], img0.index((c, r)).0[0], img0.index((c + 1, r)).0[0],
                            img0.index((c - 1, r + 1)).0[0], img0.index((c, r + 1)).0[0], img0.index((c + 1, r + 1)).0[0],
                        ]);
                        let m1 = Matrix3::from_column_slice(&[
                            img1.index((c - 1, r - 1)).0[0], img1.index((c, r - 1)).0[0], img1.index((c + 1, r - 1)).0[0],
                            img1.index((c - 1, r)).0[0], img1.index((c, r)).0[0], img1.index((c + 1, r)).0[0],
                            img1.index((c - 1, r + 1)).0[0], img1.index((c, r + 1)).0[0], img1.index((c + 1, r + 1)).0[0],
                        ]);
                        let m2 = Matrix3::from_column_slice(&[
                            img2.index((c - 1, r - 1)).0[0], img2.index((c, r - 1)).0[0], img2.index((c + 1, r - 1)).0[0],
                            img2.index((c - 1, r)).0[0], img2.index((c, r)).0[0], img2.index((c + 1, r)).0[0],
                            img2.index((c - 1, r + 1)).0[0], img2.index((c, r + 1)).0[0], img2.index((c + 1, r + 1)).0[0],
                        ]);
                        
                        if !Self::is_pixel_extremum(m0, m1, m2, threshold) {
                            continue;
                        }

                        if let Some((kp, imidx)) = self.localize_extremum(img_idx + 1, o, (c, r), octw, octh) {
                            let mut kp_o = self.compute_keypoint_orientation(&kp, o, octw, octh, imidx);
                            if kp_o.is_empty() {
                                continue;
                            }

                            keypoints.append(&mut kp_o);
                        }
                    }
                }
            }
        }
        
        // Sort keypoints and remove duplicates if any
        {
            keypoints.sort_unstable_by(|k1, k2| {
                if k1.pcr.0 != k2.pcr.0 && let Some(c) = k1.pcr.0.partial_cmp(&k2.pcr.1) {
                    return c;
                }
    
                if k1.pcr.1 != k2.pcr.1 && let Some(c) = k2.pcr.1.partial_cmp(&k2.pcr.1) {
                    return c;
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
            keypoints.dedup();

            // rescaling keypoints as per https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-2-c4350274be2b
            // not sure of the `WHY` for the octave though
            keypoints.iter_mut()
            .for_each(|k| {
                k.pcr = (k.pcr.0/ 2., k.pcr.1/2.);
                k.size *= 0.5;
                k.oct = (k.oct & !255) | ((k.oct - 1) & 255);
            });
        }


        // let kp = keypoints.read().clone();

        // Ok(kp)
        Ok(keypoints)
    }

    fn localize_extremum(&self, im1_idx: usize, octave_idx: usize, px_idx: (u32, u32), oct_w: u32, oct_h: u32) -> Option<(KeyPoint, usize)> {
        let divisor: f32 = 255.;
        let mut converge = false;

        let mut c = px_idx.0;
        let mut r = px_idx.1;
        let mut imidx = im1_idx;

        let mut grad: Vector3<f32> = Vector3::default();
        let mut hess: Matrix3<f32> = Matrix3::default();
        let mut lstq: OMatrix<f32, U3, U1> = OMatrix::default();

        let mut pxstack: [Matrix3<f32>; 3] = [Matrix3::<f32>::default(); 3];

        for _ in 0 .. 5 {
            pxstack = [
                Matrix3::from_column_slice(
                    &[
                        self.dogs[octave_idx][imidx - 1].get_pixel(c - 1, r - 1).0[0], self.dogs[octave_idx][imidx - 1].get_pixel(c, r - 1).0[0], self.dogs[octave_idx][imidx - 1].get_pixel(c + 1, r - 1).0[0],
                        self.dogs[octave_idx][imidx - 1].get_pixel(c - 1, r).0[0], self.dogs[octave_idx][imidx - 1].get_pixel(c, r).0[0], self.dogs[octave_idx][imidx - 1].get_pixel(c + 1, r).0[0],
                        self.dogs[octave_idx][imidx - 1].get_pixel(c - 1, r + 1).0[0], self.dogs[octave_idx][imidx - 1].get_pixel(c, r + 1).0[0], self.dogs[octave_idx][imidx - 1].get_pixel(c + 1, r + 1).0[0]
                    ]
                ).div(divisor),
                Matrix3::from_column_slice(
                    &[
                        self.dogs[octave_idx][imidx].get_pixel(c - 1, r - 1).0[0], self.dogs[octave_idx][imidx].get_pixel(c, r - 1).0[0], self.dogs[octave_idx][imidx].get_pixel(c + 1, r - 1).0[0],
                        self.dogs[octave_idx][imidx].get_pixel(c - 1, r).0[0], self.dogs[octave_idx][imidx].get_pixel(c, r).0[0], self.dogs[octave_idx][imidx].get_pixel(c + 1, r).0[0],
                        self.dogs[octave_idx][imidx].get_pixel(c - 1, r + 1).0[0], self.dogs[octave_idx][imidx].get_pixel(c, r + 1).0[0], self.dogs[octave_idx][imidx].get_pixel(c + 1, r + 1).0[0]
                    ]
                ).div(divisor),
                Matrix3::from_column_slice(
                    &[
                        self.dogs[octave_idx][imidx + 1].get_pixel(c - 1, r - 1).0[0], self.dogs[octave_idx][imidx + 1].get_pixel(c, r - 1).0[0], self.dogs[octave_idx][imidx + 1].get_pixel(c + 1, r - 1).0[0],
                        self.dogs[octave_idx][imidx + 1].get_pixel(c - 1, r).0[0], self.dogs[octave_idx][imidx + 1].get_pixel(c, r).0[0], self.dogs[octave_idx][imidx + 1].get_pixel(c + 1, r).0[0],
                        self.dogs[octave_idx][imidx + 1].get_pixel(c - 1, r + 1).0[0], self.dogs[octave_idx][imidx + 1].get_pixel(c, r + 1).0[0], self.dogs[octave_idx][imidx + 1].get_pixel(c + 1, r + 1).0[0]
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

            c += lstq[0].round() as u32;
            r += lstq[1].round() as u32;
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
        
        if xy_hess_det <= 0. || EIGENVALUE_RATIO * xy_hess_trace.powf(2.) >= (EIGENVALUE_RATIO + 1.).powf(2.) * xy_hess_det {
            return None;
        }
        
        Some((KeyPoint{
            pcr: (
                (c as f32 + lstq[0]) * 2_f32.powf(octave_idx as f32),
                (r as f32 + lstq[1]) * 2_f32.powf(octave_idx as f32)
            ),
            oct: octave_idx + imidx * 2_u32.pow(8) as usize + (((lstq[2] + 0.5) * 255.).round() * 2_f32.powf(16.)) as usize,
            size: SIGMA * 2_f32.powf((imidx as f32 + lstq[2]) / INTERVAL) * 2_f32.powf(octave_idx as f32 + 1.),
            res: val_at_extremum.abs(),
            angle: 0.
        }, imidx))
        // None
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

    fn is_pixel_extremum(im0: Matrix3<f32>, im1: Matrix3<f32>, im2: Matrix3<f32>, threshold: f32) -> bool {
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
        let dx = 0.5 * (px_cube[1].index((2, 1)) - px_cube[1].index((0, 1)));
        let dy = 0.5 * (px_cube[1].index((1, 2)) - px_cube[1].index((1, 0)));
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

        let dxx = px_cube[1].index((2, 1)) - 2. * px_cent + px_cube[1].index((0, 1));
        let dyy = px_cube[1].index((1, 2)) - 2. * px_cent + px_cube[1].index((1, 0));
        let dss = px_cube[2].index((1, 1)) - 2. * px_cent + px_cube[0].index((1, 1));
        let dxy = 0.25 * (px_cube[1].index((2, 2)) - px_cube[1].index((0, 2)) - px_cube[1].index((2, 0)) + px_cube[1].index((0, 0)));
        let dxs = 0.25 * (px_cube[2].index((2, 1)) - px_cube[2].index((0, 1)) - px_cube[0].index((2, 1)) + px_cube[0].index((0, 1)));
        let dys = 0.25 * (px_cube[2].index((1, 2)) - px_cube[2].index((1, 0)) - px_cube[0].index((1, 2)) + px_cube[0].index((1, 0)));

        Matrix3::from_column_slice(&[
            dxx, dxy, dxs,
            dxy, dyy, dys,
            dxs, dys, dss
        ])
    }


    // Compute orientations for each keypoint
    fn compute_keypoint_orientation(&self, kp: &KeyPoint, octave_idx: usize, oct_w: u32, oct_h: u32, img_idx: usize) -> Vec<KeyPoint> {
        const NUM_BINS: usize = 36;

        let scale = 1.5 * kp.size / 2_f32.powf(octave_idx as f32 + 1.);
        let radius = (3. * scale).round() as i32;
        let weight_f = -0.5 / scale.powf(2.);

        let mut histogram_raw = [0.; NUM_BINS];
        let mut histogram_smooth = [0.; NUM_BINS];

        let mut keypoint_with_orient = Vec::<KeyPoint>::new();

        // // let gaus_in_oct = self.images.len() / self.octaves as usize;
        // // let img_idx = (gaus_in_oct * octave_idx) + kp.idx_oct;
        let img = &self.images[octave_idx][img_idx];
        // // let img_data = &img.v[..];

        for r in -radius .. radius + 1 {
            let region_y = (kp.pcr.1 as f32 / 2_f32.powf(octave_idx as f32)).round() as i32 + r;
            if region_y <= 0 || region_y >= oct_h as i32 - 1 {
                continue;
            }

            for c in -radius .. radius + 1 {
                let region_x = (kp.pcr.0 as f32 / 2_f32.powf(octave_idx as f32)).round() as i32 + c;
                if region_x <= 0 || region_x >= oct_w as i32 - 1 {
                    continue;
                }

                let dc = img.get_pixel(region_x as u32 + 1, region_y as u32).0[0] - img.get_pixel(region_x as u32 - 1, region_y as u32).0[0];
                let dr = img.get_pixel(region_x as u32, region_y as u32 + 1).0[0] - img.get_pixel(region_x as u32, region_y as u32 - 1).0[0];

                let grad_mag = (dc.powf(2.) + dr.powf(2.)).sqrt();
                let grad_deg = dr.atan2(dc).to_degrees();
                
                let weight = (weight_f * ((r as f32).powf(2.) + (c as f32).powf(2.))).exp();

                let hist_idx = (grad_deg * NUM_BINS as f32 / 360.).round() as i32;
                let hist_idx = if hist_idx < 0 {
                    (NUM_BINS as i32 + hist_idx) as usize
                } else {
                    hist_idx as usize
                };
                
                histogram_raw[hist_idx % NUM_BINS as usize] += weight * grad_mag;
            }
        }

        let mut max = 0.;
    
        for n in 0 .. NUM_BINS {
            let n_1 = if n == 0 {
                35
            } else {
                n - 1
            };

            let n_2 = if (n as i32 - 2) < 0 {
                (NUM_BINS as i32 + (n as i32 - 2))  as usize
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
                if h > e2 && h > e3 {
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
        // vec![]
    }

    fn generate_descriptors(&self, kp: &[KeyPoint]) -> Result<Vec<Vec<u8>>> {
        const NUM_BINS: f32 = 8.;
        const WINDOW_WIDTH: f32 = 4.;
        const SCALE_MULT: f32 = 3.;
        const DESCRIPTOR_MAX_VAL: f32 = 0.2;

        let imgs = &self.images[..];
        let octlen = self.octaves as i32;
        let bins_per_deg = NUM_BINS/ 360.;

        let mut descriptors = vec![vec![0]];

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
            
            let img = &imgs[octave as usize + 1][layer];
            let (num_c, num_r) = self.octave_dims[octave as usize + 1];

            let point = ((k.pcr.0 as f32 * scale).round() as usize, (k.pcr.1 as f32 * scale).round() as usize);
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
                    
                    let (grad_mag, grad_orient) = {
                        let dx = img.get_pixel(window_c + 1, window_r).0[0] - img.get_pixel(window_c - 1, window_r).0[0];
                        let dy = img.get_pixel(window_c, window_r + 1).0[0] - img.get_pixel(window_c, window_r - 1).0[0];
                        ((dx.powf(2.) + dy.powf(2.)).sqrt(), dy.atan2(dx).to_degrees() % 360.)
                    };

                    let weight = (weight_mul * ((r_rot / hist_width).powf(2.) + (c_rot / hist_width).powf(2.))).exp();

                    row_bin_list.push(r_bin);
                    col_bin_list.push(c_bin);
                    mag_list.push(weight * grad_mag);
                    orient_bin_list.push((grad_orient - angle) * bins_per_deg);
                }
            }

            // let mut histogram_tensor = zeros((window_width + 2, window_width + 2, num_bins))
            let mut hist_tensor = vec![
                vec![
                    vec![0.; NUM_BINS as usize];
                    (WINDOW_WIDTH + 2.) as usize
                ];
                (WINDOW_WIDTH + 2.) as usize
            ];

            // for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            for (rb, (cb, (mag, orient))) in zip(row_bin_list, zip( col_bin_list, zip(mag_list, orient_bin_list))) {
                // Smoothing via trilinear interpolation
                // Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
                // Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
                let (rbf, cbf, mut orientf) = (rb.floor(), cb.floor(), orient.floor());
                let (rfract, cfract, orientfract) = (rb - rbf, cb - cbf, orient - orientf);
                if orientf < 0. {
                    orientf += NUM_BINS
                }
                    
                if orientf >= NUM_BINS {
                    orientf -= NUM_BINS
                }

                let c1 = mag * rfract;
                let c0 = mag * (1. - rfract);
                let c11 = c1 * cfract;
                let c10 = c1 * (1. - cfract);
                let c01 = c0 * cfract;
                let c00 = c0 * (1. - cfract);
                let c111 = c11 * orientfract;
                let c110 = c11 * (1. - orientfract);
                let c101 = c10 * orientfract;
                let c100 = c10 * (1. - orientfract);
                let c011 = c01 * orientfract;
                let c010 = c01 * (1. - orientfract);
                let c001 = c00 * orientfract;
                let c000 = c00 * (1. - orientfract);

                let rbf_usize = rbf as usize;
                let cbf_usize = cbf as usize;
                let obf_usize = orientf as usize;

                hist_tensor[rbf_usize + 1][cbf_usize + 1][obf_usize] += c000;
                hist_tensor[rbf_usize + 1][cbf_usize + 1][(obf_usize + 1) % NUM_BINS as usize] += c001;
                hist_tensor[rbf_usize + 1][cbf_usize + 2][obf_usize] += c010;
                hist_tensor[rbf_usize + 1][cbf_usize + 2][(obf_usize + 1) % NUM_BINS as usize] += c011;
                hist_tensor[rbf_usize + 2][cbf_usize + 1][obf_usize] += c100;
                hist_tensor[rbf_usize + 2][cbf_usize + 1][(obf_usize + 1) % NUM_BINS as usize] += c101;
                hist_tensor[rbf_usize + 2][cbf_usize + 2][obf_usize] += c110;
                hist_tensor[rbf_usize + 2][cbf_usize + 2][(obf_usize + 1) % NUM_BINS as usize] += c111;
            }

            let mut descriptor_vec = vec![0.];
            let mut desc_sum = 0.;
            for (i, j) in hist_tensor.iter().enumerate() {
                if i == 0 || i == hist_tensor.len() - 1 {
                    continue;
                }

                for (k, l) in j.iter().enumerate() {
                    if k == 0 || k == l.len() - 1 {
                        continue;
                    }

                    for p in l.iter() {
                        descriptor_vec.push(*p);
                        desc_sum += *p;
                    }
                }
            }

            if desc_sum == 0. {
                return;
            }
            let mut ov = OVector::<f32, Dynamic>::from_vec(descriptor_vec);
            let desc_norm = ov.norm();
            let threshold = desc_norm * DESCRIPTOR_MAX_VAL;
            
            for o in ov.iter_mut() {
                if *o <= threshold {
                    continue;
                }

                *o = threshold;
            }

            ov = ov.div(desc_norm.max(1e-7)).mul(512.);

            let mut desc: Vec<u8> = vec![0; ov.len()];
            for (i, o) in ov.iter().enumerate() {
                desc[i] = if *o > 255. {
                    255
                } else if *o < 0.{
                    0
                } else {
                    o.round() as u8
                }
            }

            descriptors.push(desc);
        });

        Ok(descriptors)
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
    // use std::f32::EPSIL

    use super::Sift;

    use anyhow::{Result, Ok};
    // use image::imageops::resize;

    #[test]
    fn sift() -> Result<()> {
        let mut im = Sift::new("data/1_small.png")?;
        // im.visualize();
        
        im.generate()?;

        Ok(())
    }

    // #[test]
    // fn resize_img() -> Result<()> {
    //     // let im = image::open("data/1_mini.png")?;

    //     // let k = resize(&im, im.width() / 2, im.height() / 2, image::imageops::FilterType::Nearest);

    //     let _im = Sift::new("data/1_mini.png")?;
    //     let l = resize(&_im.img, _im.img.width()/ 2, _im.img.height() / 2, image::imageops::FilterType::Nearest);

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