use std::{sync::{Arc}, f32::consts::E};
use anyhow::Result;
use image::{save_buffer_with_format, ColorType, DynamicImage, GenericImageView, imageops};
use parking_lot::RwLock;
use rayon::{prelude::*, scope};

pub struct Image {
    w: usize,
    h: usize,
    img: DynamicImage,
    viz: bool
}

const BLUR: f32 = 0.5;
const SIGMA: f32 = 1.6;
const INTERVAL: f32 = 3.;
const CONTRAST_THESHOLD: f32 = 0.04; // from OpenCV implementation

impl Image {
    pub fn new(input_path: &str) -> Result<Image> {
        // Image { input_path }
        let img = image::open(input_path)?;

        Ok(Image {
            w: img.width() as usize,
            h: img.height() as usize,
            img,
            viz: false
        })
    }

    pub fn visualize(&mut self) {
        self.viz = true;
    }

    /// the filter looks like:
    ///     [
    ///          1, 1, 1,
    ///          0, 0, 0,
    ///         -1,-1,-1
    ///     ]
    /// So, basically the multiplication of pixels on (top, top-1, top+1) + (-bottom, -bottom+1, -bottom-1)
    pub fn prewitt_h(&self) -> Result<Vec<u8>> {
        let grey = self.to_grey();

        let len = self.get_bytes_size();
        let widtoparse = self.w - 1;

        let mut d = vec![0; len];

        d.par_iter_mut()
            .skip(self.w)
            .take(len - (self.w * 2))
            .enumerate()
            .for_each(|(i, p)| {
                let m = i % self.w;
                // skip first column and last column
                if m == 0 || m == widtoparse {
                    return;
                }

                let x = i % self.w;
                let y = (i / self.w) + 1;

                let top = Self::get_px_val_grey(&grey, (x - 1, y - 1))
                    + Self::get_px_val_grey(&grey, (x, y - 1))
                    + Self::get_px_val_grey(&grey, (x + 1, y - 1));

                let bot = Self::get_px_val_grey(&grey, (x - 1, y + 1))
                    + Self::get_px_val_grey(&grey, (x, y + 1))
                    + Self::get_px_val_grey(&grey, (x + 1, y + 1));

                *p = (top - bot).unsigned_abs() as u8;
            });
        Ok(d)
    }

    /// the filter looks like:
    ///     [
    ///          1, 0,-1,
    ///          1, 0,-1,
    ///          1, 0,-1
    ///     ]
    /// So, basically the multiplication of pixels on (left, left-1, left+1) + (-right, -right+1, -right-1)
    pub fn prewitt_v(&self) -> Result<Vec<u8>> {
        let grey = self.to_grey();

        let len = self.get_bytes_size();
        let widtoparse = self.w - 1;

        let mut d = vec![0; len];

        // for (i, p) in d
        d.par_iter_mut()
            .skip(self.w) // skip first row
            .take(len - (self.w * 2)) // skip last row
            .enumerate()
            .for_each(|(i, p)| {
                let m = i % self.w;

                // skip first column and last column
                if m == 0 || m == widtoparse {
                    return;
                }

                let x = i % self.w;
                let y = (i / self.w) + 1;

                let lft = Self::get_px_val_grey(&grey, (x - 1, y - 1))
                    + Self::get_px_val_grey(&grey, (x - 1, y))
                    + Self::get_px_val_grey(&grey, (x - 1, y + 1));

                let rit = Self::get_px_val_grey(&grey, (x + 1, y - 1))
                    + Self::get_px_val_grey(&grey, (x + 1, y))
                    + Self::get_px_val_grey(&grey, (x + 1, y + 1));

                *p = (lft - rit).unsigned_abs() as u8;
            });

        Ok(d)
    }

    pub fn prewitt(&self) -> Result<Vec<u8>> {
        let grey = self.to_grey();

        let len = self.get_bytes_size();
        let widtoparse = self.w - 1;

        let mut d = vec![0; len];
        
        d.par_iter_mut()
            .skip(self.w) // skip first row
            .take(len - (self.w * 2)) // skip last row
            .enumerate()
            .for_each(|(i, p)| {
                let m = i % self.w;

                // skip first column and last column
                if m == 0 || m == widtoparse {
                    return;
                }

                let x = i % self.w;
                let y = (i / self.w) + 1;

                let lft = Self::get_px_val_grey(&grey, (x - 1, y - 1))
                    + Self::get_px_val_grey(&grey, (x - 1, y))
                    + Self::get_px_val_grey(&grey, (x - 1, y + 1));

                let rit = Self::get_px_val_grey(&grey, (x + 1, y - 1))
                    + Self::get_px_val_grey(&grey, (x + 1, y))
                    + Self::get_px_val_grey(&grey, (x + 1, y + 1));

                let top = Self::get_px_val_grey(&grey, (x - 1, y - 1))
                    + Self::get_px_val_grey(&grey, (x, y - 1))
                    + Self::get_px_val_grey(&grey, (x + 1, y - 1));

                let bot = Self::get_px_val_grey(&grey, (x - 1, y + 1))
                    + Self::get_px_val_grey(&grey, (x, y + 1))
                    + Self::get_px_val_grey(&grey, (x + 1, y + 1));

                *p = (((lft - rit) + (top - bot)).unsigned_abs() / 2) as u8;
            });

        Ok(d)
    }

    /// Histogram Oriented Gradients
    /// [
    ///  0, 1, 2, 3, 4, 5, 6, 7,
    ///  8, 9,10,11,12,13,14,15,
    /// 16,17,18,19,20,21,22,23,
    /// 24,25,26,27,28,29,30,31,
    /// 32,33,34,35,36,37,38,39,
    /// 40,41,42,43,44,45,46,47,
    /// 48,49,50,51,52,53,54,55,
    /// 56,57,58,59,60,61,62,63
    /// ]
    pub fn hog(&self) -> Result<Vec<f32>> {
        let hists = Arc::new(RwLock::new(vec![vec![0.; 9]; self.w/8 * self.h/8]));
        let num_blocks_per_row = self.w / 8;

        scope(|s| {
            for y in 1 .. self.h - 1 {
                for x in 1 .. self.w - 1 {
                    let hists = hists.clone();

                    s.spawn(move |_| {
                        let (mag, dir) = if let Ok(m) = self.calc_magnitude_direction(x, y) {
                            m
                        } else {
                            (0., 0.)
                        };
    
                        let block_row = y/ 8;
                        let block_col = x/ 8;
                        let block_idx = (block_row * num_blocks_per_row) + block_col;
    
                        let bin = (dir/20.).floor();
                        let j_val = (((bin + 1.) * 20. - dir)/ 20.) * mag;
    
                        let j_1_val = mag - j_val;
                        let j_1_bin = if j_val > 0. && dir < 160. {
                                bin + 1.
                        } else {
                            0.
                        };
    
                        {
                            let mut hist = hists.write();
                            if let Some(h) = hist.get_mut(block_idx) {
                                h[bin as usize] += j_val;
                                if j_1_val > 0. {
                                    h[j_1_bin as usize] += j_1_val;
                                }
                            }
                        }
                    });
                }
            }
        });

        let mut feature_vec = Vec::new();

        for y in 0 .. self.h/8 {
            for x in 0 .. self.w/8 {
                let elm = RwLock::new(0.);
                let tvc = RwLock::new(vec![0.; 36]);

                scope(|s| {
                    // grid at (x, y)
                    s.spawn(|_| {
                        let r = hists.read();

                        if let Some(e) = r.get(x * y) {
                            e.par_iter().enumerate().for_each(|(i, d)| {
                                {
                                    let mut w = elm.write();
                                    *w += d.powf(2.);
                                }
                                {
                                    let mut w = tvc.write();
                                    w[i] = *d;
                                }
                            });
                        }
                    });

                    // grid at // grid at (x + 1, y)
                    s.spawn(|_| {
                        const ROOT: usize = 1;
                        let r = hists.read();

                        if let Some(e) = r.get((x + 1) * y) {
                            e.par_iter().enumerate().for_each(|(i, d)| {
                                {
                                    let mut w = elm.write();
                                    *w += d.powf(2.);
                                }
                                {
                                    let mut w = tvc.write();
                                    w[(ROOT * 9) + i] = *d;
                                }
                                
                            });
                        }
                    });

                    // grid at // grid at (x, y + 1)
                    s.spawn(|_| {
                        const ROOT: usize = 2;
                        let r = hists.read();
                        if let Some(e) = r.get(x * (y + 1)) {
                            e.par_iter().enumerate().for_each(|(i, d)| {
                                {
                                    let mut w = elm.write();
                                    *w += d.powf(2.);
                                }
                                {
                                    let mut w = tvc.write();
                                    w[(ROOT * 9) + i] = *d;
                                }
                            });
                        }
                    });

                    // grid at // grid at (x + 1, y + 1)
                    s.spawn(|_| {
                        const ROOT: usize = 3;
                        let r = hists.read();
                        if let Some(e) = r.get((x + 1) * (y + 1)) {
                            e.par_iter().enumerate().for_each(|(i, d)| {
                                {
                                    let mut w = elm.write();
                                    *w += d.powf(2.);
                                }
                                {
                                    let mut w = tvc.write();
                                    w[(ROOT * 9) + i] = *d;
                                }
                            });
                        }
                    });
                });

                {
                    let mut l = tvc.write();
                    let elm = elm.into_inner().sqrt();

                    l.par_iter_mut().for_each(|d| {
                        let r = *d;
                        *d = r/ elm;
                    });
                }
                let mut f = tvc.into_inner();
                feature_vec.append(&mut f);
            }
        }

        Ok(feature_vec)
    }

    fn calc_magnitude_direction(&self, x: usize, y: usize) -> Result<(f32, f32)> {
        let (x, y) = (x as u32, y as u32);
        let right = self.img.get_pixel(x + 1, y);
        let left = self.img.get_pixel(x - 1, y);
        let bottom = self.img.get_pixel(x, y + 1);
        let top = self.img.get_pixel(x, y - 1);

        let mag = Arc::new(RwLock::new(vec![(0., 0., 0.); 3]));

        scope(|s| {
            // calc for r channel
            s.spawn(|_| {
                let r_gx = right.0[0] as f32 - left.0[0] as f32;
                let r_gy = bottom.0[0] as f32 - top.0[0] as f32;

                // r_mag = (r_gx.powf(2.) + r_gy.powf(2.)).sqrt();
                let mut l = mag.write();
                l[0] = (r_gx, r_gy, (r_gx.powf(2.) + r_gy.powf(2.)).sqrt());

            });

            // calc for g channel
            s.spawn(|_| {
                let g_gx = right.0[1] as f32 - left.0[1] as f32;
                let g_gy = bottom.0[1] as f32 - top.0[1] as f32;

                let mut l = mag.write();
                l[1] = (g_gx, g_gy, (g_gx.powf(2.) + g_gy.powf(2.)).sqrt());
            });

            // calc for g channel
            s.spawn(|_| {
                let b_gx = right.0[2] as f32 - left.0[2] as f32;
                let b_gy = bottom.0[2] as f32 - top.0[2] as f32;

                let mut l = mag.write();
                l[2] = (b_gx, b_gy, (b_gx.powf(2.) + b_gy.powf(2.)).sqrt());
            });
        });

        {
            let mut l = mag.write();
            l.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        }

        let (m, d) = {
            let l = mag.read();

            let d = if l[2].0 > 0. {
                (l[2].1/l[2].0).atan().to_degrees().abs()
            } else {
                0.
            };

            (l[2].2, d)
        };
        
        Ok((m, d))
    }

    
    fn to_grey(&self) -> DynamicImage {
        self.img.grayscale()
    }

    fn get_bytes_size(&self) -> usize {
        self.w * self.h
    }

    fn get_px_val_grey(im: &DynamicImage, v: (usize, usize)) -> i32 {
        im.get_pixel(v.0 as u32, v.1 as u32)[0] as i32
    }

    pub fn draw_grey(&self, name: &str, buf: &[u8]) -> Result<()> {
        save_buffer_with_format(
            name,
            buf,
            self.w as u32,
            self.h as u32,
            ColorType::L8,
            image::ImageFormat::Png,
        )?;

        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
struct OctaveImage {
    w: u32,
    h: u32,
    v: Vec<u8>
}

impl Image {
    /// Scale Invarient Feature Transform
    /// 
    pub fn sift(&self) -> Result<()> {
        let gaussian_kernels = Self::sift_gaussian_kernels(SIGMA, INTERVAL);

        let octaves = self.sift_compute_octaves();
        let images = self.sift_generate_images(octaves, &gaussian_kernels[..])?;
        let dogs = self.sift_dogs(&images[..], octaves)?;

        let keypoints = self.sift_keypoints(octaves, &images[..], &dogs[..])?;

        Ok(())
    }

    fn sift_base_image(&self, sigma: f32, b: f32) -> DynamicImage {
        let sigma_diff = sigma.powf(2.) - (2. * b).powf(2.).sqrt();
        
        self.to_grey()
            .resize((self.w * 2) as u32, (self.h * 2) as u32, imageops::FilterType::Nearest)
            .blur(sigma_diff)
    }

    fn sift_compute_octaves(&self) -> u32 {
        let min = if self.w < self.h {
            self.w as f32
        } else {
            self.h as f32
        };
        let b: f32 = 2.;

        ((min.log(E) / b.log(E)) - 1.).floor() as u32
    }

    fn sift_gaussian_kernels(sigma: f32, intervals: f32) -> Vec<f32> {
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

    fn sift_generate_images(&self, octaves: u32, kernels: &[f32]) -> Result<Vec<OctaveImage>> {
        let kernel_len = kernels.len() as u32;

        let mut bimg = self.sift_base_image(SIGMA, BLUR);
        let imgs = Arc::new(RwLock::new(vec![OctaveImage::default(); (octaves * kernel_len) as usize]));
        
        // generating base images for each octave has to happen in sequence because each octave is dependent on the last image of the previous octave
        for i in 0 .. octaves {
            {
                let first = (i * kernel_len) as usize;
                let mut l = imgs.write();
                l[first] = OctaveImage {
                    w: bimg.width(),
                    h: bimg.height(),
                    v: bimg.as_bytes().to_vec()
                };

                if self.viz {
                    // this is a sample visualization
                    save_buffer_with_format(
                        format!("data/sift/{i}_0.png").as_str(),
                        bimg.as_bytes(),
                        bimg.dimensions().0,
                        bimg.dimensions().1,
                        ColorType::L8,
                        image::ImageFormat::Png,
                    ).unwrap();
                }
            }

            let timg = RwLock::new(DynamicImage::new_luma8(0, 0));

            kernels.par_iter()
                .skip(1)
                .enumerate()
                .for_each(|(ki, k)| {
                    let ti = ki + 1;
                    let idx = (i * kernel_len) + ti as u32;

                    let gimg = bimg.blur(*k);
                    let g_bytes = gimg.as_bytes();
                    {
                        let mut w = imgs.write();
                        w[idx as usize] = OctaveImage {
                            w: gimg.width(),
                            h: gimg.height(),
                            v: g_bytes.to_vec()
                        }
                    }

                    let dims = gimg.dimensions();

                    if self.viz {
                        // this is a sample visualization
                        save_buffer_with_format(
                            format!("data/sift/{i}_{ki}.png").as_str(),
                            g_bytes,
                            dims.0,
                            dims.1,
                            ColorType::L8,
                            image::ImageFormat::Png,
                        ).unwrap();
                    }

                    if ti == 3 {
                        let mut w = timg.write();
                        *w = gimg;
                    }
                });
            let timg = timg.into_inner();
            // let (w, h) = timg.dimensions();
                
            bimg = timg.resize(timg.width()/2, timg.height()/2, imageops::FilterType::Nearest);
        }

        let b = imgs.read();
        
        Ok(b.clone())
    }

    // Difference-of-Gradients
    fn sift_dogs(&self, images: &[OctaveImage], octaves: u32) -> Result<Vec<Vec<f32>>> {
        let img_per_oct = images.len() / octaves as usize;
        let dogs = Arc::new(RwLock::new(vec![Vec::<f32>::new(); (img_per_oct - 1) * octaves as usize]));

        println!("total dogs: {}", dogs.read().len());

        scope(|s| {
            for i in 0 .. octaves as usize {
                for j in 0 .. img_per_oct - 1 {
                    let dogs = dogs.clone();
                    s.spawn(move |_| {
                        let im0 = &images.get((i * img_per_oct) + j).unwrap().v;
                        let im1 = &images.get((i * img_per_oct) + j + 1).unwrap().v;

                        let mut v = vec![0.;im0.len()];
                        
                        v.par_iter_mut()
                            .enumerate()
                            .for_each(|(k, v)| {
                                let d = im1[k] as f32 - im0[k] as f32;
                                *v = d;
                            });
                        
                        if self.viz {
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
                            
                            let g = &images.get((i * img_per_oct) + j).unwrap();
                            save_buffer_with_format(
                                format!("data/sift/dog_{}_{}.png", i, j).as_str(),
                                &buf.into_inner(),
                                g.w,
                                g.h,
                                ColorType::L8,
                                image::ImageFormat::Png,
                            ).unwrap();
                        }
                        let mut l = dogs.write();
                        
                        let didx = (i * (img_per_oct - 1)) + j;
                        l[didx] = v;
                    });
                }
            }
        });
        
        let d = dogs.read().clone();

        Ok(d)
    }

    // First, we generate scale-space extrema
    fn sift_keypoints(&self, octaves: u32, images: &[OctaveImage], dogs: &[Vec<f32>] ) -> Result<()> {
        let kp = self.sift_scale_space_extrema(octaves, images, dogs)?;
        Ok(())
    }

    // Find pixel positions of all scale-space extrema in the image pyramid
    fn sift_scale_space_extrema(&self, octaves: u32, images: &[OctaveImage], dogs: &[Vec<f32>]) -> Result<()> {
        let threshold = (0.5 * 255. * CONTRAST_THESHOLD / INTERVAL).floor(); // from OpenCV implementation

        let octaves = octaves as usize;
        let imgs_in_octave = images.len() / octaves; 
        let dogs_in_octave = dogs.len() / octaves;

        scope(|s| {
            for o in 0 .. octaves {
                let (w, h) = {
                    let first_in_octave = images.get(o * imgs_in_octave).unwrap();
                    (first_in_octave.w as usize, first_in_octave.h as usize)
                };
                
                for i in 0 .. dogs_in_octave - 2 {
                    let img0 = (o * dogs_in_octave) + i;
                    let (img1, img2) = (img0 + 1, img0 + 2);
                    
                    s.spawn(move |_| {
                        println!("Triplets for Octave[{o}][w[{w}] h[{h}]]: {img0} {img1} {img2}");
                        let im0 = &dogs.get(img0).unwrap();
                        let im1 = &dogs.get(img1).unwrap();
                        let im2 = &dogs.get(img2).unwrap();

                        im1.par_iter()
                            .skip(w - 1)
                            .take(w * (h - 2) - 1)
                            .enumerate()
                            .for_each(|(i, p)| {
                                let fl = i % w;
                                {
                                    if fl == 0 || fl == 1 {
                                        return;
                                    }
                                }
                                
                                // At this stage, the current `p` is the center pixel because we are iterating over the middle image.
                                if p.abs() <= threshold {
                                    return;
                                }

                                // Here, we are going to find out the true indexes of all other pixels to be evaluated
                                let cent = w + i;
                                let top = cent - w;
                                let bot = cent + w;
                                let left = cent - 1;
                                let right = cent + 1;
                                let topleft = top - 1;
                                let topright = top + 1;
                                let botleft = bot - 1;
                                let botright = bot + 1;
                                
                                // let p = *p;

                                // now we have to evaluate agaist 27 points from all three images.
                                // This would essentially represent a 3 x 3 x 3 array
                                let px_cube: [f32; 27] = [
                                    im0[topleft], im0[top], im0[topright],
                                    im0[left], im0[cent], im0[right],
                                    im0[botleft], im0[bot], im0[botright],
                                    im1[topleft], im1[top], im1[topright],
                                    im1[left], im1[cent], im1[right],
                                    im1[botleft], im1[bot], im1[botright],
                                    im2[topleft], im2[top], im2[topright],
                                    im2[left], im2[cent], im2[right],
                                    im2[botleft], im2[bot], im2[botright]
                                ];
                                
                                if !Self::check_local_maxima_minima(px_cube) {
                                    return;
                                }

                                let local_extremum = Self::sift_localize_extremum_via_quadratic(&px_cube);
                                // if !(
                                //     // greater than all equals all
                                //     (
                                //         p > 0. &&
                                //         p >= im0[cent] && p >= im2[cent] && 
                                //         p >= im0[top] && p >= im1[top] && p >= im2[top] &&
                                //         p >= im0[bot] && p >= im1[bot] && p >= im2[bot] &&
                                //         p >= im0[left] && p >= im1[left] && p >= im2[left] &&
                                //         p >= im0[right] && p >= im1[right] && p >= im2[right] &&
                                //         p >= im0[topleft] && p >= im1[topleft] && p >= im2[topleft] &&
                                //         p >= im0[topright] && p >= im1[topright] && p >= im2[topright] &&
                                //         p >= im0[botleft] && p >= im1[botleft] && p >= im2[botleft] &&
                                //         p >= im0[botright] && p >= im1[botright] && p >= im2[botright]
                                //     ) ||
                                //     // less than equals all
                                //     (
                                //         p < 0. &&
                                //         p <= im0[cent] && p <= im2[cent] && 
                                //         p <= im0[top] && p <= im1[top] && p <= im2[top] &&
                                //         p <= im0[bot] && p <= im1[bot] && p <= im2[bot] &&
                                //         p <= im0[left] && p <= im1[left] && p <= im2[left] &&
                                //         p <= im0[right] && p <= im1[right] && p <= im2[right] &&
                                //         p <= im0[topleft] && p <= im1[topleft] && p <= im2[topleft] &&
                                //         p <= im0[topright] && p <= im1[topright] && p <= im2[topright] &&
                                //         p <= im0[botleft] && p <= im1[botleft] && p <= im2[botleft] &&
                                //         p <= im0[botright] && p <= im1[botright] && p <= im2[botright]
                                //     )
                                // ) {
                                //     return;
                                // }

                            })
                    });
                }
            }
        });
        
        Ok(())
    }

    fn check_local_maxima_minima(check: [f32; 27]) -> bool {
        let p = check[13];

        if p > 0. {
            for (i, _p) in check.iter().enumerate() {
                if i == 13 {
                    continue;
                }
                if _p < &p {
                    return false;
                }
            }

            return true;
        } else if p < 0. {
            for (i, _p) in check.iter().enumerate() {
                if i == 13 {
                    continue;
                }
                if _p > &p {
                    return false;
                }
            }

            return true;
        }
        false
    }

    fn sift_localize_extremum_via_quadratic(px_cube: &[f32; 27]) -> Option<()> {
        const ATT_TO_CONVERGE: usize = 5;

        for i in 0 .. ATT_TO_CONVERGE {
            let grad = Self::sift_compute_gradient(px_cube);
            let hess = Self::sift_compute_hessian(px_cube);
            // let grad = Self::sift_compute_gradient(px_cube);
        }

        None
    }

    // Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    // https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-1-306a99b50aa5

    // With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
    // Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
    // NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    fn sift_compute_gradient(px_cube: &[f32; 27]) -> [f32; 3] {
        let dx = 0.5 * (px_cube[14] - px_cube[12]);
        let dy = 0.5 * (px_cube[16] - px_cube[10]);
        let ds = 0.5 * (px_cube[22] - px_cube[4]);
        
        [dx, dy, ds]
    }

    // Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    // https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-1-306a99b50aa5

    // With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    // Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    // With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    // Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    // NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    fn sift_compute_hessian(px_cube: &[f32; 27]) -> [f32; 9] {
        let px_cent = px_cube[13];

        // todo!("Re-visit this when we have some free time!");

        [0.; 9]
    }

}

#[cfg(test)]
mod tests {
    use super::Image;

    use anyhow::Result;

    #[test]
    fn hod() -> Result<()> {
        let im = Image::new("data/1_small.png")?;
        im.hog()?;

        Ok(())
    }

    #[test]
    fn sift() -> Result<()> {
        let mut im = Image::new("data/1_small.png")?;
        im.visualize();

        im.sift().unwrap();

        Ok(())
    }
}