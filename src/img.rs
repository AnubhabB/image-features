use std::{sync::{Arc}};
use anyhow::Result;
use image::{save_buffer_with_format, ColorType, DynamicImage, GenericImageView};
use parking_lot::RwLock;
use rayon::{prelude::*, scope};


pub struct Image {
    w: usize,
    h: usize,
    img: DynamicImage,
    viz: bool
}


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


#[cfg(test)]
mod tests {
    // use std::f32::EPSILON;

    use super::Image;

    use anyhow::{Result, Ok};

    #[test]
    fn hod() -> Result<()> {
        let im = Image::new("data/1_small.png")?;
        im.hog()?;

        Ok(())
    }

    // #[test]
    // fn sift() -> Result<()> {
    //     let mut im = Image::new("data/1_small.png")?;
    //     // im.visualize();

    //     im.sift().unwrap();

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