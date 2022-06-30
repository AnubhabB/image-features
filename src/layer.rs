use anyhow::Result;
use ndarray::{Array4};

pub trait Layer {
    fn forward(&self, x: Array4<f32>) -> Result<()>;

    fn set_name(&mut self, name: &'static str);
}
