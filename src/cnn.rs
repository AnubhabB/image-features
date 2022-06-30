use crate::{layer::Layer, activation::{Activation}, utils::Batch};

use anyhow::{Result, anyhow};
use ndarray::{Array, Array1, Dim, Array4, Array2, s, Slice, Array3, Axis};
use ndarray_rand::{rand_distr::{Normal, Distribution}, rand::thread_rng};


// todo: implement padding!
fn conv_3d(v: Array4<f32>, kernel: Array4<f32>, stride: usize, _pad: usize) {
    let view = as_stride(v, &kernel.shape()[0 .. 2], stride);
}

fn as_stride(v: Array4<f32>, sub_shape: &[usize], stride: usize) {
    let (sm, sh, sw, sc) = if let [sm, sh, sw, sc] = v.strides() { (*sm, *sh, *sw, *sc) } else { todo!() };
    let (m, h, w, c) = {
        let shape = v.shape();
        (shape[0], shape[1], shape[2], shape[3])
    };
    let (f1, f2) = (sub_shape[0], sub_shape[1]);

    // view_shape = (m, 1+(hi-f1)//stride, 1+(wi-f2)//stride, f1, f2, ci)
    // strides = (sm, stride*sh, stride*sw, sh, sw, sc)
    let view_shape = (m, 1 + (h - f1)/ stride, 1 + (w - f2)/ stride, f1, f2, c);
    let strides = (sm, stride * sh as usize, stride * sw as usize, sc);

    
}

struct Cnn {
    layers: Vec<Box<dyn Layer>>,
}


impl Cnn {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    // Add new layers to the network
    pub fn add(&mut self, layer: Box<dyn Layer>) -> &mut Self {
        self.layers.push(layer);

        self
    }


    pub fn train(&self, x: Array4<f32>, y: Array2<f32>, epochs: usize, batch_size: usize) -> Result<Vec<f32>> {
        for e in 0 .. epochs {
            for b in Batch::new(x.dim().0, batch_size) {
                // let p = x.slice(s![0, ..,.., ..]);
                let mut xii: Array4<f32> = Array4::default((0, x.dim().1, x.dim().2, x.dim().3));
                let mut yii: Array2<f32> = Array2::default((0, y.dim().1));

                for p in b.iter() {
                    xii.push(Axis(0), x.slice(s![*p, .., .., ..]))?;
                    yii.push(Axis(0), y.slice(s![*p, ..]))?;
                }

                self.forward(xii);                
            }
        }
        Ok(vec![])
    }

    fn forward(&self, x: Array4<f32>) -> Result<()> {
        let mut a = x;
        for (i, l) in self.layers.iter().enumerate() {
            l.forward(a.clone())?;
        }

        Ok(())
    }

}


pub struct Conv {
    // kernel size for height and width.
    f: u8,
    // learning rate
    lr: f32,
    std: f32,
    // padding
    pad: u8,
    // channels of current layer
    chan: u8,
    // a name for this layer
    name: &'static str,
    // strides for convolution
    stride: u8,
    // regularization param
    lambda: f32,
    // channels of input layer
    chan_in: u8,
    // activation function, defaults to relu
    activation: Activation,

    filt: Array<f32, Dim<[usize; 4]>>,
    bias: Array1<f32>,
}

impl Layer for Conv {
    fn forward(&self, x: Array4<f32>) -> Result<()> {

        Ok(())
    }

    fn set_name(&mut self, name: &'static str) {
        self.name = name;
    }
}

impl Conv {
    pub fn new(f: u8, pad: u8, stride: u8, chan_in: u8, chan: u8, lr: f32, activation: Option<Activation>) -> Result<Self> {
        if f % 2 == 0 {
            return Err(anyhow!("even numbered filer size"));
        }

        if stride == 0 {
            return Err(anyhow!("Stride of 0 passed!"));
        }

        let activation = if let Some(acticvation) = activation {
            acticvation
        } else {
            Activation::Relu
        };

        let lambda = 0.01;
        let rng = thread_rng();

        let std = ((2_f32/f.pow(2) as f32)/ chan_in as f32).sqrt();
        let filt: Vec<f32> = Normal::<f32>::new(0., std)?.sample_iter(rng.clone()).take((f.pow(2) * chan_in * chan) as usize).collect();
        let bias: Vec<f32> = Normal::<f32>::new(0., std)?.sample_iter(rng).take(chan as usize).collect();

        Ok(Self {
            f,
            lr,
            std,
            pad,
            chan,
            name: "conv",
            stride,
            lambda,
            chan_in,
            activation,
            filt: Array::from_shape_vec((f as usize, f as usize, chan_in as usize, chan as usize), filt)?,
            bias: Array::from_shape_vec(chan as usize, bias)?
        })
    }
}

pub enum PoolMethod {
    Max,
    Mean
}

pub struct Pool {
    // kernel size for height and width.
    // because this is non-overlapping pooling we'll use this for stride as well
    f: u8,
    // padding
    pad: u8,
    name: &'static str,
    // `max` or `mean` pooling
    method: PoolMethod
}

impl Layer for Pool {
    fn forward(&self, x: Array4<f32>) -> Result<()> {
        Ok(())
    }

    fn set_name(&mut self, name: &'static str){
        self.name = name;
    }
}

impl Pool {
    pub fn new(f: u8, pad: u8, method: PoolMethod) -> Self {
        Self {
            f,
            pad,
            name: "pool",
            method
        }
    }
}

pub struct Flatten {
    name: &'static str,
}

impl Layer for Flatten {
    fn forward(&self, x: Array4<f32>) -> Result<()> {
        Ok(())
    }

    fn set_name(&mut self, name: &'static str) {
        self.name = name
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self {
            name: "flatten"
        }   
    }
}

impl Flatten {
    pub fn new() -> Self {
        Self::default()
    }
}

pub struct Dense {
    lr: f32,
    name: &'static str,
    bias: Array1<f32>,
    // clip gradients between [-clip, clip]
    clip: f32,
    input: u32,
    output: u32,
    // regularization param
    lambda: f32,
    weight: Array<f32, Dim<[usize; 2]>>,
    activation: Activation
}

impl Layer for Dense {
    fn forward(&self, x: Array4<f32>) -> Result<()> {
        Ok(())
    }

    fn set_name(&mut self, name: &'static str) {
        self.name = name;
    }
}

impl Dense {
    pub fn new(input: u32, output: u32, lr: f32, activation: Option<Activation>) -> Result<Self> {
        let activation = if let Some(act) = activation {
            act
        } else {
            Activation::Relu
        };

        let lambda = 0.01;
        let clip = 0.5;

        let std = (2_f32/input as f32).sqrt();
        let rng = thread_rng();

        // self.weights = np.random.normal(0, std, size=[self.n_outputs, self.n_inputs])
        // self.biases = np.random.normal(0, std, size=self.n_outputs)
        let wght: Vec<f32> = Normal::<f32>::new(0., std)?.sample_iter(rng.clone()).take((input * output) as usize).collect();
        let bias: Vec<f32> = Normal::<f32>::new(0., std)?.sample_iter(rng).take(output as usize).collect();

        Ok(Self {
            lr,
            bias: Array1::from(bias),
            name: "dense",
            clip,
            input,
            output,
            lambda,
            weight: Array::from_shape_vec((input as usize, output as usize), wght)?,
            activation
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    
    use crate::{activation::Activation, layer::Layer};

    use super::{Cnn, Conv, Pool, PoolMethod, Flatten, Dense};

    #[test]
    fn cnn() -> Result<()> {
        let mut conv_1 = Conv::new(3, 0, 1, 1, 10, 0.1, Some(Activation::Relu))?;
        conv_1.set_name("conv_1");

        let mut pool_1 = Pool::new(2, 0, PoolMethod::Max);
        pool_1.set_name("pool_1");

        let mut conv_2 = Conv::new(5, 0, 1, 10, 16, 0.1, Some(Activation::Relu))?;
        conv_2.set_name("conv_2");

        let mut flat_1 = Flatten::new();
        flat_1.set_name("flat_1");

        let mut dense_1 = Dense::new(1296, 100, 0.1, Some(Activation::Relu))?;
        dense_1.set_name("dense_1");

        let mut dense_2 = Dense::new(100, 10, 0.01, Some(Activation::Softmax))?;
        dense_2.set_name("dense_2");

        let cnn = Cnn::new()
            .add(Box::new(conv_1))
            .add(Box::new(pool_1))
            .add(Box::new(conv_2))
            .add(Box::new(flat_1))
            .add(Box::new(dense_1))
            .add(Box::new(dense_2));

        Ok(())
    }
}