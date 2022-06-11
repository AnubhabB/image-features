// use anyhow::Result;
// use img::Image;

pub mod img;

// fn main() -> Result<()> {
//     let img = Image::new("data/1.png")?;
//     let h = img.prewitt_h()?;
//     img.draw_grey("data/1_prewitt_h.png", &h[..])?;
//     let h = img.prewitt_v()?;
//     img.draw_grey("data/1_prewitt_v.png", &h[..])?;
//     let h = img.prewitt()?;
//     img.draw_grey("data/1_prewitt.png", &h[..])?;
//     img.hog()?;

//     Ok(())
// }
