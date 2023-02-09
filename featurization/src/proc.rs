use image::{ImageBuffer, Luma, Pixel, GrayImage};
use ndarray::{Array, Array2, ArrayView2};
use std::f32::consts::PI;

pub type MatrixAsImage = ImageBuffer<Luma<f32>, Vec<<Luma<f32> as Pixel>::Subpixel>>;

pub fn to_ndarray(image: MatrixAsImage) -> Array2<f32> {
    let (width, height) = image.dimensions(); 
    let buffer = image.into_raw(); 
    Array::from_shape_vec((height as usize, width as usize), buffer).unwrap()
}

pub fn to_image(array: ArrayView2<f32>) -> MatrixAsImage {
    let shape = array.shape(); 
    let buffer = array.as_slice().unwrap().to_vec(); 
    ImageBuffer::from_raw(shape[1] as u32, shape[0] as u32, buffer).unwrap()
}

pub fn down_sample(image: &MatrixAsImage) -> MatrixAsImage {
    ImageBuffer::from_fn(
        image.width() / 2, 
        image.height() / 2, 
        |x, y| { *image.get_pixel(x * 2, y * 2) }
    )
}

pub fn normalize(image: &MatrixAsImage) -> MatrixAsImage {
    let mut max_pixel = f32::MIN;

    for &Luma([pixel]) in image.pixels() {
        max_pixel = max_pixel.max(pixel);
    }

    ImageBuffer::from_fn(image.width(), image.height(), |x, y| {
        Luma([image.get_pixel(x, y)[0] / max_pixel])
    })
}

pub fn gaussian_blur(image: &MatrixAsImage, sigma: f32) -> MatrixAsImage {
    let n = 2 * sigma.ceil() as u32;
    let kernel = gaussian_kernel(n, sigma);
    normalize(&convolve(image, &kernel))
}

pub fn gaussian_kernel(n: u32, sigma: f32) -> MatrixAsImage {
    let width = 2 * n + 1;

    ImageBuffer::from_fn(width, width, |x, y| {
        let x = (x as f32) - n as f32;
        let y = (y as f32) - n as f32;
        let sq_sigma = sigma * sigma;

        let prefactor = 0.5 / PI / sq_sigma;
        let distance = x * x + y * y;
        let value = prefactor * (-0.5 * distance / sq_sigma).exp();

        Luma([value])
    })
}

pub fn convolve(image: &MatrixAsImage, kernel: &MatrixAsImage) -> MatrixAsImage {
    let kernel_width = kernel.width();
    let kernel_height = kernel.height();

    let nx = ((kernel_width - 1) / 2) as i32;
    let ny = ((kernel_height - 1) / 2) as i32;

    let result = ImageBuffer::from_fn(image.width(), image.height(), |x, y| {
        let mut value = 0.0f32;

        for kx in 0..kernel_width {
            for ky in 0..kernel_height {
                let ox = kx as i32 - nx;
                let oy = ky as i32 - ny;
                
                let vx = (x as i32 - ox).abs() as u32; 
                let vy = (y as i32 - oy).abs() as u32; 
                
                let vx = if vx > image.width() - 1 { 2 * image.width() - vx - 2 } else { vx }; 
                let vy = if vy > image.height() - 1 { 2 * image.height() - vy - 2 } else { vy }; 
                
                value += 
                    kernel.get_pixel(kx, ky).0[0] * 
                    image.get_pixel(vx, vy).0[0];
            }
        }

        Luma([value])
    });

    result
}

pub fn embed(image: &GrayImage) -> MatrixAsImage {
    ImageBuffer::from_fn(image.width(), image.height(), |x, y| {
        Luma([image.get_pixel(x, y)[0] as f32 / 255.0])
    })
}

pub fn discretize(image: &MatrixAsImage) -> GrayImage { 
    let image: MatrixAsImage = normalize(image); 
    ImageBuffer::from_fn(image.width(), image.height(), |x, y| {
        Luma([(image.get_pixel(x, y)[0] * 255.0).round() as u8])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const THRESHOLD: f32 = 0.0001; 

    #[test]
    fn test_blur() {
        let mut image: MatrixAsImage = ImageBuffer::new(9, 9);
        image.put_pixel(4, 4, Luma([1.0f32]));

        let result: MatrixAsImage = gaussian_blur(&image, 1.0_f32);
        let expected: MatrixAsImage = MatrixAsImage::from_raw(9, 9, vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0029150245, 0.013064233, 0.021539278, 0.013064233, 0.0029150245, 0.0, 0.0, 
            0.0, 0.0, 0.013064233, 0.05854983, 0.09653235, 0.05854983, 0.013064233, 0.0, 0.0, 
            0.0, 0.0, 0.021539278, 0.09653235, 0.15915494, 0.09653235, 0.021539278, 0.0, 0.0, 
            0.0, 0.0, 0.013064233, 0.05854983, 0.09653235, 0.05854983, 0.013064233, 0.0, 0.0, 
            0.0, 0.0, 0.0029150245, 0.013064233, 0.021539278, 0.013064233, 0.0029150245, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]).unwrap(); 

        for x in 2..7 {
            for y in 2..7 {
                let r = result.get_pixel(x, y)[0]; 
                let e = expected.get_pixel(x, y)[0]; 
                assert!((r - e).abs() < THRESHOLD); 
            }
        }
    }

    #[test]
    fn test_down_sampling() {
        let mut image: MatrixAsImage = ImageBuffer::new(20, 20); 
        
        for x in 0..10 {
            for y in 0..10 {
                image.put_pixel(x * 2, y * 2, Luma([1.0]))
            }
        }
    }
}
