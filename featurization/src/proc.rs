#![allow(dead_code)]

use image::{ImageBuffer, Luma, Pixel, GrayImage};
use ndarray::{Array, Array2, ArrayView2};

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

pub fn normalize(image: &MatrixAsImage) -> MatrixAsImage {
    let mut max_pixel = f32::MIN;

    for &Luma([pixel]) in image.pixels() {
        max_pixel = max_pixel.max(pixel);
    }

    ImageBuffer::from_fn(image.width(), image.height(), |x, y| {
        Luma([image.get_pixel(x, y)[0] / max_pixel])
    })
}