use image::io::Reader as ImageReader;
use ndarray::Array2;
use sift::util::gaussian_blur; 
use std::env;
use proc::{embed, discretize, to_ndarray, to_image}; 

mod sift; 
mod proc; 

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        println!("Usage: featurize <input> <output>");
        return;
    }

    let img = ImageReader::open(&args[1])
        .unwrap()
        .decode()
        .unwrap()
        .grayscale()
        .as_luma8()
        .unwrap()
        .clone(); 

    let matrix: Array2<f32> = to_ndarray(embed(&img));
    let blurred: Array2<f32> = gaussian_blur(matrix.view(), 3.0f32);
    discretize(&to_image(blurred.view())).save(&args[2]).unwrap(); 
}
