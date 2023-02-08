use image::{io::Reader as ImageReader};
use std::env;
use proc::{gaussian_blur, embed, discretize, down_sample}; 

mod proc; 
mod octave; 

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
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

    let matrix = embed(&img);
    let blurred = gaussian_blur(&matrix, 3.0f32);
    discretize(&blurred).save(&args[2]).unwrap(); 
    discretize(&down_sample(&blurred)).save(&args[3]).unwrap(); 
}
