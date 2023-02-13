use image::io::Reader as ImageReader;
use ndarray::Array2;
use proc::{embed, to_ndarray};
use std::env;
use sift::{
    difference::compute_difference_of_gaussian,
    extrema::compute_extrema,
    refiner::{refine_keypoints_on_edge, refine_keypoints_with_low_contrast},
    scale::compute_scale_space,
};

mod proc;
mod sift;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        println!("Usage: featurize <input>");
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
    let scales = compute_scale_space(matrix);
    let differences = compute_difference_of_gaussian(&scales);

    let keypoints = compute_extrema(&differences);
    println!("⭐\tOriginally extracted keypoints: {}", keypoints.len());

    let keypoints = refine_keypoints_with_low_contrast(&scales, &differences, keypoints);
    println!(
        "\t⌞ After discarded low-contrast keypoints: {}",
        keypoints.len()
    );

    let keypoints = refine_keypoints_on_edge(&differences, keypoints);
    println!("\t⌞ After discarded keypoints on edge: {}", keypoints.len());
}
