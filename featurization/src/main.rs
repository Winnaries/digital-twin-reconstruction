use image::io::Reader as ImageReader;
use ndarray::{Array, Array2};
use ndarray_npy::write_npy;
use proc::{embed, to_ndarray};
use sift::{
    difference::compute_difference_of_gaussian,
    extrema::compute_extrema,
    keypoint::AbsoluteKeypoint,
    refiner::{refine_keypoints_on_edge, refine_keypoints_with_low_contrast},
    scale::compute_scale_space,
};
use std::env;

use crate::sift::{descriptor::refine_with_reference_orientation, keypoint::OrientedKeypoint};

mod proc;
mod sift;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        println!("Usage: featurize <input> <npz_output>");
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
        "｜\t⌞ After discarded low-contrast keypoints: {}",
        keypoints.len()
    );

    let keypoints = refine_keypoints_on_edge(&differences, keypoints);

    println!(
        "｜\t⌞ After discarded keypoints on edge: {}",
        keypoints.len()
    );

    let keypoints = refine_with_reference_orientation(&scales, keypoints, 10);

    println!(
        "｜\t⌞ After discarded keypoints with incomplete patch: {}",
        keypoints.len()
    );

    let keypoints: Vec<f32> = keypoints
        .into_iter()
        .flat_map(|p| {
            let AbsoluteKeypoint { x, y, sigma, .. } = *p;
            let OrientedKeypoint { theta, .. } = p;

            [x, y, sigma, theta]
        })
        .collect();

    let keypoints = Array::from_shape_vec([keypoints.len() / 4, 4], keypoints).unwrap();

    write_npy(&args[2], &keypoints).unwrap();
}
