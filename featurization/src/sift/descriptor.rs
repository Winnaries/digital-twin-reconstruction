use std::f32::consts::PI;

use super::{
    keypoint::{AbsoluteKeypoint, OrientedKeypoint},
    scale::ScaleSpace,
};
use ndarray::{s, Array, ArrayView, Ix1, Ix2, Ix3};

#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

const LAMBDA: f32 = 1.5;
const REF: f32 = 0.8;

fn compute_distance_to_border(x: f32, y: f32, width: f32, height: f32) -> f32 {
    x.min(y).min(width - x).min(height - y)
}

fn compute_spatial_gradient(space: ArrayView<f32, Ix2>) -> Array<f32, Ix3> {
    let shape = space.raw_dim();
    let (m, n) = (shape[0], shape[1]);

    let dy: Array<f32, Ix2> =
        0.5 * (&space.slice(s![2 as isize.., ..]) - &space.slice(s![..-2 as isize, ..]));

    let dx: Array<f32, Ix2> =
        0.5 * (&space.slice(s![.., 2 as isize..]) - &space.slice(s![.., ..-2 as isize]));

    let mut gradient = Array::zeros([m, n, 2]);
    gradient
        .slice_mut(s![1..-1 as isize, .., 0])
        .assign(&dy);
    gradient
        .slice_mut(s![.., 1..-1 as isize, 1])
        .assign(&dx);

    gradient
}

pub fn refine_with_reference_orientation(
    space: &ScaleSpace,
    keypoint: Vec<AbsoluteKeypoint>,
    num_bins: usize,
) -> Vec<OrientedKeypoint> {
    let fx = |&p| {
        let theta = compute_reference_orientation(space, p, num_bins);
        theta.map(|x| OrientedKeypoint { inner: p, theta: x })
    };

    #[cfg(not(feature = "parallel"))]
    let result: Vec<OrientedKeypoint> = keypoint.iter().filter_map(fx).collect();

    #[cfg(feature = "parallel")]
    let result: Vec<OrientedKeypoint> = keypoint.par_iter().filter_map(fx).collect();

    result
}

pub fn compute_reference_orientation(
    space: &ScaleSpace,
    keypoint: AbsoluteKeypoint,
    num_bins: usize,
) -> Option<f32> {
    let clamp = |x: isize| {
        if x == -1 { num_bins as isize - 1 } 
        else if x == num_bins as isize { 0 }
        else { x } 
    }; 

    // Retrieving the scale space that has its sigma
    // nearest to the keypoint's sigma.
    let AbsoluteKeypoint { x, y, sigma, .. } = keypoint;
    let nearest: (f32, ArrayView<f32, Ix2>) = space.nearest_sigma_space(sigma);
    let delta = nearest.0;
    let space: ArrayView<f32, Ix2> = nearest.1;

    // Preparing needed arrays for the following
    // computation: 1D histogram and spatial gradient.
    let mut histogram: Array<f32, Ix1> = Array::zeros([num_bins]);
    let gradient = compute_spatial_gradient(space.view());

    // Compute distance from the nearest border using
    // absolute coordinate for generalization.
    let dist_border = compute_distance_to_border(
        x as f32,
        y as f32,
        delta * space.shape()[1] as f32,
        delta * space.shape()[0] as f32,
    );

    
    // Check if the patch is entirely contained
    // in the image, otherwise return None.
    let n = 3.0 * LAMBDA * sigma;
    if dist_border <= 3.0 * LAMBDA * sigma {
        return None;
    }
    
    // Defining the starting point and end point of the
    // patch to be analyzed in this function.
    let sy = ((y - n) / delta)
        .round()
        .clamp(1.0, space.shape()[0] as f32 - 2.0) as isize;
    let ey = ((y + n) / delta)
        .round()
        .clamp(1.0, space.shape()[0] as f32 - 2.0) as isize;
    let sx = ((x - n) / delta)
        .round()
        .clamp(1.0, space.shape()[1] as f32 - 2.0) as isize;
    let ex = ((x + n) / delta)
        .round()
        .clamp(1.0, space.shape()[1] as f32 - 2.0) as isize;

    // Constructing the histogram by calculating the weight
    // of each pixel found within the defined patch.
    // The weight is just a product of gaussian and
    // gradient norm (the magnitude). In the future, 
    // we should pre-compute the theta value to avoid 
    // redundant re-calculation in different iteration. 
    for m in sy..ey {
        for n in sx..ex {
            let dm = delta * m as f32 - y;
            let dn = delta * n as f32 - x;
            let dist = dm * dm + dn * dn;
            let denom = 2.0 * (LAMBDA * sigma).powi(2);
            let exp = (-dist * denom.recip()).exp();

            let grad = gradient.slice(s![m, n, ..]);
            let norm = (&grad * &grad).sum().sqrt();

            let weight = exp * norm;

            // Assign the target bin to each pixel
            // based on their x and y gradient.
            let theta = grad[0].atan2(grad[1]);
            let theta = if theta < 0.0 { 2.0 * PI + theta } else { theta }; 

            let bin = num_bins as f32 / 2.0 / PI * theta;
            let bin = bin.floor() as usize;
            
            histogram[bin] += weight;
        }
    }

    // Apply six-times circular convolution
    // with filter [1, 1, 1]/3. In the future, might have to
    // replace with mirroed pad.
    for _ in 0..6 {
        histogram = Array::from_shape_fn(histogram.raw_dim(), |k| {
            let a = clamp(k as isize - 1) as usize; 
            let b = clamp(k as isize) as usize; 
            let c = clamp(k as isize + 1) as usize; 
            (histogram[[a]] + histogram[[b]] + histogram[[c]]) / 3.0
        });
    }


    // Finding the maximum within the histogram for reference.
    let maximum = REF * histogram.iter().map(|&x| x).reduce(f32::max).unwrap();
    let mut h = 0;

    // Going through the histogram value to find local maximum
    // and test wheter they are more than 0.8 * maximum.
    for (k, &curr) in histogram.iter().enumerate() {
        let prev = histogram[clamp(k as isize - 1) as usize];
        let next = histogram[clamp(k as isize + 1) as usize];
        if curr > prev && curr > next && curr > maximum {
            h = k;
            break;
        }
    }

    // Because the histogram is discretized into bins,
    // we have to interpolate to find the actual continuous
    // dominant orientation within the patch.
    let hk = histogram[[h]];
    let hm = clamp(h as isize - 1) as usize;
    let hp = clamp(h as isize + 1) as usize;
    let hm = histogram[[hm]];
    let hp = histogram[[hp]];

    // The actual interpolation goes here. In SIFT,
    // many mathematical technique is applied such as
    // finite differencing and interpolation. It might be
    // a good idea to study them first.
    let interp = (hm - hp) / (hm - 2.0 * hk + hp);
    let theta_k = 2.0 * PI * h as f32 / num_bins as f32;
    let theta_ref = theta_k + PI / num_bins as f32 * interp;

    Some(theta_ref)
}
