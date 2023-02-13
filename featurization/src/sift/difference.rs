use crate::sift::scale::ScaleSpace;
use ndarray::{s, Array, Ix3};

pub struct DoGSpace {
    pub num_octaves: usize,
    pub num_dog_per_octaves: usize,
    pub spaces: Vec<Array<f32, Ix3>>,
}

pub fn compute_difference_of_gaussian(gss: &ScaleSpace) -> DoGSpace {
    let num_octaves = gss.num_octaves as usize;
    let num_dog_per_octaves = gss.num_scales_per_octave as usize;
    let mut spaces: Vec<Array<f32, Ix3>> = Vec::with_capacity(num_octaves);

    for octave in 0..num_octaves {
        let prev = gss.spaces[octave].slice(s![..-1, .., ..]);
        let next = gss.spaces[octave].slice(s![1.., .., ..]);
        spaces.push(&next - &prev);
    }

    DoGSpace {
        num_octaves,
        num_dog_per_octaves,
        spaces,
    }
}
