use ndarray::{s, Ix3, Array};
use crate::sift::scale::ScaleSpace; 

pub struct DoGSpace {
    pub num_octaves: usize, 
    pub num_dog_per_octaves: usize, 
    pub spaces: Vec<Array<f32, Ix3>>, 
}

#[allow(dead_code)]
pub fn compute_difference_of_gaussian(gss: ScaleSpace) -> DoGSpace {
    let num_octaves = gss.num_octaves as usize; 
    let num_dog_per_octaves = gss.num_scales_per_octave as usize - 1; 
    let mut spaces: Vec<Array<f32, Ix3>> = Vec::with_capacity(num_octaves); 

    for octave in 0..num_octaves {
        let prev = gss.spaces[octave].slice(s![..-2, .., ..]); 
        let next = gss.spaces[octave].slice(s![1.., .., ..]); 
        spaces.push(&next - &prev); 
    }

    DoGSpace {
        num_octaves, 
        num_dog_per_octaves, 
        spaces, 
    }
}

