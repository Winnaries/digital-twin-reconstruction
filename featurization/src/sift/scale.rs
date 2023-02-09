#![allow(dead_code)]

use ndarray::{Array, Ix2, Ix3};

use super::util::bilinear_upsample;

pub struct ScaleSpace {
    /// Total of number of octaves. 
    pub num_octaves: u32, 

    /// Number of scales per octave. 
    pub num_scales_per_octave: u32,

    /// Target sampling distance
    /// of the seed image, the image
    /// is upsampled when this value < 1.
    pub delta_seed: f32, 

    /// Target blur of the seed image, 
    /// this value is computed
    pub sigma_seed: f32, 

    /// Assumed blur of the input
    /// image or matrix.
    pub sigma_input: f32,

    /// An 4-dimensional array
    /// with each axis representing
    /// octave, sigma, y, x.
    pub spaces: Vec<Array<f32, Ix3>>, 
}

impl ScaleSpace {

    fn default(image: Array<f32, Ix2>) -> Self {
        let num_octaves: u32 = 5; 
        let num_scales_per_octave: u32 = 6; 
        let delta_seed = 0.5; 
        let sigma_seed = 0.8; 
        let sigma_input = 0.5; 

        let mut spaces: Vec<Array<f32, Ix3>> = Vec::with_capacity(num_octaves as usize); 
        let image_seed = bilinear_upsample(image.view()); 
        
        ScaleSpace {
            num_octaves, 
            num_scales_per_octave, 
            delta_seed, 
            sigma_seed, 
            sigma_input, 
            spaces, 
        }
    }

    fn dsyx(&self, index: impl Into<[usize; 4]>) -> [f32; 4] {
        let [o, i, j, k]: [usize; 4] = index.into(); 

        [
            self.compute_octave_delta_seed(o), 
            self.compute_specific_sigma(o, i), 
            j as f32, 
            k as f32, 
        ]
    }

    fn ds(&self, index: impl Into<[usize; 2]>) -> [f32; 2] {
        let [o, i]: [usize; 2] = index.into(); 
        
        [
            self.compute_octave_delta_seed(o), 
            self.compute_specific_sigma(o, i)
        ]
    }

    fn compute_octave_delta_seed(&self, octave: usize) -> f32 {
        self.delta_seed * 2.0f32.powi(octave as i32)
    }

    fn compute_octave_sigma_seed(&self, octave: usize) -> f32 {
        self.sigma_seed * 2.0f32.powi(octave as i32)
    }

    fn compute_specific_sigma(&self, octave: usize, index: usize) -> f32 {
        let denom = self.num_scales_per_octave as f32 - 3f32; 
        let octave_sigma_seed = self.compute_octave_sigma_seed(octave); 
        let k = 2.0f32.powf(denom.recip());
        k.powi(index as i32) * octave_sigma_seed
    }

    fn compute_rho(&self, s: usize) -> f32 {
        let m = self.sigma_seed / self.delta_seed; 
        let next = 2.0f32.powf(2.0 * (s + 1) as f32 / self.num_scales_per_octave as f32); 
        let prev = 2.0f32.powf(2.0 * s as f32 / self.num_scales_per_octave as f32);
        m * (next - prev).sqrt()
    }

    fn compute_initial_rho(&self) -> f32 {
        (self.sigma_seed.powi(2) - self.sigma_input.powi(2)).sqrt() / self.delta_seed
    }

}

#[cfg(test)]
mod test {
    use std::fmt::Debug;

    use num::Signed;
    use super::*; 

    fn all_close<T, I>(a: T, b: T, tol: I) -> bool
    where 
        T: Copy + IntoIterator<Item = I> + Debug, 
        I: Signed + PartialOrd, 
    {
        let ai = a.into_iter(); 
        let bi = b.into_iter(); 
        
        if ai.zip(bi).map(|(x, y)| { (x - y).abs() }).all(|x| { x < tol }) {
            true
        } else {
            println!("left: {:?}", a); 
            println!("right: {:?}", b);
            false
        }
    }

    #[test]
    pub fn test_index_to_parameters_conversion() {
        let space = ScaleSpace {
            num_octaves: 5, 
            num_scales_per_octave: 6, 
            delta_seed: 0.5, 
            sigma_seed: 0.8, 
            sigma_input: 0.5, 
            spaces: vec![], 
        }; 

        assert!(all_close(space.ds([0, 0]), [0.5,  0.80], 0.1));
        assert!(all_close(space.ds([0, 5]), [0.5,  2.54], 0.1));
        assert!(all_close(space.ds([2, 3]), [2.0,  6.40], 0.1));
        assert!(all_close(space.ds([4, 2]), [8.0, 20.32], 0.1));
    }

}