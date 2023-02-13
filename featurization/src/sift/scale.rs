use ndarray::{s, Array, Ix2, Ix3};
use super::util::{bilinear_upsample, downsample, gaussian_blur};

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

    /// A vector of 3-dimensional array
    /// with each axis representing
    /// octave, sigma, y, x. Cannot stack 
    /// inner arrays because of different size. 
    pub spaces: Vec<Array<f32, Ix3>>,
}

#[allow(dead_code)]
pub fn compute_scale_space(image: Array<f32, Ix2>) -> ScaleSpace {
    let mut gss = ScaleSpace {
        num_octaves: 5,
        num_scales_per_octave: 3,
        delta_seed: 0.5,
        sigma_seed: 0.8,
        sigma_input: 0.5,
        spaces: Vec::with_capacity(5),
    };

    println!(); 
    println!("🪴\tGenerating seed image...");
    let mut sigma = gss.compute_initial_rho();
    let seed: Array<f32, Ix2> = bilinear_upsample(image.view());
    let mut seed: Array<f32, Ix2> = gaussian_blur(seed.view(), sigma);
    let per_octave = gss.num_scales_per_octave + 3;

    for o in 0..gss.num_octaves {
        println!("\t⌞ Generating octave #{}...", o);
        
        let dim = [per_octave as usize, seed.dim().0, seed.dim().1];
        let mut ss: Array<f32, Ix3> = Array::zeros(dim);
        let mut next_image: Array<f32, Ix2>;
        ss.slice_mut(s![0, .., ..]).assign(&seed);

        for idx in 1..per_octave {
            sigma = gss.compute_rho((idx - 1) as usize);
            next_image = gaussian_blur(ss.slice(s![(idx - 1) as usize, .., ..]), sigma);
            ss.slice_mut(s![idx as usize, .., ..]).assign(&next_image);
        }

        seed = downsample(ss.slice(s![(per_octave - 2) as usize, .., ..]));
        gss.spaces.push(ss);
    }

    gss
}

#[allow(dead_code)]
impl ScaleSpace {
    pub fn dsyx(&self, o: usize, s: f32, m: f32, n: f32) -> [f32; 4] {
        let delta = self.compute_octave_delta_seed(o); 
        let sigma = delta / self.delta_seed * self.sigma_seed * 2.0f32.powf(s / self.num_scales_per_octave as f32);

        [
            delta, 
            sigma, 
            m as f32 * delta,
            n as f32 * delta,
        ]
    }

    pub fn ds(&self, index: impl Into<[usize; 2]>) -> [f32; 2] {
        let [o, i]: [usize; 2] = index.into();

        [
            self.compute_octave_delta_seed(o),
            self.compute_specific_sigma(o, i),
        ]
    }

    fn compute_octave_delta_seed(&self, octave: usize) -> f32 {
        self.delta_seed * 2.0f32.powi(octave as i32)
    }

    fn compute_octave_sigma_seed(&self, octave: usize) -> f32 {
        self.sigma_seed * 2.0f32.powi(octave as i32)
    }

    fn compute_specific_sigma(&self, octave: usize, index: usize) -> f32 {
        let denom = self.num_scales_per_octave as f32;
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

    use super::*;
    use num::Signed;

    fn all_close<T, I>(a: T, b: T, tol: I) -> bool
    where
        T: Copy + IntoIterator<Item = I> + Debug,
        I: Signed + PartialOrd,
    {
        let ai = a.into_iter();
        let bi = b.into_iter();

        if ai.zip(bi).map(|(x, y)| (x - y).abs()).all(|x| x < tol) {
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
            num_scales_per_octave: 3,
            delta_seed: 0.5,
            sigma_seed: 0.8,
            sigma_input: 0.5,
            spaces: vec![],
        };

        assert!(all_close(space.ds([0, 0]), [0.5, 0.80], 0.1));
        assert!(all_close(space.ds([0, 5]), [0.5, 2.54], 0.1));
        assert!(all_close(space.ds([2, 3]), [2.0, 6.40], 0.1));
        assert!(all_close(space.ds([4, 2]), [8.0, 20.32], 0.1));
    }
}
