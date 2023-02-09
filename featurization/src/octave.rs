use ndarray::{Array, Axis, Ix2, Ix3, stack, ArrayView, s};
use crate::proc::{MatrixAsImage, to_image, to_ndarray, gaussian_blur};

pub struct Octave {
    pub s: u32, 
    pub gaussian: Array<f32, Ix3>, 
    pub difference: Array<f32, Ix3>, 
}

impl Octave {

    pub fn from_image(image: &MatrixAsImage, s: u32) -> Self {
        let mut scale_space = Vec::<MatrixAsImage>::new(); 
        let length = s + 3;

        for idx in 0..length {
            let k = (2.0f32).powf((s as f32).recip()); 
            let sigma = k.powi(idx as i32); 
            let blurred = gaussian_blur(image, sigma); 
            scale_space.push(blurred);
        }

        let scale_space: Vec<Array<f32, Ix2>> = scale_space
            .into_iter()
            .map(|x| { to_ndarray(x) })
            .collect(); 

        let scale_view: Vec<ArrayView<f32, Ix2>> = scale_space
            .iter()
            .map(|x| { x.view() })
            .collect();

        let gaussian = stack(Axis(0), &scale_view).unwrap(); 
        let left = gaussian.slice(s![1.., .., ..]); 
        let right = gaussian.slice(s![..-1, .., ..]);
        let difference = &left - &right; 

        Octave {
            s, 
            gaussian, 
            difference
        }
    }

    pub fn from_ndarray(array: Array<f32, Ix2>, s: u32) -> Self {
        let image = &to_image(array.view()); 
        Self::from_image(&image, s)
    } 

}