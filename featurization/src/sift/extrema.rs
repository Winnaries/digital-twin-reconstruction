use ndarray::s;

use crate::octave::Octave;

#[allow(dead_code)]
pub fn extrema_handler(octave: Octave) -> (Octave, Vec<(usize, usize, usize)>) {
    let mut extremas = Vec::<_>::new(); 
    let difference_of_gaussian = octave.difference.view(); 
    let dim = difference_of_gaussian.raw_dim(); 

    for i in 1..(dim[0] - 1) {
        for j in 1..(dim[1] - 1) {
            for k in 1..(dim[2] - 1) {
                let window = difference_of_gaussian
                    .slice(s![i-1..i+1, j-1..j+1, k-1..k+1]);
                let maximum = window
                    .into_iter()
                    .map(|&x| { x })
                    .reduce(f32::max)
                    .unwrap(); 
                if window[[1, 1, 1]] == maximum {
                    extremas.push((i as usize, j as usize, k as usize))
                }
            }
        }
    }

    (octave, extremas)
}