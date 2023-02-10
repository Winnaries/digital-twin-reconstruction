#![allow(dead_code)]

use ndarray::{s, Array, ArrayBase, ArrayView, Ix2};

/// Resample image using bilinear interpolation.
/// Result in an image of size double of the original.
/// In other word, this function assume delta = 0.5
pub fn bilinear_upsample(image: ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
    let odim = image.raw_dim();
    let (owidth, oheight) = (odim[1], odim[0]);
    let (nwidth, nheight) = (owidth * 2, oheight * 2);
    let symmetrize = |(x, y): (usize, usize)| -> (usize, usize) {
        let (dw, dh) = (2 * owidth, 2 * oheight);
        let x = (x % dw).min((dw - 1 - x) % dw);
        let y = (y % dh).min((dh - 1 - y) % dh);
        (x, y)
    };

    ArrayBase::from_shape_fn((nheight, nwidth), |(y, x)| {
        let y = 0.5 * y as f32;
        let x = 0.5 * x as f32;
        let xf = x.floor();
        let yf = y.floor();
        
        #[rustfmt::skip]
        let a = (x - xf) * (y - yf) * 
            image[symmetrize((xf as usize + 1, yf as usize + 1))];
        
        #[rustfmt::skip]
        let b = (1.0 + xf - x) * (y - yf) * 
            image[symmetrize((xf as usize, yf as usize + 1))];
        
        #[rustfmt::skip]
        let c = (x - xf) * (1.0 + yf - y) * 
            image[symmetrize((xf as usize + 1, yf as usize))];
        
        #[rustfmt::skip]
        let d = (1.0 + xf - x) * (1.0 + yf - y) * 
            image[symmetrize((xf as usize, yf as usize))];

        a + b + c + d
    })
}

pub fn gaussian_blur(image: ArrayView<f32, Ix2>, sigma: f32) -> Array<f32, Ix2> {
    let n = (4.0 * sigma).ceil() as usize;
    let width = 2 * n + 1;
    let padded: Array<f32, Ix2> = mirrored_pad(image.view(), n); 

    let kernel: Array<f32, Ix2> = Array::from_shape_fn([width, width], |(y, x)| {
        let (x, y) = (x as i32 - n as i32, y as i32 - n as i32);
        let distance = ((x * x + y * y) as f32).sqrt();
        let prefactor = -(2.0 * sigma * sigma).recip();
        (distance * prefactor).exp()
    });

    let blurred = Array::from_shape_fn(image.raw_dim(), |(y, x)| {
        let slice: ArrayView<f32, Ix2> = padded.slice(s![y..y + width, x..x + width]);
        if x == 0 && y == 0 {
            println!("{:?}", slice); 
            println!("{:?}", kernel); 
        }
        (&slice * &kernel).sum()
    }); 

    let maximum = blurred.iter().map(|&x| { x }).reduce(f32::max).unwrap(); 
    blurred / maximum
}

pub fn mirrored_pad(image: ArrayView<f32, Ix2>, radius: usize) -> Array<f32, Ix2> {
    let dim = image.raw_dim();
    let (oh, ow) = (dim[0] as isize, dim[1] as isize);
    let (nh, nw) = (oh + 2 * radius as isize, ow + 2 * radius as isize);
    let (dh, dw) = (oh * 2, ow * 2);

    Array::from_shape_fn((nh as usize, nw as usize), |(y, x)| {
        let y = y as isize - radius as isize;
        let x = x as isize - radius as isize;

        let sy = (y % dh).min((dh - y - 2) % dh).abs();
        let sx = (x % dw).min((dw - x - 2) % dw).abs();

        image[[sy as usize, sx as usize]]
    })
}

pub fn downsample(image: ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
    let dim = image.raw_dim(); 
    Array::from_shape_fn(((dim[0] + 1) / 2, (dim[1] + 1) / 2), |(y, x)| { image[[y * 2, x * 2]] })
}

#[cfg(test)]
mod test {
    use num::Signed;
    use std::fmt::Debug;

    use super::*;

    fn all_close<T, I>(a: T, b: T, tol: I) -> bool
    where
        T: Clone + IntoIterator<Item = I> + Debug,
        I: Signed + PartialOrd,
    {
        let ai = a.clone().into_iter();
        let bi = b.clone().into_iter();

        if ai.zip(bi).map(|(x, y)| (x - y).abs()).all(|x| x < tol) {
            true
        } else {
            println!("left: {:?}", a);
            println!("right: {:?}", b);
            false
        }
    }

    #[test]
    fn test_mirrored_padding() {
        let input: Array<f32, Ix2> = Array::range(0.0, 9.0, 1.0).into_shape([3, 3]).unwrap();
        let output = mirrored_pad(input.view(), 2);

        #[rustfmt::skip]
        let expected: Array<f32, Ix2> = Array::from_shape_vec([7, 7], vec![
            8.0, 7.0, 6.0, 7.0, 8.0, 7.0, 6.0,
            5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0,
            2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0,
            5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0,
            8.0, 7.0, 6.0, 7.0, 8.0, 7.0, 6.0,
            5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0,
            2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0
        ]).unwrap();

        assert!(all_close(output, expected, 1e-4f32)); 
    }

    #[test]
    fn test_gaussian_blur() {
        let mut input: Array<f32, Ix2> = Array::zeros([9, 9]); 
        input[[4, 4]] = 1.0; 
        let output = gaussian_blur(input.view(), 1.0); 

        #[rustfmt::skip]
        let expected: Array<f32, Ix2> = Array::from_shape_vec([9, 9], vec![
            0.236, 0.164, 0.213, 0.254, 0.270, 0.254, 0.213, 0.164, 0.236,
            0.164, 0.119, 0.164, 0.205, 0.223, 0.205, 0.164, 0.119, 0.164,
            0.213, 0.164, 0.243, 0.326, 0.367, 0.326, 0.243, 0.164, 0.213,
            0.254, 0.205, 0.326, 0.493, 0.606, 0.493, 0.326, 0.205, 0.254,
            0.270, 0.223, 0.367, 0.606, 1.000, 0.606, 0.367, 0.223, 0.270,
            0.254, 0.205, 0.326, 0.493, 0.606, 0.493, 0.326, 0.205, 0.254,
            0.213, 0.164, 0.243, 0.326, 0.367, 0.326, 0.243, 0.164, 0.213,
            0.164, 0.119, 0.164, 0.205, 0.223, 0.205, 0.164, 0.119, 0.164,
            0.236, 0.164, 0.213, 0.254, 0.270, 0.254, 0.213, 0.164, 0.236
        ]).unwrap();

        assert!(all_close(output, expected, 0.001))
    }

    #[test]
    fn test_downsample() {
        let input: Array<_, Ix2> = Array::range(0.0, 16.0, 1.0).into_shape([4, 4]).unwrap(); 
        let output = downsample(input.view()); 
        let expected = Array::from_shape_vec([2, 2], vec![
            0.0, 2.0, 
            8.0, 10.0, 
        ]).unwrap(); 

        assert!(all_close(output, expected, 0.001)); 
    }
}
