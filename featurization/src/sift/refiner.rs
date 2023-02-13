use na::{Matrix3, Vector3};
use nalgebra as na;
use ndarray::{
    s, stack, Array, Array1, Array3, Array4, Array5, ArrayView, Axis, Ix1, Ix3, Ix4, Ix5,
    Slice,
};

#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::absolute_keypoint;

use super::{
    difference::DoGSpace,
    keypoint::{AbsoluteKeypoint, DiscreteKeypoint}, scale::ScaleSpace,
};

pub fn compute_gradient(space: ArrayView<f32, Ix3>) -> Array<f32, Ix4> {
    let right = Slice::from((2 as isize)..);
    let left = Slice::from(..(-2 as isize));

    #[rustfmt::skip]
    let sdiff: Array<f32, Ix3> = 0.5 * (
        &space.slice_axis(Axis(0), right) - 
        &space.slice_axis(Axis(0), left));
    let sdiff = sdiff
        .slice(s![.., 1..-1 as isize, 1..-1 as isize])
        .to_owned();

    #[rustfmt::skip]
    let mdiff: Array<f32, Ix3> = 0.5 * (
        &space.slice_axis(Axis(1), right) - 
        &space.slice_axis(Axis(1), left));
    let mdiff = mdiff
        .slice(s![1..-1 as isize, .., 1..-1 as isize])
        .to_owned();

    #[rustfmt::skip]
    let ndiff: Array<f32, Ix3> = 0.5 * (
        &space.slice_axis(Axis(2), right) - 
        &space.slice_axis(Axis(2), left));
    let ndiff = ndiff
        .slice(s![1..-1 as isize, 1..-1 as isize, ..])
        .to_owned();

    let dim = space.raw_dim();
    let ndim = [dim[0], dim[1], dim[2], 3];
    let mut grad = Array::zeros(ndim);

    grad.slice_mut(s![1..-1 as isize, 1..-1 as isize, 1..-1 as isize, ..])
        .assign(&stack![Axis(3), sdiff, mdiff, ndiff]);

    grad
}

pub fn compute_hessian(space: ArrayView<f32, Ix3>) -> Array<f32, Ix5> {
    let left = Slice::from(..(-2 as isize));
    let center = Slice::from((1 as isize)..(-1 as isize));
    let right = Slice::from((2 as isize)..);

    #[rustfmt::skip]
    let hxx = |axis: usize| -> Array<f32, Ix3> {
        &space.slice_axis(Axis(axis), left) + 
        &space.slice_axis(Axis(axis), right) - 
        2.0 * &space.slice_axis(Axis(axis), center)
    };

    let hxy = |axis0: usize, axis1: usize| -> Array<f32, Ix3> {
        let axis0 = Axis(axis0);
        let axis1 = Axis(axis1);
        0.25 * (&space.slice_axis(axis0, right).slice_axis(axis1, right)
            - &space.slice_axis(axis0, right).slice_axis(axis1, left)
            - &space.slice_axis(axis0, left).slice_axis(axis1, right)
            + &space.slice_axis(axis0, left).slice_axis(axis1, left))
    };

    let h11: Array3<f32> = hxx(0)
        .slice_axis(Axis(1), center)
        .slice_axis(Axis(2), center)
        .to_owned();
    let h22: Array3<f32> = hxx(1)
        .slice_axis(Axis(0), center)
        .slice_axis(Axis(2), center)
        .to_owned();
    let h33: Array3<f32> = hxx(2)
        .slice_axis(Axis(0), center)
        .slice_axis(Axis(1), center)
        .to_owned();

    let h12: Array3<f32> = hxy(0, 1).slice_axis(Axis(2), center).to_owned();
    let h13: Array3<f32> = hxy(0, 2).slice_axis(Axis(1), center).to_owned();
    let h23: Array3<f32> = hxy(1, 2).slice_axis(Axis(0), center).to_owned();

    let dim = space.raw_dim();
    let pdim = [dim[0], dim[1], dim[2], 3, 3];
    let ndim = [dim[0] - 2, dim[1] - 2, dim[2] - 2, 3, 3];

    let hessian = stack!(Axis(3), h11, h12, h13, h12, h22, h23, h13, h23, h33,)
        .into_shape(ndim)
        .unwrap();

    let mut padded = Array::zeros(pdim);
    padded
        .slice_mut(s![1..-1 as isize, 1..-1 as isize, 1..-1 as isize, .., ..])
        .assign(&hessian);
    padded
}

pub fn quadratic_interpolate(
    space: ArrayView<f32, Ix3>,
    gradient: ArrayView<f32, Ix4>,
    alpha: ArrayView<f32, Ix1>,
    syx: [usize; 3],
) -> f32 {
    let [s, y, x] = syx;
    let space = space.slice(s![s, y, x].to_owned());
    let gradient = gradient.slice(s![s, y, x, ..]).to_owned();
    let omega = &space + 0.5 * gradient.dot(&alpha);
    omega.into_scalar()
}

pub fn max_quadratic_interpolate(
    gradient: ArrayView<f32, Ix4>,
    hessian: ArrayView<f32, Ix5>,
    syx: [usize; 3],
) -> Array<f32, Ix1> {
    let [s, y, x] = syx;

    let gradient = gradient.slice(s![s, y, x, ..]).to_owned();
    let hessian = hessian.slice(s![s, y, x, .., ..]).to_owned();

    let gradient = Vector3::from_iterator(gradient);
    let hessian = Matrix3::from_iterator(hessian);

    let alpha = -hessian.try_inverse().unwrap() * gradient;

    Array::from_vec(vec![alpha[(0, 0)], alpha[(1, 0)], alpha[(2, 0)]])
}

pub fn compute_edgeness(hessian: ArrayView<f32, Ix5>, syx: [usize; 3]) -> f32 {
    let [s, y, x] = syx;
    let hessian = hessian.slice(s![s, y, x, 1 as isize.., 1 as isize..]);
    (hessian[[0, 0]] + hessian[[1, 1]]).powi(2)
        / (hessian[[0, 0]] * hessian[[1, 1]] - hessian[[0, 1]] * hessian[[1, 0]])
}

pub fn refine_keypoints_with_low_contrast(
    scale: &ScaleSpace, 
    dog: &DoGSpace,
    points: Vec<DiscreteKeypoint>,
) -> Vec<AbsoluteKeypoint> {
    println!("｜\t⌞ Computing gradient"); 
    let gradient: Vec<Array4<_>> = dog
        .spaces
        .iter()
        .map(|s: &Array3<_>| compute_gradient(s.view()))
        .collect();
        
    println!("｜\t⌞ Computing hessian"); 
    let hessian: Vec<Array5<_>> = dog
        .spaces
        .iter()
        .map(|s: &Array3<_>| compute_hessian(s.view()))
        .collect();

    let dim: Vec<[usize; 3]> = gradient
        .iter()
        .map(|x| [x.dim().0, x.dim().1, x.dim().2])
        .collect(); 

    let compute_alpha = |point: DiscreteKeypoint| {
        let mut attempt = 0;
        let DiscreteKeypoint {
            o,
            mut s,
            mut m,
            mut n,
        } = point;

        loop {
            if attempt > 5 {
                break None;
            } else {
                attempt += 1
            }

            #[rustfmt::skip]
            let a: Array1<f32> = max_quadratic_interpolate(
                gradient[o].view(), 
                hessian[o].view(), 
                [s, m, n]
            );

            if a.iter().all(|x| x.abs() < 0.6) {
                break Some((o, s, m, n, a));
            } else {
                s = ((s as f32 + a[[0]]).round() as usize).clamp(1, dim[o][0] - 2);
                m = ((m as f32 + a[[1]]).round() as usize).clamp(1, dim[o][1] - 2);
                n = ((n as f32 + a[[2]]).round() as usize).clamp(1, dim[o][2] - 2);
            }
        }
    }; 

    let compute_absolute_point = |(o, s, m, n, alpha): (usize, usize, usize, usize, Array1<f32>)| {
        let omega = quadratic_interpolate(
            dog.spaces[o].view(), 
            gradient[o].view(),
            alpha.view(),
            [s, m, n],
        );

        let [_, sigma, y, x] = scale.dsyx(
            o, 
            s as f32 + alpha[[0]], 
            m as f32 + alpha[[1]], 
            n as f32 + alpha[[2]]
        ); 

        if omega > 0.015 {
            Some(absolute_keypoint!(o, s, m, n, y, x, sigma, omega))
        } else {
            None
        }
    }; 

    #[cfg(not(feature = "parallel"))]
    let result = points
        .into_iter()
        .filter_map(compute_alpha)
        .filter_map(compute_absolute_point)
        .collect(); 

    #[cfg(feature = "parallel")]
    let result = points.into_par_iter()
        .filter_map(compute_alpha)
        .filter_map(compute_absolute_point)
        .collect(); 

    result
}

pub fn refine_keypoints_on_edge(dog: &DoGSpace, points: Vec<AbsoluteKeypoint>) -> Vec<AbsoluteKeypoint> {
    let hessian: Vec<Array5<_>> = dog
        .spaces
        .iter()
        .map(|s: &Array3<_>| compute_hessian(s.view()))
        .collect();

    let filter = |p: AbsoluteKeypoint| {
        let DiscreteKeypoint {
            o, 
            s, 
            m, 
            n, 
        } = p.inner; 

        let edgeness = compute_edgeness(hessian[o].view(), [s, m, n]); 
        let c_edge = 10.0f32; 
        let check = (c_edge + 1.0).powi(2) / c_edge; 

        if edgeness > check { None } else { Some(p) }
    }; 

    #[cfg(not(feature = "parallel"))]
    let result = points.into_iter().filter_map(filter).collect(); 
    
    #[cfg(feature = "parallel")]
    let result = points.into_par_iter().filter_map(filter).collect(); 

    result
}

#[cfg(test)]
mod test {
    use std::fmt::Debug;

    use super::*;
    use ndarray::{s, Array};
    use num::Signed;

    fn all_close<T, K, I>(a: T, b: K, tol: I) -> bool
    where
        T: Clone + IntoIterator<Item = I> + Debug,
        K: Clone + IntoIterator<Item = I> + Debug,
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
    fn test_compute_gradient() {
        let input = Array::range(0.0, 27.0, 1.0).into_shape([3, 3, 3]).unwrap();
        let output = compute_gradient(input.view());

        assert_eq!(output.dim().0, 3);
        assert_eq!(output.dim().1, 3);
        assert_eq!(output.dim().2, 3);
        assert_eq!(output.dim().3, 3);

        assert!(all_close(
            output
                .slice(s![1..-1 as isize, 1..-1 as isize, 1..-1 as isize, ..])
                .to_owned(),
            [9.0, 3.0, 1.0],
            0.0001
        ));
    }

    #[test]
    fn test_compute_hessian() {
        let input = Array::range(0.0, 27.0, 1.0).into_shape([3, 3, 3]).unwrap();
        let output = compute_hessian(input.view());

        assert_eq!(output.dim().0, 3);
        assert_eq!(output.dim().1, 3);
        assert_eq!(output.dim().2, 3);
        assert_eq!(output.dim().3, 3);
        assert_eq!(output.dim().4, 3);

        assert!(all_close(
            output
                .slice(s![1..-1 as isize, 1..-1 as isize, 1..-1 as isize, .., ..])
                .to_owned(),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
            0.0001
        ));

        let mut input = Array::zeros([3, 3, 3]);
        input.fill(1.0);
        input[[1, 1, 1]] = 0.0;

        let output = compute_hessian(input.view());

        assert_eq!(output.dim().0, 3);
        assert_eq!(output.dim().1, 3);
        assert_eq!(output.dim().2, 3);
        assert_eq!(output.dim().3, 3);
        assert_eq!(output.dim().4, 3);

        assert!(all_close(
            output
                .slice(s![1..-1 as isize, 1..-1 as isize, 1..-1 as isize, .., ..])
                .to_owned(),
            [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0,],
            0.0001
        ));
    }

    #[test]
    fn test_max_quadratic_interpolation() {
        let mut input = Array::zeros([3, 3, 3]);
        input.fill(1.0);
        input[[1, 1, 1]] = 2.0;
        input[[2, 1, 1]] = 1.5;

        let gradient = compute_gradient(input.view());
        let hessian = compute_hessian(input.view());
        let syx = [1, 1, 1];

        let output = max_quadratic_interpolate(gradient.view(), hessian.view(), syx);
        assert!(all_close([0.1666, 0.0, 0.0], output, 0.0001));
    }

    #[test]
    fn test_quadratic_interpolation() {
        let mut input = Array::zeros([3, 3, 3]);

        input.fill(1.0);
        input[[1, 1, 1]] = 2.0;
        input[[2, 1, 1]] = 1.5;

        let gradient = compute_gradient(input.view());
        let hessian = compute_hessian(input.view());
        let syx = [1, 1, 1];

        let alpha = max_quadratic_interpolate(gradient.view(), hessian.view(), syx);
        let output = quadratic_interpolate(input.view(), gradient.view(), alpha.view(), syx);

        assert!((output - 2.0208).abs() < 0.0001);
    }
}
