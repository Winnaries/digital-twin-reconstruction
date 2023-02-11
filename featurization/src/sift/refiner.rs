use na::{Matrix3, Vector3};
use nalgebra as na;
use ndarray::{s, stack, Array, Array3, ArrayView, Axis, Ix3, Ix4, Ix5, Slice};

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
pub fn quadratic_interpolate(
    gradient: ArrayView<f32, Ix4>,
    hessian: ArrayView<f32, Ix5>,
    syx: [usize; 3],
) -> Option<[f32; 3]> {
    let [s, y, x] = syx;
    let gradient = gradient.slice(s![s, y, x, ..]).to_owned();
    let hessian = hessian.slice(s![s, y, x, .., ..]).to_owned();

    let gradient = Vector3::from_iterator(gradient);
    let hessian = Matrix3::from_iterator(hessian);

    let alpha = -hessian.try_inverse().unwrap() * gradient;

    if alpha.iter().all(|x| x.abs() < 0.6) {
        Some([alpha[(0, 0)], alpha[(1, 0)], alpha[(2, 0)]])
    } else {
        None
    }
}

#[cfg(test)]
mod test {
    use std::fmt::Debug;

    use super::{compute_gradient, compute_hessian, quadratic_interpolate};
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
    fn test_quadratic_interplocation() {
        let mut input = Array::zeros([3, 3, 3]);
        input.fill(1.0);
        input[[1, 1, 1]] = 2.0;
        input[[2, 1, 1]] = 1.5;

        let gradient = compute_gradient(input.view());
        let hessian = compute_hessian(input.view());
        let syx = [1, 1, 1];

        let output = quadratic_interpolate(gradient.view(), hessian.view(), syx);
        assert!(output.is_some());

        let output = output.unwrap();
        assert!(all_close([0.1666, 0.0, 0.0], output, 0.0001));
    }
}
