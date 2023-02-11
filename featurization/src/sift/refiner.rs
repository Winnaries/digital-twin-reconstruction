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

    println!("{:?}", (sdiff.view(), mdiff.view(), ndiff.view()));
    stack![Axis(3), sdiff, mdiff, ndiff]
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
    let ndim = [dim[0] - 2, dim[1] - 2, dim[2] - 2, 3, 3];

    #[rustfmt::skip]
    let hessian = stack!(Axis(3), 
        h11, h12, h13, 
        h12, h22, h23, 
        h13, h23, h33,
    );

    hessian.into_shape(ndim).unwrap()
}


#[cfg(test)]
mod test {
    use std::fmt::Debug;

    use ndarray::Array;
    use num::Signed;
    use super::{compute_gradient, compute_hessian};

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
        assert!(all_close(output, [9.0, 3.0, 1.0], 0.0001)); 
    }

    #[test]
    fn test_compute_hessian() {
        let input = Array::range(0.0, 27.0, 1.0).into_shape([3, 3, 3]).unwrap();
        let output = compute_hessian(input.view());
        
        assert_eq!(output.dim().0, 1); 
        assert_eq!(output.dim().1, 1); 
        assert_eq!(output.dim().2, 1); 
        assert_eq!(output.dim().3, 3); 
        assert_eq!(output.dim().4, 3); 

        assert!(all_close(output, [
            0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 
        ], 0.0001)); 

        let mut input = Array::zeros([3, 3, 3]); 
        input.fill(1.0); 
        input[[1, 1, 1]] = 0.0; 
        
        let output = compute_hessian(input.view()); 
    
        assert_eq!(output.dim().0, 1); 
        assert_eq!(output.dim().1, 1); 
        assert_eq!(output.dim().2, 1); 
        assert_eq!(output.dim().3, 3); 
        assert_eq!(output.dim().4, 3); 

        assert!(all_close(output, [
            2.0, 0.0, 0.0, 
            0.0, 2.0, 0.0,
            0.0, 0.0, 2.0,
        ], 0.0001));
    }

}