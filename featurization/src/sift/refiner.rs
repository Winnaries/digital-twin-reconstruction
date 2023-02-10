use ndarray::{s, stack, Array, Array3, ArrayView, Axis, Ix3, Ix4, Ix5, Slice};

#[allow(dead_code)]
pub fn compute_gradient(space: ArrayView<f32, Ix3>) -> Array<f32, Ix4> {
    let left = Slice::from((2 as isize)..);
    let right = Slice::from(..(-3 as isize));

    #[rustfmt::skip]
    let sdiff: Array<f32, Ix3> = 0.5 * (
        &space.slice_axis(Axis(0), left) - 
        &space.slice_axis(Axis(0), right));
    let sdiff = sdiff
        .slice(s![.., 1..-2 as isize, 1..-2 as isize])
        .to_owned();

    #[rustfmt::skip]
    let mdiff: Array<f32, Ix3> = 0.5 * (
        &space.slice_axis(Axis(1), left) - 
        &space.slice_axis(Axis(1), right));
    let mdiff = mdiff
        .slice(s![1..-2 as isize, .., 1..-2 as isize])
        .to_owned();

    #[rustfmt::skip]
    let ndiff: Array<f32, Ix3> = 0.5 * (
        &space.slice_axis(Axis(2), left) - 
        &space.slice_axis(Axis(2), right));
    let ndiff = ndiff
        .slice(s![1..-2 as isize, 1..-2 as isize, ..])
        .to_owned();

    stack![Axis(3), sdiff, mdiff, ndiff]
}

#[allow(dead_code)]
pub fn compute_hessian(space: ArrayView<f32, Ix3>) -> Array<f32, Ix5> {
    let left = Slice::from(..(-3 as isize));
    let center = Slice::from((1 as isize)..(-2 as isize));
    let right = Slice::from((2 as isize)..);

    #[rustfmt::skip]
    let hxx = |axis: usize| -> Array<f32, Ix3> {
        &space.slice_axis(Axis(axis), left) + 
        &space.slice_axis(Axis(axis), right) - 
        2.0 * &space.slice_axis(Axis(axis), center)
    };

    let hxy = |axis0: usize, axis1: usize| -> Array<f32, Ix3> {
        let axis0 = Axis(axis0);
        let axis1 = Axis(axis1 - 1);
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
