use ndarray::{Array, ArrayView, Ix3, Ix4, s};

#[allow(dead_code)]
fn compute_spatial_gradient(space: ArrayView<f32, Ix3>) -> Array<f32, Ix4> {
    let shape = space.raw_dim(); 
    let (s, m, n) = (shape[0], shape[1], shape[2]); 

    let dy: Array<f32, Ix3> = 0.5 * (
        &space.slice(s![.., 2 as isize.., ..]) - 
        &space.slice(s![.., ..-2 as isize, ..])
    );
    
    let dx: Array<f32, Ix3> = 0.5 * (
        &space.slice(s![.., .., 2 as isize..]) -
        &space.slice(s![.., .., ..-2 as isize])
    );

    let mut gradient = Array::zeros([s, m, n, 2]);
    gradient.slice_mut(s![.., 1..-1 as isize, 1..-1 as isize, 0]).assign(&dy); 
    gradient.slice_mut(s![.., 1..-1 as isize, 1..-1 as isize, 1]).assign(&dx); 

    gradient
}