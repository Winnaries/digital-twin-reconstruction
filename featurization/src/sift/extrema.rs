use crate::sift::difference::DoGSpace;
use ndarray::{s, Array, Ix2};

#[allow(dead_code)]
pub fn compute_extrema(dog: DoGSpace) -> (DoGSpace, Array<usize, Ix2>) {
    let mut extremas = Vec::<usize>::new();

    for (o, space) in dog.spaces.iter().enumerate() {
        let dim = space.raw_dim();

        for i in 1..(dim[0] - 1) {
            for j in 1..(dim[1] - 1) {
                for k in 1..(dim[2] - 1) {
                    let window = space.slice(s![i - 1..=i + 1, j - 1..=j + 1, k - 1..=k + 1]);
                    let maximum = window.into_iter().map(|&x| x).reduce(f32::max).unwrap();
                    let minimum = window.into_iter().map(|&x| x).reduce(f32::min).unwrap(); 
                    if window[[1, 1, 1]] == maximum || window[[1, 1, 1]] == minimum {
                        extremas.extend([o, i, j, k].iter());
                    }
                }
            }
        }
    }

    (dog, Array::from_shape_vec([extremas.len() / 4, 4], extremas).unwrap())
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{Axis, Ix3};
    use num::Integer;
    use std::fmt::Debug;

    fn all_eq<T, K, I>(a: T, b: K) -> bool
    where
        T: Clone + IntoIterator<Item = I> + Debug,
        K: Clone + IntoIterator<Item = I> + Debug,
        I: Integer,
    {
        let ai = a.clone().into_iter();
        let bi = b.clone().into_iter();

        if ai.zip(bi).all(|(x, y)| x == y) {
            true
        } else {
            println!("left: {:?}", a);
            println!("right: {:?}", b);
            false
        }
    }

    #[test]
    fn test_extremas_detection() {
        let mut input = DoGSpace {
            num_octaves: 1,
            num_dog_per_octaves: 3,
            spaces: vec![],
        };

        let mut space: Array<f32, Ix3> = Array::zeros([3, 4, 4]);

        space.index_axis_mut(Axis(0), 0).assign(
            &Array::from_shape_vec(
                [4, 4],
                vec![
                    1.0, 1.0, 1.0, 1.0, 
                    0.0, 2.0, 1.0, 1.0,
                    0.5, 1.0, 0.5, 1.0,
                    1.0, 1.0, 1.0, 1.0,
                ],
            )
            .unwrap(),
        );

        space.index_axis_mut(Axis(0), 1).assign(
            &Array::from_shape_vec(
                [4, 4],
                vec![
                    1.0, 1.0, 1.0, 1.0, 
                    0.0, 7.5, 2.0, 1.0,
                    0.5, 0.0, 7.5, 1.0,
                    1.0, 1.0, 1.0, 1.0,
                ],
            )
            .unwrap(),
        );

        space.index_axis_mut(Axis(0), 2).assign(
            &Array::from_shape_vec(
                [4, 4],
                vec![
                    1.5, 2.0, 1.0, 1.0, 
                    0.5, 4.0, 4.0, 1.0,
                    3.0, 2.0, 0.5, 1.0,
                    1.0, 1.0, 1.0, 1.0,
                ],
            )
            .unwrap(),
        );

        input.spaces.push(space);

        let (_, keypoints) = compute_extrema(input);
        println!("{:?}", keypoints);

        assert_eq!(keypoints.raw_dim()[0], 3);
        assert!(all_eq(
            keypoints,
            [0, 1, 1, 1, 0, 1, 2, 1, 0, 1, 2, 2]
        ));
    }
}
