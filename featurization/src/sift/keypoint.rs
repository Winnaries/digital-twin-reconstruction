use std::ops::{Deref, DerefMut};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DiscreteKeypoint {
    pub o: usize, 
    pub s: usize, 
    pub m: usize, 
    pub n: usize, 
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AbsoluteKeypoint {
    pub x: f32, 
    pub y: f32, 
    pub sigma: f32, 
    pub omega: f32, 
    pub inner: DiscreteKeypoint, 
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct OrientedKeypoint {
    pub inner: AbsoluteKeypoint, 
    pub theta: f32, 
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct FeaturedKeypoint {
    pub inner: OrientedKeypoint, 
    pub features: Box<[f32; 128]>, 
}

impl Deref for AbsoluteKeypoint {
    type Target = DiscreteKeypoint; 

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Deref for OrientedKeypoint {
    type Target = AbsoluteKeypoint; 

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Deref for FeaturedKeypoint {
    type Target = OrientedKeypoint; 

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for AbsoluteKeypoint {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl DerefMut for OrientedKeypoint {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl DerefMut for FeaturedKeypoint {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[macro_export]
macro_rules! discrete_keypoint {
    ($o:expr, $s:expr, $m:expr, $n:expr) => {
        DiscreteKeypoint {
            o: $o, 
            s: $s,
            m: $m,
            n: $n,
        }
    };
}

#[macro_export]
macro_rules! absolute_keypoint {
    ($o:expr, $s:expr, $m:expr, $n:expr, $y:expr, $x:expr, $sigma:expr, $omega: expr) => {
        AbsoluteKeypoint {
            x: $x, 
            y: $y, 
            sigma: $sigma, 
            omega: $omega, 
            inner: DiscreteKeypoint {
                o: $o, 
                s: $s,
                m: $m,
                n: $n,
            }
        }
    };
}

#[cfg(test)]
mod test {
    use super::*; 

    #[test]
    fn test_implicit_derefencing() {
        let keypoint = OrientedKeypoint::default(); 
        let mut keypoint = FeaturedKeypoint {
            inner: keypoint, 
            features: Box::new([0.0; 128]), 
        }; 

        keypoint.m = 1; 
        keypoint.x = 2.0; 
        keypoint.sigma = 1.0; 

        assert_eq!(keypoint.m, 1);
        assert_eq!(keypoint.x, 2.0);
        assert_eq!(keypoint.sigma, 1.0);
    }

}