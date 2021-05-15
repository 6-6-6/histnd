use ndarray::{Array1, ArrayView1};

use std::cmp::Ordering;
use std::cmp::PartialOrd;

#[derive(Debug)]
pub struct DimensionNotMatch;
impl std::fmt::Display for DimensionNotMatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "The dimensions of input matrices were not matched.")
    }
}
impl std::error::Error for DimensionNotMatch {}

pub fn binary_search<T>(array: &ArrayView1<T>, x: &T) -> Result<usize, usize>
where
    T: PartialOrd,
{
    let mut right = match array.len() {
        usize::MIN => return Err(0),
        _ => array.len() - 1,
    };
    let mut left = 0;
    let mut mid: usize;

    while left <= right {
        mid = (right + left) / 2;
        match array[mid].partial_cmp(&x) {
            Some(Ordering::Greater) => right = mid.checked_sub(1).map_or(usize::MIN, |n| n),
            Some(Ordering::Less) => left = mid.checked_add(1).map_or(usize::MAX, |n| n),
            Some(Ordering::Equal) => return Ok(mid),
            None => return Err(usize::MIN),
        }
    }
    Err(left)
}

// return what as an error?
pub fn binary_search_nd<T>(array: &[ArrayView1<T>], x: &ArrayView1<T>) -> Array1<usize>
where
    T: PartialOrd,
{
    let mut dim = x.len();
    let mut ret: Array1<usize> = Array1::zeros((dim,));

    while dim != 0 {
        dim -= 1;
        match binary_search(&array[dim], &x[dim]) {
            // TODO: determine whether it should goto left bin or not
            Ok(n) => ret[dim] = n,
            Err(e) => ret[dim] = e,
        }
    }
    ret
}
