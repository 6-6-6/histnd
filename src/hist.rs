use ndarray::parallel::prelude::*;
use ndarray::{Array, ArrayD};
use ndarray::{ArrayView1, ArrayView2};
use ndarray::{Axis, IxDyn};

use std::cmp::PartialOrd;

use crate::binary_search_nd;

pub fn histnd_serial<T>(samples: &ArrayView2<T>, bins: &[ArrayView1<T>]) -> Option<ArrayD<usize>>
where
    T: PartialOrd,
{
    // samples
    let sample_shape = samples.shape();
    //let number_of_elems = sample_shape[0];
    let dimensions = sample_shape[1];

    // bins
    let hist_dimension = bins.len();

    //
    if dimensions != hist_dimension {
        return None;
    }

    let mut ret_shape = Vec::new();
    for bin in bins.iter() {
        ret_shape.push(bin.len() + 1);
    }
    let mut ret = Array::zeros(ret_shape);

    // TODO: into_par_iter
    let search_results = samples
        .axis_iter(Axis(0))
        .into_iter()
        .map(|x| binary_search_nd(bins, &x));

    for result in search_results {
        ret[IxDyn(&result.to_vec())] += 1
    }

    Some(ret)
}

pub fn histnd_parallel<T>(
    samples: &ArrayView2<T>,
    bins: &[ArrayView1<T>],
    chunksize: usize,
) -> Option<ArrayD<usize>>
where
    T: PartialOrd + Send + Sync,
{
    // samples
    let sample_shape = samples.shape();
    //let number_of_elems = sample_shape[0];
    let dimensions = sample_shape[1];

    // bins
    let hist_dimension = bins.len();

    //
    if dimensions != hist_dimension {
        return None;
    }

    let mut ret_shape = Vec::new();
    for bin in bins.iter() {
        ret_shape.push(bin.len() + 1);
    }
    let mut ret = Array::zeros(ret_shape);

    // search in parallel
    let mut search_results = Vec::new();
    samples
        .axis_chunks_iter(Axis(0), chunksize)
        .into_par_iter()
        .map(|chunk| histnd_serial(&chunk, bins))
        .collect_into_vec(&mut search_results);

    for result in search_results {
        if let Some(n) = result {
            ret = ret + n
        }
    }

    Some(ret)
}
