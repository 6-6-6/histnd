use std::thread;

use ndarray::parallel::prelude::*;
use ndarray::{Array, ArrayD};
use ndarray::{ArrayView1, ArrayView2};
use ndarray::{Axis, IxDyn};

use std::cmp::PartialOrd;

use crate::binary_search_nd;

fn serial_search<T>(samples: &ArrayView2<T>, bins: &[ArrayView1<T>]) -> Vec<IxDyn>
where
    T: PartialOrd,
{
    let mut ret = Vec::with_capacity(samples.len_of(Axis(0)));
    for sample in samples.axis_iter(Axis(0)) {
        ret.push(binary_search_nd(bins, &sample));
    }
    ret
}

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

    let mut ret_shape = Vec::with_capacity(samples.len_of(Axis(1)));
    for bin in bins.iter() {
        ret_shape.push(bin.len() + 1);
    }

    let ret_allocator = thread::spawn(move || {
        Array::zeros(ret_shape)
    });

    let search_results = serial_search(samples, bins);

    let mut ret = ret_allocator.join().unwrap();
    for result in search_results {
        ret[result] += 1
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

    let mut ret_shape = Vec::with_capacity(samples.len_of(Axis(0)));
    for bin in bins.iter() {
        ret_shape.push(bin.len() + 1);
    }

    let ret_allocator = thread::spawn(move || {
        Array::zeros(ret_shape)
    });

    let mut search_chunks = Vec::new();
    // TODO: into_par_iter
    samples.axis_chunks_iter(Axis(0), chunksize)
        .into_par_iter()
        .map(|chunk| serial_search(&chunk, bins))
        .collect_into_vec(&mut search_chunks);

    let mut ret = ret_allocator.join().unwrap();
    for chunk in search_chunks {
        for result in chunk {
            ret[result] += 1
        }
    }

    Some(ret)
}
