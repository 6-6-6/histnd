use std::thread;

use ndarray::parallel::prelude::*;
use ndarray::{Array, ArrayD};
use ndarray::{ArrayView1, ArrayView2};
use ndarray::{Axis, IxDyn};

use std::cmp::PartialOrd;

use crate::binary_search_nd;

pub enum Weight<'a> {
    NoWeight(f64),
    HasWeight(ArrayView1<'a, f64>),
}

fn serial_search<T>(samples: ArrayView2<T>, bins: &[ArrayView1<T>]) -> Vec<IxDyn>
where
    T: PartialOrd,
{
    let mut ret = Vec::with_capacity(samples.len_of(Axis(0)));
    for sample in samples.axis_iter(Axis(0)) {
        ret.push(binary_search_nd(bins, &sample));
    }
    ret
}

pub fn histnd_serial<T>(
    samples: ArrayView2<T>,
    bins: &[ArrayView1<T>],
    weight: Weight,
) -> Option<ArrayD<f64>>
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

    let ret_allocator = thread::spawn(move || Array::zeros(ret_shape));

    let search_results = serial_search(samples, bins);

    let mut ret = ret_allocator.join().unwrap();

    // add support for weighting the samples
    match weight {
        Weight::NoWeight(constant) => search_results
            .iter()
            .map(|result| ret[result] += constant)
            .collect(),
        Weight::HasWeight(my_weight) => search_results
            .iter()
            .zip(my_weight.iter())
            .map(|(result, weight)| ret[result] += weight)
            .collect(),
    }

    Some(ret)
}

pub fn histnd_parallel<T>(
    samples: ArrayView2<T>,
    bins: &[ArrayView1<T>],
    weight: Weight,
    chunksize: usize,
) -> Option<ArrayD<f64>>
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

    let ret_allocator = thread::spawn(move || Array::zeros(ret_shape));

    let mut search_chunks = Vec::new();
    // TODO: into_par_iter
    samples
        .axis_chunks_iter(Axis(0), chunksize)
        .into_par_iter()
        .map(|chunk| serial_search(chunk, bins))
        .collect_into_vec(&mut search_chunks);

    let mut ret = ret_allocator.join().unwrap();
    match weight {
        Weight::NoWeight(constant) => {
            for chunk in search_chunks.iter() {
                chunk.iter().map(|result| ret[result] += constant).collect()
            }
        }
        Weight::HasWeight(my_weight) => {
            let weight_iter = my_weight.axis_chunks_iter(Axis(0), chunksize);
            for (chunk, weight_chunk) in search_chunks.iter().zip(weight_iter) {
                chunk
                    .iter()
                    .zip(weight_chunk)
                    .map(|(result, weight)| ret[result] += weight)
                    .collect()
            }
        }
    }

    Some(ret)
}
