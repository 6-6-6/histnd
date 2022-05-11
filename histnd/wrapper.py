# -*- coding:utf-8 -*-

import numpy as np

from .histnd import histnd_serial_f64, histnd_parallel_f64, histnd_serial_weighted_f64, histnd_parallel_weighted_f64
from .histnd import histnd_serial_i64, histnd_parallel_i64, histnd_serial_weighted_i64, histnd_parallel_weighted_i64
from .histnd import histnd_serial_u64, histnd_parallel_u64, histnd_serial_weighted_u64, histnd_parallel_weighted_u64

def check_input_shape(samples, bins):
    if samples.shape[1] != len(bins):
        return False
    else:
        return True

def histnd_parallel(samples, bins, chunksize, weight=1):
    if not check_input_shape(samples, bins):
        return None
    # check weight
    if isinstance(weight, np.ndarray):
        weight = weight.astype(np.double)
    else:
        weight = np.double(weight)
    # check and format input datatypes
    # for floating points
    if samples.dtype in [np.float16, np.float32, np.float64]:
        if isinstance(weight, np.ndarray):
            call = histnd_parallel_weighted_f64
        else:
            call = histnd_parallel_f64
        samples = samples.astype(np.double)
        bins = [ each_bin.astype(np.double) for each_bin in bins]
    elif samples.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if isinstance(weight, np.ndarray):
            call = histnd_parallel_weighted_u64
        else:
            call = histnd_parallel_u64
        samples = samples.astype(np.uint64)
        bins = [ each_bin.astype(np.uint64) for each_bin in bins]
    elif samples.dtype in [np.int8, np.int16, np.int32, np.int64]:
        if isinstance(weight, np.ndarray):
            call = histnd_parallel_weighted_i64
        else:
            call = histnd_parallel_i64
        samples = samples.astype(np.int64)
        bins = [ each_bin.astype(np.int64) for each_bin in bins]
    else:
        raise NotImplementedError(f"Datatype {samples.dtype} is not supported.")
    return call(samples, bins, weight, chunksize)

def histnd_serial(samples, bins, weight=1):
    if not check_input_shape(samples, bins):
        return None
    # check weight
    if isinstance(weight, np.ndarray):
        weight = weight.astype(np.double)
    else:
        weight = np.double(weight)
    # check and format input datatypes
    # for floating points
    if samples.dtype in [np.float16, np.float32, np.float64]:
        if isinstance(weight, np.ndarray):
            call = histnd_serial_weighted_f64
        else:
            call = histnd_serial_f64
        samples = samples.astype(np.double)
        bins = [ each_bin.astype(np.double) for each_bin in bins]
    # for unsigned intergers
    elif samples.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if isinstance(weight, np.ndarray):
            call = histnd_serial_weighted_u64
        else:
            call = histnd_serial_u64
        samples = samples.astype(np.uint64)
        bins = [ each_bin.astype(np.uint64) for each_bin in bins]
    # for integers
    elif samples.dtype in [np.int8, np.int16, np.int32, np.int64]:
        if isinstance(weight, np.ndarray):
            call = histnd_serial_weighted_i64
        else:
            call = histnd_serial_i64
        samples = samples.astype(np.int64)
        bins = [ each_bin.astype(np.int64) for each_bin in bins]
    else:
        raise NotImplementedError(f"Datatype {samples.dtype} is not supported.")
    # check

    return call(samples, bins, weight)
