# -*- coding:utf-8 -*-

import numpy as np

from .histnd import histnd_serial_f64, histnd_parallel_f64
from .histnd import histnd_serial_i64, histnd_parallel_i64
from .histnd import histnd_serial_u64, histnd_parallel_u64

def check_input_shape(samples, bins):
    if samples.shape[1] != len(bins):
        return False
    else:
        return True

def histnd_parallel(samples, bins, chunksize):
    if not check_input_shape(samples, bins):
        return None
    if samples.dtype in [np.float16, np.float32, np.float64]:
        call = histnd_parallel_f64
        samples = samples.astype(np.double)
        bins = [ each_bin.astype(np.double) for each_bin in bins]
    elif samples.dtype == [np.uint8, np.uint16, np.uint32, np.uint64]:
        call = histnd_parallel_u64
        samples = samples.astype(np.uint64)
        bins = [ each_bin.astype(np.uint64) for each_bin in bins]
    elif samples.dtype == [np.int8, np.int16, np.int32, np.int64]:
        call = histnd_parallel_u64
        samples = samples.astype(np.int64)
        bins = [ each_bin.astype(np.int64) for each_bin in bins]
    else:
        raise NotImplementedError(f"Datatype {samples.dtype} is not supported.")
    return call(samples, bins, chunksize)

def histnd_serial(samples, bins):
    if not check_input_shape(samples, bins):
        return None
    if samples.dtype in [np.float16, np.float32, np.float64]:
        call = histnd_serial_f64
        samples = samples.astype(np.double)
        bins = [ each_bin.astype(np.double) for each_bin in bins]
    elif samples.dtype == [np.uint8, np.uint16, np.uint32, np.uint64]:
        call = histnd_serial_u64
        samples = samples.astype(np.uint64)
        bins = [ each_bin.astype(np.uint64) for each_bin in bins]
    elif samples.dtype == [np.int8, np.int16, np.int32, np.int64]:
        call = histnd_serial_u64
        samples = samples.astype(np.int64)
        bins = [ each_bin.astype(np.int64) for each_bin in bins]
    else:
        raise NotImplementedError(f"Datatype {samples.dtype} is not supported.")
    return call(samples, bins)
