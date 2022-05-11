
import sys
import timeit

def init_string(nsize):
    return f"""
import numpy as np
import histnd

nsize = {nsize}

x_np = np.array(np.random.random(nsize), dtype='float64')
y_np = np.array(np.random.random(nsize), dtype='float64')
z_np = np.array(np.random.random(nsize), dtype='float64')

c3d_np = np.vstack([x_np, y_np, z_np]).T

bins = np.arange(101, dtype='float64')/100"""

# directly call Rust function
# use 6 cores
run_parallel = "histnd.histnd_parallel_f64(c3d_np, [bins]*3, 1, int(nsize/6))"

# with a wrapper, simplifies the usage
run_serial = "histnd.histnd_serial_f64(c3d_np, [bins]*3, 1)"

# serial version
run_np = 'np.histogramdd(c3d_np, bins=100, range=[(0,1)]*3)'

n = int(sys.argv[1])

print(f"It takes {timeit.timeit(setup=init_string(n), stmt=run_parallel, number=10)}s to compute histgram for {n} samples for the parallel version.")
print(f"It takes {timeit.timeit(setup=init_string(n), stmt=run_serial, number=10)}s to compute histgram for {n} samples for the serial version.")
print(f"It takes {timeit.timeit(setup=init_string(n), stmt=run_np, number=10)}s to compute histgram for {n} samples for numpy.")