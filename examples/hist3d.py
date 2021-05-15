
import numpy as np
import histnd

nsize = 1000

# prepare samples
x_np = np.array(np.random.random(nsize), dtype='float64')
y_np = np.array(np.random.random(nsize), dtype='float64')
z_np = np.array(np.random.random(nsize), dtype='float64')
c3d_np = np.vstack([x_np, y_np, z_np]).T
c3d_np_f32 = c3d_np.astype(np.float32)
# bins
bins = np.arange(101, dtype='float64')/100

# directly call Rust function
histnd.histnd_parallel_f64(c3d_np, [bins]*3, int(nsize/8))

# with a wrapper, simplifies the usage
histnd.histnd_parallel(c3d_np, [bins]*3, int(nsize/8))

# serial version
histnd.histnd_serial(c3d_np, [bins]*3)
