import numpy as np
import sys

if __name__ == '__main__':
    theta, filepath = float(sys.argv[1]), sys.argv[2]
    mat = np.loadtxt(filepath, dtype=np.float32)
    mat[mat < theta] = np.nan
    # us, vs = np.nonzero(~np.isnan(mat))
    # print(us, vs, len(us), len(vs))
    mat[mat == 1.] = np.nan
    us, vs = np.nonzero(~np.isnan(mat))
    print(us, vs, len(us), len(vs))
    for i in range(len(us)):
        print(f'{us[i]}\ttt\t{vs[i]}')
