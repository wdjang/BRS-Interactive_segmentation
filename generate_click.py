import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

def generate_click(fn_map, fp_map, click_map, net_size, y_meshgrid, x_meshgrid):
    fn_map = np.pad(fn_map, ((1,1),(1,1)), 'constant')
    fndist_map = distance_transform_edt(fn_map)
    fndist_map = fndist_map[1:-1, 1:-1]
    fndist_map = np.multiply(fndist_map, 1-click_map)

    fp_map = np.pad(fp_map, ((1,1),(1,1)), 'constant')
    fpdist_map = distance_transform_edt(fp_map)
    fpdist_map = fpdist_map[1:-1, 1:-1]
    fpdist_map = np.multiply(fpdist_map, 1-click_map)

    if np.max(fndist_map) > np.max(fpdist_map):
        is_pos = 1
        return fndist_map, is_pos
    else:
        is_pos = 0
        return fpdist_map, is_pos