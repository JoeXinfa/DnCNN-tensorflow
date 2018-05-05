# -*- coding: utf-8 -*-

"""
"""

from ezcad.gosurvey.loadfile import load_survey_tovadb
from ezcad.gocube.impt.loadfile import load_cube_javaseis
from dncnn.generate_patches_from_cube import generate_patches
from dncnn.img_add_noise import img_add_noise

def main():

    img_dir = "/cpfs/lfs02/data/zhuu/seam/data/Set12"
    img_add_noise(img_dir)
    return

    fn = '/cpfs/lfs02/data/zhuu/seam/tova.db'
    survey = load_survey_tovadb(fn)

    # fn = '/cpfs/lfs02/data/zhuu/seam/subsalt_clean.js'
    # cube = load_cube_javaseis(fn, survey)
    # fn = '/cpfs/lfs02/data/zhuu/seam/data/patches_clean_il1.npy'
    # generate_patches(cube, patch_size=100, stride=50, batch_size=9, save_file=fn)

    fn = '/cpfs/lfs02/data/zhuu/seam/subsalt_randm.js'
    cube = load_cube_javaseis(fn, survey)
    fn = '/cpfs/lfs02/data/zhuu/seam/data/patches_randm_il1.npy'
    generate_patches(cube, patch_size=100, stride=50, batch_size=9, save_file=fn)

    # fn = '/cpfs/lfs02/data/zhuu/seam/subsalt_noisy.js'
    # cube = load_cube_javaseis(fn, survey)
    # fn = '/cpfs/lfs02/data/zhuu/seam/data/patches_noisy.npy'
    # generate_patches(cube, patch_size=100, stride=50, save_file=fn)

if __name__ == '__main__':
    main()
