# -*- coding: utf-8 -*-

"""
"""

from ezcad.gosurvey.loadfile import load_survey_tovadb
from ezcad.gocube.impt.loadfile import load_cube_javaseis
from dncnn.generate_patches_from_cube import generate_patches

def main():
    fn = '/cpfs/lfs02/data/zhuu/seam/tova.db'
    survey = load_survey_tovadb(fn)

    # fn = '/cpfs/lfs02/data/zhuu/seam/subsalt_clean.js'
    # cube = load_cube_javaseis(fn, survey)
    # fn = '/cpfs/lfs02/data/zhuu/seam/patches_clean.npy'
    # generate_patches(cube, stride=20, save_file=fn)

    fn = '/cpfs/lfs02/data/zhuu/seam/subsalt_randm.js'
    cube = load_cube_javaseis(fn, survey)
    fn = '/cpfs/lfs02/data/zhuu/seam/patches_randm.npy'
    generate_patches(cube, stride=20, save_file=fn)

if __name__ == '__main__':
    main()
