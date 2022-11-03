# basic imports
from argparse import ArgumentParser
import os
import glob
import numpy as np
# method imports
import util
from algo import pdhg_rotate_isotv_anisotv

def preprocess(data):
    '''Preprocess the data'''
    # renormalise data
    data_renorm = util.correct_normalisation(data)
    # pad data
    data_pad = util.pad_zeros(data_renorm)
    # apply beam hardening correction
    data_BHC = util.apply_BHC(data_pad)
    return data_BHC

def segment(data, segment_type):
    '''Run segmentation on data with method segment_type'''
    if segment_type == 1:
        ss = util.apply_global_threshold(data)
    elif segment_type == 2:
        ss = util.apply_crazy_threshold(data)
    return util.flipud_unpack(ss)
 
def create_lb_ub(data, ig, ub_mask_type, lb_mask_type, ub_val, lb_val, basic_mask_radius, lb_inner_radius):
    # create default lower bound mask
    lb = ig.allocate(0.0)
    # create upper bound mask
    if ub_mask_type == 1:
        ub = ig.allocate(ub_val)
        ub = util.apply_circular_mask(ub, basic_mask_radius)
    elif ub_mask_type == 2:
        # sample mask with upper bound to acrylic attenuation
        ub = ig.allocate(0)
        circle_parameters = util.find_circle_parameters(data, ig)
        util.fill_circular_mask(circle_parameters, ub.array, \
            ub_val, *ub.shape)
        # create lower bound mask annulus if needed
        if lb_mask_type == 1:
            inner_circle_parameters = circle_parameters.copy()
            inner_circle_parameters[0] = lb_inner_radius
            util.fill_circular_mask(circle_parameters, lb.array, lb_val, *ub.shape)
            inner = ig.allocate(0.0)
            util.fill_circular_mask(inner_circle_parameters, inner.array, 1.0, *ub.shape)
            lb.array[inner.array.astype(bool)==1.0] = 0.0

    return lb, ub


def main():
    '''
    Example of use:

    python main_full.py <input folder> <output folder> <difficulty> -alpha=0.01 -ang_range=60 -ang_start=0 -ub_mask_type=2 -lb_mask_type=0 -ub_val=0.040859 -lb_val=0.040859 -num_iters=2000 -seg_method=2

    This example uses all of the same parameters as main.py [except ang_start and range aren't set there, but by the data]. This was used to test that we get the same result by
    running this script with the htc_2022_ta_full.mat file, as we do with running main.py with the htc_2022_ta_sparse_example.mat, by checking the scores are equal.
    
    Note, in the above example, omega was not set as an input, so it fell back to the default of 90/ang_range, but omega can be set using -omega=
    '''
    parser = ArgumentParser(description= 'CIL Team Algorithm 1')
    parser.add_argument('in_folder', help='Input folder, containing FULL data')
    parser.add_argument('out_folder', help='Output folder, where PNGs will be written')
    parser.add_argument('difficulty', type=int)
    parser.add_argument('-omega', type=float, help="Omega. If not set, defaults to 90/ang_range")
    parser.add_argument('-alpha', type=float, required=True, help= "Alpha. This is required.")
    parser.add_argument('-ang_start', type=int, required=True, help="Starting angle, degrees. This is required.")
    parser.add_argument('-ang_range', type=int, required=True, help="Angular range, degrees. This is required.")
    parser.add_argument('-ub_mask_type', type=int, required=True, choices=[1, 2],  help= "1 basic 0.97 circle. 2 fitted")
    parser.add_argument('-lb_mask_type', type=int, required=True, choices=[0, 1],  help= "0:  lower bound 0 everywhere, 1: outer annulus equal to upper bound acrylic")
    parser.add_argument('-ub_val', type=float, required=True, help= "Upper bound value, acrylic_attenuation in unit 1/mm")
    parser.add_argument('-lb_val', type=float, required=True, help= "Lower bound value, could be changed to 0.04 or other smaller values")
    parser.add_argument('-num_iters', type=int, required=True, help="Number of iterations.")
    parser.add_argument('-seg_method', type=int, choices=[1, 2], required=True, help="1: basic thresholding, 2: crazy")


    args = parser.parse_args()

    input_folder = os.path.abspath(args.in_folder)
    output_folder = os.path.abspath(args.out_folder)
    difficulty = int(args.difficulty)

    input_files = glob.glob(os.path.join(glob.escape(input_folder),"*.mat"))
    if input_files == []:
        raise Exception(f"No input files found, looking in folder '{input_folder}' for files with extension '.mat'")


    ###########################################################
    # CONFIGURATION
    # Image size
    im_size = 512

    # Upper bound mask
    ub_val = args.ub_val # acrylic_attenuation in unit 1/mm
    ub_mask_type = args.ub_mask_type   # 1 basic 0.97 circle. 2 fitted
    basic_mask_radius = 0.97


    # Lower bound mask
    lb_mask_type = args.lb_mask_type  # 0:  lower bound 0 everywhere, 1: outer annulus equal to upper bound acrylic
    lb_inner_radius = 200
    lb_val = args.lb_val  # could be changed to 0.04 or other smaller values

    # Reconstruction
    num_iters = args.num_iters
    # with this algo we do not change alpha with difficulty level
    alpha = args.alpha
    update_objective_interval = 100
    verbose = 1
    
    # Segmentation
    segmentation_method = args.seg_method  # 1 basic thresholding, 2 crazy

    ang_range = args.ang_range
    ang_start = args.ang_start
    omega = args.omega
    if omega is None:
        omega = 90.0/ang_range

    #####################################################

    print("Omega: ", omega, "Alpha: ", alpha, "Ang Start: ", ang_start, "Ang Range: ", ang_range)
    print("Num iterations: ", num_iters, "Segmentation Method: ", segmentation_method)
    print("Lower Bound Mask Type: ", lb_mask_type, "Lower Bound Value: ", lb_val)
    print("Upper Bound Mask Type: ", ub_mask_type, "Upper Bound Value: ", ub_val)


    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # MAIN FOR LOOP TO PROCESS EACH FILE IN THE INPUT DIRECTORY
    for input_file in input_files:
        # Load the data:
        datafull = util.load_htc2022data(input_file, dataset_name="CtDataFull")
        data = util.generate_reduced_data(datafull, ang_start, ang_range)

        # Preprocess
        data_preprocessed = preprocess(data)
        # discover angular range
        

        ig = data_preprocessed.geometry.get_ImageGeometry()
        ig.voxel_num_x = im_size
        ig.voxel_num_y = im_size

        # NOTE: the fit of the sample mask works best with the non pre-processed data
        # as the FDK reconstruction is more blurry!!!
        lb, ub = create_lb_ub(data, ig, ub_mask_type, lb_mask_type, 
                                ub_val, lb_val, basic_mask_radius, lb_inner_radius)
        
        
        # algorithmic parameters
        args = [omega, alpha, alpha_dx]
        
        # Run reconstruction
        data_recon = pdhg_rotate_isotv_anisotv(data_preprocessed, ig, lb, ub, *args, num_iters=num_iters, 
                update_objective_interval=update_objective_interval, verbose=verbose)
        
        data_segmented = segment(data_recon, segmentation_method)

        util.write_data_to_png(data_segmented, input_file, output_folder)

    print("Omega: ", omega, "Alpha: ", alpha, "Ang Start: ", ang_start, "Ang Range: ", ang_range)
    print("Num iterations: ", num_iters, "Segmentation Method: ", segmentation_method)
    print("Lower Bound Mask Type: ", lb_mask_type, "Lower Bound Value: ", lb_val)
    print("Upper Bound Mask Type: ", ub_mask_type, "Upper Bound Value: ", ub_val)

    return 0

if __name__ == '__main__':
    ret = main()
    exit(ret)