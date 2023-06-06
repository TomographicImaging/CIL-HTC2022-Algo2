#%%
import util
from cil.utilities.display import show2D
import matplotlib.pyplot as plt
import numpy as np
from cil.recon import FDK

data_set_identifier = 'tb' #'solid_disc' 'ta' 'tb' 'tc' 'td'
datafull = util.load_htc2022data('../HTC2022/data/htc2022_'+data_set_identifier+'_full.mat')

#%%
# Reduce data
ang_start = 0
ang_range = 30

# Image size
im_size = 512

# Upper bound mask
ub_val = 0.040859 # acrylic_attenuation in unit 1/mm
ub_mask_type = 2   # 1 basic 0.97 circle. 2 fitted
def ub_mask_str(ub_mask_type):
    if ub_mask_type ==1:
        typ = '97% radius'
    elif ub_mask_type == 2:
        typ = 'fitted'
    elif ub_mask_type is None:
        typ = 'none'
    return typ

basic_mask_radius = 0.97

# Lower bound mask
lb_mask_type = 0   # 0:  lower bound 0 everywhere, 1: outer annulus equal to upper bound acrylic
lb_inner_radius = 200
lb_val = 0

# Reconstruction
num_iters = 2000

# Segment
segment_type = 2  # 1 basic thresholding, 2 crazy

#%%
# Reduce the data
data = util.generate_reduced_data(datafull, ang_start, ang_range)
# data = util.load_htc2022data(os.path.abspath("C:/Users/ofn77899/Data/HTC2022/test_input/A.mat"), dataset_name="CtDataLimited")
ang_range = np.abs(data.geometry.angles[-1]-data.geometry.angles[0])
#%%
# Create the geometry
ig = data.geometry.get_ImageGeometry()
ig.voxel_num_x = im_size
ig.voxel_num_y = im_size

#%%
# make the full FDK reconstruction
recfull = FDK(datafull, ig).run()
#%%
out = util.find_circle_parameters_step(data, ig)
print (out)
#%%
# Calculate FDK 
recon = FDK(data, ig).run()
# calculate gradient magnitude    
mag = util.calculate_gradient_magnitude(recon)
#%%
show2D([recon, mag], title=['FDK', 'Gradient magnitude'], cmap=['gray', 'gray_r'])
#%%
# create the circumference mask
def get_circle_mask(circle_parameters, ig):
    outercircle = ig.allocate(0)
    util.fill_circular_mask(circle_parameters, outercircle.as_array(), 1, *outercircle.shape)
    # fill the circle with 0 in a smaller circle to highlight the circumference
    smaller_circle = circle_parameters.copy()
    smaller_circle[0] = smaller_circle[0] - 2
    print (circle_parameters, smaller_circle)
    innercircle = ig.allocate(0)
    util.fill_circular_mask(smaller_circle, innercircle.as_array(), 1, *innercircle.shape)

    dcircle = outercircle - innercircle
    return dcircle

circ = []
for i,el in enumerate(out[0]):
    circ.append(get_circle_mask(el, ig))

# both = circ[0] + circ[1]

show2D(circ, cmap = 'gray_r') 
#%%
for i,el in enumerate(out[1]):
    a = np.asarray(el, dtype=np.float32)
    # add disk with points removed 
    # or difference in colors with points used and removed
    plt.imshow(circ[i].as_array(), cmap='seismic', vmin=-1, vmax=1)
    plt.imshow(a, cmap='seismic_r', vmin=-1, vmax=1, alpha=0.5)
    plt.imshow(recfull.as_array(), cmap='gray_r', alpha=0.1)
    plt.show()

#%%
# preprocess the data
data_renorm = util.correct_normalisation(data)
data_pad = util.pad_zeros(data_renorm)
data_BHC = util.apply_BHC(data_pad)

out = util.find_circle_parameters_step(data_BHC, ig)
print (out)
circ = []
for i,el in enumerate(out):
    circ.append(get_circle_mask(out[0][i], ig))

both = circ[0] + circ[1]

show2D(circ + [both], title=['start', 'final', 'both'], cmap = 'gray_r') 
#%%
# Circle fitting
circle_parameters = util.find_circle_parameters(data, ig)

# Display something
dcircle = get_circle_mask(circle_parameters, ig)
show2D(dcircle, title='Circle fitting', cmap='gray')
#%%


configs = {
    'LS+isoTV':{'iso_weight':0.01, 'aniso_weight_x': 0.03 * 1e-7, 'ub_mask_type': None, 'submitted_algo':False, 'step_size_submitted':True},
    'LS+isoTV+disk':{'iso_weight':0.01, 'aniso_weight_x': 0.03 * 1e-7, 'ub_mask_type': 2, 'submitted_algo':False, 'step_size_submitted':True}, 
    'LS+isoTV+xTV+disk':{'iso_weight':0.01, 'aniso_weight_x': 0.03 , 'ub_mask_type': 2, 'submitted_algo':False, 'step_size_submitted':True},
    'LS+isoTV+xTV+disk_converged':{'iso_weight':0.01, 'aniso_weight_x': 0.03 , 'ub_mask_type': 2, 'submitted_algo':False, 'step_size_submitted':False}
}

which = 'LS+isoTV+xTV+disk_converged'

iso_weight = configs[which]['iso_weight']
aniso_weight_x = configs[which]['aniso_weight_x']
ub_mask_type = configs[which]['ub_mask_type']
submitted_algo = configs[which]['submitted_algo']
step_size_submitted = configs[which]['step_size_submitted']

# Show the procedure for the circle fit
from skimage.filters import threshold_otsu, threshold_multiotsu
mag = util.calculate_gradient_magnitude(recons[1])

# initial binary mask
thresh = threshold_otsu(mag.array)
binary_mask = mag.array > thresh

disk = create_circular_mask(2, ub_val, data=data, ig=ig)


lb1, ub1 = util.create_lb_ub(data, ig, ub_mask_type, lb_mask_type, 
                                ub_val, lb_val, basic_mask_radius, lb_inner_radius)
show2D([ub, ub1, ub-ub1], cmap=['gray','gray', 'seismic'])