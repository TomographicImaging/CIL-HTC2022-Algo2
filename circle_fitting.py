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
    smaller_circle[0] = smaller_circle[0] - 1
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
import matplotlib
Nr = 1
Nc = 4
origin = 'lower'
fig, axs = plt.subplots(Nr, Nc)
n = 2
plt.figure(figsize=tuple([n * el for el in (10,6)]),\
    dpi=300)
for i,el in enumerate(out[1]):
    a = np.asarray(el, dtype=np.float32)
    # add disk with points removed 
    innercircle = ig.allocate(0)
    innercircle_par = out[0][i].copy()
    innercircle_par[0] -= 4
    util.fill_circular_mask(innercircle_par, innercircle.as_array(), 
                            1, *innercircle.shape)
    
    # make a single row of plot
    axs[i].imshow(circ[i].as_array(), cmap='seismic', vmin=-1, vmax=1, origin=origin)
    axs[i].imshow(a, cmap='seismic_r', vmin=-1, vmax=1, alpha=0.5, origin=origin)
    axs[i].imshow(recfull.as_array(), cmap='gray_r', alpha=0.1, origin=origin)
    axs[i].imshow(innercircle.as_array(), cmap='gray_r', alpha=0.05, origin=origin)
    axs[i].title.set_text(f'iteration {i}')
    if i > 0:
        axs[i].yaxis.set_major_locator(matplotlib.ticker.NullLocator())
#%%
# plt.show()
# fig2.set_facecolor('w')
# fig2.savefig('prova.png', dpi='figure')

#%%
#%%
print([n * el for el in (10,6)])
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
# dfig.save('circle_fitting.png', dpi=600)
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

#%%

from PIL import Image
def calculate_score(path_to_reference_image_png, img_data):

    # img_data = Image.open(path_to_our_segmentation)
    img_arr = np.array(img_data)

    ref = util.loadImg(path_to_reference_image_png)

    return util.calcScoreArray(img_arr, np.flipud(ref))  

#%%
mask_scores = []
for el in ['ta', 'tb', 'tc', 'td']:

    data_tmp = util.load_htc2022data(f'../HTC2022/data/htc2022_{el}_full.mat')

    out = util.find_circle_parameters_step(data, ig)
    # print (out)
    # mask = get_circle_mask(out[0][-1], ig)
    mask = ig.allocate(0)
    util.fill_circular_mask(out[0][-1], mask.as_array(), 1, *mask.shape)
        
    show2D(mask)

    segmentation = f"../HTC2022/data/segmented_references/htc2022_{el}_full_recon_fbp_seg.png"
    ref = np.flipud(util.loadImg(segmentation))
    show2D([ref, mask])
    score_mask = util.calcScoreArray(mask.as_array(), ref)
    # score_mask = calculate_score(segmentation, mask)
    print (score_mask)
    mask_scores.append(score_mask)
# %%

for x,y in zip (['ta', 'tb', 'tc', 'td'], mask_scores):
    print (f'{x}: {y}')

# %%
#97% 
# np.asarray([r,x0,y0])
r97 = ig.voxel_num_x * 0.5 * 0.97
x97 = ig.voxel_num_x * 0.5
y97 = ig.voxel_num_y * 0.5

mask_scores_97 = []
for el in ['ta', 'tb', 'tc', 'td']:

    
    # print (out)
    # mask = get_circle_mask(out[0][-1], ig)
    mask = ig.allocate(0)
    util.fill_circular_mask([r97,x97,y97], mask.as_array(), 1, *mask.shape)
        
    show2D(mask)

    segmentation = f"../HTC2022/data/segmented_references/htc2022_{el}_full_recon_fbp_seg.png"
    ref = np.flipud(util.loadImg(segmentation))
    show2D([ref, mask])
    score_mask = util.calcScoreArray(mask.as_array(), ref)
    # score_mask = calculate_score(segmentation, mask)
    print (score_mask)
    mask_scores_97.append(score_mask)
# %%

for x,y in zip (['ta', 'tb', 'tc', 'td'], mask_scores_97):
    print (f'{x}: {y}')
# %%
avg = 0
for x,y in zip(mask_scores, mask_scores_97):
    print (x-y)
    avg += (x-y)
print (avg/4)
# %%
