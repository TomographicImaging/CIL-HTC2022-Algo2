#   Copyright 2022 United Kingdom Research and Innovation
#   Copyright 2022 Technical University of Denmark
#   Copyright 2022 Finden   
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
from cil.optimisation.functions import Function
from cil.optimisation.operators import GradientOperator, BlockOperator, IdentityOperator, FiniteDifferenceOperator
from cil.optimisation.functions import L2NormSquared, MixedL21Norm, L1Norm, BlockFunction, IndicatorBox
from cil.optimisation.algorithms import PDHG, SPDHG
from cil.framework import AcquisitionData
from cil.plugins.tigre import ProjectionOperator
from skimage.transform import rotate    

class IndicatorBoxPixelwise(Function):
    def __init__(self,lower,upper):
        '''creator
        :param lower: lower bound
        :type lower: float, default = :code:`-numpy.inf`
        :param upper: upper bound
        :type upper: float, optional, default = :code:`numpy.inf`
        '''
        super(IndicatorBoxPixelwise, self).__init__()
        self.lower = lower
        self.upper = upper
    
    def __call__(self,x):
        '''Evaluates IndicatorBox at x'''
        if (np.all(x.as_array() >= self.lower.as_array()) and 
            np.all(x.as_array() <= self.upper.as_array()) ):
            val = 0
        else:
            val = np.inf
        return val
    
    def proximal(self, x, tau, out=None):
        r'''Proximal operator of IndicatorBox at x
            .. math:: prox_{\tau * f}(x)
        '''
        if out is None:
            return (x.maximum(self.lower)).minimum(self.upper)        
        else:               
            x.maximum(self.lower, out=out)
            out.minimum(self.upper, out=out) 

    def convex_conjugate(self,x):
        '''Convex conjugate of IndicatorBox at x'''
        return x.maximum(0).sum()


def pdhg_tv(data, ig, lb, ub, *args, num_iters=100, update_objective_interval=100, verbose=1):
    omega = args[0]
    alpha = args[1]
    
    Grad = GradientOperator(ig)
    A = ProjectionOperator(ig, data.geometry)
    Id = IdentityOperator(ig)
    K = BlockOperator(A, Grad)

    f1 = omega*L2NormSquared(b=data)
    f2 = alpha*MixedL21Norm()
    F = BlockFunction(f1, f2)
    
    G = IndicatorBoxPixelwise(lower=lb, upper=ub)

    normK = K.norm()
    sigma = 1.0
    tau = 1.0/(sigma*normK**2)

    algo = PDHG(initial=ig.allocate(0.0),
                f=F, g=G, operator=K, 
                sigma=sigma, tau=tau,
                update_objective_interval=100, 
                max_iteration=num_iters)
    algo.run(verbose=verbose)
    return algo.solution.copy()


def pdhg_rotate_isotv_anisotv(data, ig, lb, ub, *args, num_iters=100, update_objective_interval=100, verbose=1):
    
    ag_rotated = data.geometry.copy()
    
    ang_middle = (data.geometry.angles[0]+data.geometry.angles[-1])/2
    #ang_start = np.abs(data.geometry.angles[0])
    #ang_range = np.abs(data.geometry.angles[-1]-data.geometry.angles[0])
    #ang_rotate = (ang_start+ang_range/2.0)
    ag_rotated.set_angles(data.geometry.angles - ang_middle, angle_unit='degree')
    
    data = AcquisitionData(data.as_array(), geometry=ag_rotated)
    
    omega = args[0]
    alpha = args[1]
    alpha_dx = args[2]
    
    Grad = GradientOperator(ig)
    A = ProjectionOperator(ig, data.geometry)
    Dx = FiniteDifferenceOperator(ig, direction='horizontal_x')
    K12x = BlockOperator(A, Grad, Dx)

    f1 = omega*L2NormSquared(b=data)
    f2 = alpha*MixedL21Norm()
    f_dx = alpha_dx*L1Norm()
    F12x = BlockFunction(f1, f2, f_dx)
    
    lb_copy = lb.copy()
    ub_copy = ub.copy()
    lb_copy.array = rotate(lb.as_array(), -ang_middle)
    ub_copy.array = rotate(ub.as_array(), -ang_middle)

    G = IndicatorBoxPixelwise(lower=lb_copy, upper=ub_copy)

    normK = K12x.norm()
    sigma = 1.0
    tau = 1.0/(sigma*normK**2)

    algo = PDHG(initial=ig.allocate(0.0),
                f=F12x, g=G, operator=K12x, 
                sigma=sigma, tau=tau,
                update_objective_interval=100, 
                max_iteration=num_iters)
    algo.run(verbose=verbose)
    
    sol =  algo.solution.copy()
    sol.array = rotate(sol.as_array(), ang_middle)
    return sol



def pdhg_l1tvl1(data, ig, lb, ub, *args, num_iters=2000, update_objective_interval=100, verbose=1):
    omega = args[0]
    alpha = args[1]
    beta = args[2]
    
    Grad = GradientOperator(ig)
    A = ProjectionOperator(ig, data.geometry)
    Id = IdentityOperator(ig)
    K = BlockOperator(A, Grad, Id)
    
    f1 = omega*L1Norm(b=data)
    f2 = alpha*MixedL21Norm()
    f3 = beta*L1Norm()
    F = BlockFunction(f1, f2, f3)
    G = IndicatorBoxPixelwise(lower=lb, upper=ub)
    
    normK = K.norm()
    sigma = 1.0
    tau = 1.0/(sigma*normK**2)
    algo = PDHG(initial=ig.allocate(0.0),
        f=F, g=G, operator=K,
        sigma=sigma, tau=tau,
        update_objective_interval=100, max_iteration=num_iters)
    algo.run(verbose=verbose)
    return algo.solution.copy()


def TV_iso_and_aniso_PDHG(preproc_data, fidelity_weight=10, 
                          iso_weight = 1.0,
                          aniso_weight_y = 1.0,
                          aniso_weight_x = 1.0, lower = 0.0, upper = 0.04, init_recon = None,
                          max_iterations = 1000, update_objective_interval = 100, verbose=1, imsize=None):
        
    # image geometry
    ig = preproc_data.geometry.get_ImageGeometry()
    
    if imsize is not None:
        ig.voxel_num_x = imsize
        ig.voxel_num_y = imsize    
    
    if init_recon is None:
        init_recon = ig.allocate()    
    
    # FinDiff operators in y, x (numpy)
    DY = FiniteDifferenceOperator(ig, direction=0)
    DX = FiniteDifferenceOperator(ig, direction=1)
    
    # GradOperar with c backend
    Grad = GradientOperator(ig)
    
    
    # PDHG operator
    A = ProjectionOperator(ig, preproc_data.geometry)
    K = BlockOperator(A, DY, DX, Grad)
    
    # PDHG composite part
    f1 = (fidelity_weight/2)*L2NormSquared(b=preproc_data)
    f2 = aniso_weight_y*L1Norm() #0.05
    f3 = aniso_weight_x*L1Norm()
    f4 = -iso_weight * MixedL21Norm()
    F = BlockFunction(f1, f2, f3, f4)
    
    # PDHG no composite part
    #G = IndicatorBox(lower=lower, upper=upper)
    G = IndicatorBoxPixelwise(lower=lower, upper=upper)

    normK = K.norm()
    
    sigma = 0.1
    tau = 1./(sigma*normK**2)
    
    pdhg_anis_iso = PDHG(initial=init_recon,f=F, g=G, operator=K, 
                update_objective_interval=update_objective_interval,
                sigma=sigma, tau=tau,
               max_iteration=max_iterations)
    pdhg_anis_iso.run(verbose=verbose)    
    
    return pdhg_anis_iso.solution.copy()
