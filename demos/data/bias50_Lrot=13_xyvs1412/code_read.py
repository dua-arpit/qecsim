import collections
import itertools
import numpy as np
from qecsim import paulitools as pt
import matplotlib.pyplot as plt
import qecsim
from qecsim import app
from qecsim.models.generic import PhaseFlipErrorModel,DepolarizingErrorModel,BiasedDepolarizingErrorModel,BiasedYXErrorModel
from qecsim.models.planar import PlanarCode,PlanarMPSDecoder
from qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarMPSDecoder


if __name__=='__main__':

    def square(a):
        return a**2
    vsquare=np.vectorize(square)
    layout='rotated'
    bdry_name='surface'

    sizes= range(6,7,2)
    # codes_and_size = [PlanarCode(*(size,size)) for size in sizes]

    rotsizes= range(13,14,2)
    codes_and_size = [RotatedPlanarCode(*(size,size)) for size in rotsizes]
    p_min,p_max=0.01,0.50
    error_probabilities=np.linspace(p_min,p_max,8)

    #export data
    # code_names=['spiral_XZ','random_XZ','random_XZ_YZ','random_XY']
    # code_names=['random_all','random_XZ_YZ']
    # # code_names=['spiral_XZ','random_XZ','random_all','random_XY']
    # code_names=['XY','CSS']

    bias_list=[50]

    code_names=['XY','random_XZ_YZ4','random_XZ_YZ3','random_XZ_YZ2','random_XZ_YZ1','random_XZ_YZ0','rotXY']
    
    code_names=['random_rot_XZ_YZ','rotXY']

    perm_rates=[1,0,0,0,0,0]
    chival=12

    for L_index,code in enumerate(codes_and_size):
        for bias in bias_list:
            from itertools import cycle
            plt.figure(figsize=(20,10))
            lines=['-',':','--','-.']
            linecycler=cycle(lines)
            plt.title('TND failure rate scaling comparison at bias='+str(bias)[:7]+' for '+layout+' '+bdry_name+', L='+str(rotsizes[L_index])+', chi='+str(chival))

            #XYZ,ZYX,XZY,YXZ,YZX,ZXY
            for code_name in code_names:                                   
                error_probabilities=np.loadtxt("p_list"+code_name+str(bias)[:7]+".csv",delimiter=",")
                pL_list=np.loadtxt("pL_list"+code_name+str(bias)[:7]+".csv",delimiter=",")
                std_list=np.loadtxt("std_list"+code_name+str(bias)[:7]+".csv",delimiter=",")
                log_pL_list,log_std_list=pL_list,std_list
                for i in range(len(pL_list)):
                    log_pL_list[i]=-np.log(pL_list[i])
                    log_std_list[i]=std_list[i]/(pL_list[i]*np.log(10))
                plt.errorbar(-np.log(error_probabilities),log_pL_list,log_std_list)

            plt.xlabel('-log(p)')
            plt.ylabel('$-log(p_L)$')
            plt.legend(code_names) 
            plt.savefig('scaling_code_comparison_bias='+str(bias)[:7]+'.pdf')


