import collections
import itertools
import numpy as np
from qecsim import paulitools as pt
import matplotlib.pyplot as plt
import qecsim
from qecsim import app
from qecsim.models.generic import PhaseFlipErrorModel,DepolarizingErrorModel,BiasedDepolarizingErrorModel,BiasedYXErrorModel
from qecsim.models.planar import PlanarCode,PlanarMPSDecoder
import importlib as imp
import os,time
import multiprocessing as mp
from functools import partial



if __name__=='__main__':
    
    sizes= range(8,9,2)
    codes_and_size = [PlanarCode(*(size,size)) for size in sizes]
    bias_list=[10]
    layout_name='planar'
    bdry_name='surface'

    p_min,p_max=0.01,0.40
    error_probabilities=np.linspace(p_min,p_max,40)
    pL_list=np.zeros(len(error_probabilities))
    std_list=np.zeros(len(error_probabilities))
    log_pL_list=np.zeros(len(error_probabilities))
    log_std_list=np.zeros(len(error_probabilities))
    
    code_names=['random_XY','XY','CSS','XZZX','spiral_XZ','random_XZ','random_all']
    perm_rates=[0,0,0,0,0,0]
    chi=12
    bias_str='Z'

    for L_index,code in enumerate(codes_and_size):
        for bias in bias_list:
            from itertools import cycle
            plt.figure(figsize=(20,10))
            lines=["-",":","--","-."]
            linecycler=cycle(lines)
            plt.title('TND failure rate scaling comparison at '+bias_str+' bias='+str(bias)[:7]+' for '+layout_name+' '+bdry_name+',L='+str(sizes[L_index])+',chi='+str(chi))

            for code_name in code_names:
                error_probabilities=np.loadtxt("p_list"+code_name+str(bias)[:7]+".csv",delimiter=",")
                pL_list=np.loadtxt("pL_list"+code_name+str(bias)[:7]+".csv",delimiter=",")
                std_list=np.loadtxt("std_list"+code_name+str(bias)[:7]+".csv",delimiter=",")

                for i in range(len(pL_list)):
                    log_pL_list[i]=-np.log(pL_list[i])
                    log_std_list[i]=std_list[i]/(pL_list[i]*np.log(10))

                plt.errorbar(-np.log(error_probabilities),log_pL_list,log_std_list)

            plt.xlabel('$-log(p)$')
            plt.ylabel('$-log(p_L)$')
            plt.legend(code_names) 
            plt.savefig("scaling_code_comparison_bias="+str(bias)[:7]+".pdf")


