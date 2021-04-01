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

    def square(a):
        return a**2
    vsquare=np.vectorize(square)
    layout_name="planar"
    bdry_name='surface'

    sizes= range(10,11,2)
    codes_and_size = [PlanarCode(*(size,size)) for size in sizes]
    bias_list=[10]
    p_min,p_max=0.01,0.40
    error_probabilities=np.linspace(p_min,p_max,40)
    code_names=['random_XY','XY','CSS','XZZX','XYYX','spiral_XZ','spiral_XY','random_XZ']
    perm_rates=[0,0,0,0,0,0]
    pL_list=np.zeros(len(error_probabilities))
    std_list=np.zeros(len(error_probabilities))
    log_pL_list=np.zeros(len(error_probabilities))
    log_std_list=np.zeros(len(error_probabilities))

    for L_index,code in enumerate(codes_and_size):
        for bias in bias_list:
            from itertools import cycle
            plt.figure(figsize=(20,10))
            lines=["-",":","--","-."]
            linecycler=cycle(lines)
            plt.title('TND failure rate scaling comparison at bias='+str(bias)[:7]+' for '+layout_name+' '+bdry_name)

            for code_name in code_names:
                if code_name=='CSS':
                    num_realiz=1
                    bias_str='Z'
                    max_runs=10000
                elif code_name=='XY':
                    bias_str='Y'
                    num_realiz=1
                    max_runs=10000
                elif code_name=='XZZX':
                    num_realiz=1
                    bias_str='Z'
                    max_runs=10000
                elif code_name=='XYYX':
                    num_realiz=1
                    bias_str='Y'
                    max_runs=10000
                elif code_name=='spiral_XZ':
                    num_realiz=1
                    bias_str='Z'
                    max_runs=10000
                elif code_name=='spiral_XY':
                    num_realiz=1
                    bias_str='Y'
                    max_runs=10000
                elif code_name=='random_all':
                    num_realiz=30
                    bias_str='Z'
                    max_runs=2000
                    perm_rates=[1/6,1/6,1/6,1/6,1/6,1/6]
                elif code_name=='random_XY':
                    num_realiz=30
                    bias_str='Y'
                    max_runs=2000
                    perm_rates=[1/2,1/2,0,0,0,0]
                elif code_name=='random_XZ':
                    num_realiz=30
                    bias_str='Z'
                    max_runs=2000
                    perm_rates=[1/2,1/2,0,0,0,0]

                error_model = BiasedDepolarizingErrorModel(bias,bias_str)
                chi_val=10

                error_probabilities=np.loadtxt("p_list"+code_name+str(bias)[:7]+".csv",delimiter=",")
                pL_list=np.loadtxt("pL_list"+code_name+str(bias)[:7]+".csv",delimiter=",")
                std_list=np.loadtxt("std_list"+code_name+str(bias)[:7]+".csv",delimiter=",")

                for i in range(len(pL_list)):
                    log_pL_list[i]=-np.log(pL_list[i])
                    log_std_list[i]=std_list[i]/(pL_list[i]*np.log(10))

                plt.errorbar(-np.log(error_probabilities),log_pL_list,log_std_list)

            plt.xlabel('-log(p)')
            plt.ylabel('$-log(p_L)$')
            plt.legend(code_names) 
            plt.savefig("scaling_code_comparison_bias="+str(bias)[:7]+".pdf")


