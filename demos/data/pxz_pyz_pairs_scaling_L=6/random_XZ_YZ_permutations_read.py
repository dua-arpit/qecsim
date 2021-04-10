import collections
import itertools
import numpy as np
from qecsim import paulitools as pt
import matplotlib.pyplot as plt
import qecsim
from qecsim import app
from qecsim.models.generic import PhaseFlipErrorModel,DepolarizingErrorModel,BiasedDepolarizingErrorModel,BiasedYXErrorModel
from qecsim.models.planar import PlanarCode,PlanarMPSDecoder
import random

if __name__=='__main__':

    def square(a):
        return a**2
    vsquare=np.vectorize(square)
    layout_name='planar'
    bdry_name='surface'

    sizes= range(6,7,2)
    codes_and_size = [PlanarCode(*(size,size)) for size in sizes]
    p_min,p_max=0.1,0.45
    error_probabilities=np.linspace(p_min,p_max,7)

    #export data
    # code_names=['spiral_XZ','random_XZ','random_XZ_YZ','random_XY']
    # code_names=['random_all','random_XZ_YZ']
    # # code_names=['spiral_XZ','random_XZ','random_all','random_XY']
    # code_names=['XY','CSS']

    bias_list=[100]
    code_names=['random_XZ_YZ']

    perm_rates=[1,0,0,0,0,0]
    num_realiz=40
    bias_str='Z'
    max_runs=500
    legends=[]
    pXZ_list=[0,1/4,1/3,2/5,1/2]
    pYZ_list=[0,1/4,1/3,2/5,1/2]

    for L_index,code in enumerate(codes_and_size):
        for bias in bias_list:
            from itertools import cycle
            
            NUM_COLORS =200
            cm = plt.get_cmap('gist_rainbow')
            fig = plt.figure(figsize=(20,10))
            ax = fig.add_subplot(111)
            ax.set_prop_cycle('color', [cm(5.*i/NUM_COLORS) for i in range(NUM_COLORS)])

            # lines=['-',':','--','-.']
            # linecycler=cycle(lines)
            plt.title('TND failure rate scaling comparison at bias='+str(bias)[:7]+' for '+layout_name+' '+bdry_name+'L='+str(sizes[L_index]))

            #XYZ,ZYX,XZY,YXZ,YZX,ZXY
            for code_name in code_names:
                for pXZ in pXZ_list:
                    for pYZ in pXZ_list:
                        if (pXZ==0 and pYZ!=1/2) or (pYZ==0 and pXZ!=1/2):
                            continue
                        perm_rates=[1-pXZ-pYZ,pXZ,pYZ,0,0,0]

                        error_model = BiasedDepolarizingErrorModel(bias,bias_str)
                        # bias=1/bias
                        # error_model=BiasedYXErrorModel(bias)
                        chi_val=12
                        error_probabilities=np.loadtxt('p_list'+code_name+'pXZ,pYZ='+str(pXZ)[:4]+','+str(pYZ)[:4]+str(bias)[:7]+'.csv',delimiter=',')
                        pL_list=np.loadtxt('pL_list'+code_name+'pXZ,pYZ='+str(pXZ)[:4]+','+str(pYZ)[:4]+str(bias)[:7]+'.csv',delimiter=',')
                        std_list=np.loadtxt('std_list'+code_name+'pXZ,pYZ='+str(pXZ)[:4]+','+str(pYZ)[:4]+str(bias)[:7]+'.csv',delimiter=',')
                        log_pL_list,log_std_list=pL_list,std_list
                        for i in range(len(pL_list)):
                            log_pL_list[i]=-np.log(pL_list[i])
                            log_std_list[i]=std_list[i]/(pL_list[i]*np.log(10))

                        # ax.errorbar(-np.log(error_probabilities),log_pL_list,log_std_list)

                        r = random.random()
                        b = random.random()
                        g = random.random()
                        color = (r, g, b)
                        plt.errorbar(-np.log(error_probabilities),log_pL_list,log_std_list,c=color)

                        legends.append('pXZ,pYZ='+str(pXZ)[:4]+','+str(pYZ)[:4])

            plt.xlabel('-log(p)')
            plt.ylabel('$-log(p_L)$')
            plt.legend(legends) 
            plt.savefig('code_comparison_XZ_YZ_bias='+str(bias)[:7]+'.pdf')


