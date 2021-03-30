import collections
import itertools
import numpy as np
from qecsim import paulitools as pt
import matplotlib.pyplot as plt
import qecsim
from qecsim import app
from qecsim.models.generic import PhaseFlipErrorModel,DepolarizingErrorModel,BiasedDepolarizingErrorModel
#from qecsim.models.planar import PlanarCode, PlanarMPSDecoder
from qecsim.models.rotatedplanar import RotatedPlanarCode

from _planarmpsdecoder_def import PlanarMPSDecoder_def
from _rotatedplanarmpsdecoder_def import RotatedPlanarMPSDecoder_def
import app_def
import _rotatedplanarmpsdecoder_def
import importlib as imp
imp.reload(app_def)
imp.reload(_rotatedplanarmpsdecoder_def)
import os, time
import multiprocessing as mp
from functools import partial
from random import randint

from sklearn.utils.random import sample_without_replacement

def random_coords(dims, nsamp):
    idx = sample_without_replacement(np.prod(dims), nsamp)
    return np.vstack(np.unravel_index(idx, dims)).T


def parallel_step_p(code,hadamard_mat,hadamard_vec,XYperm_mat,XYperm_vec,ZYperm_mat,ZYperm_vec,error_model, decoder, max_runs, error_probability):
    hadamard_mat,hadamard_vec,XYperm_mat,XYperm_vec,ZYperm_mat,ZYperm_vec= deform_matsvecs(code,decoder,error_model)
    result= app_def.run_def(code,hadamard_mat,hadamard_vec,XYperm_mat,XYperm_vec,ZYperm_mat,ZYperm_vec, error_model, decoder, error_probability, max_runs)
    return result

def square(a):
    return a**2

vsquare=np.vectorize(square)


# set models
sizes= range(6,11,2) #choose odd sizes since we have defined hadamard_mat for odd sizes
codes_and_size = [RotatedPlanarCode(*(size,size)) for size in sizes]
bias_list=[10000000000]
biasstr_list=['Z']

layout_name="rot"
code_name="CSS"
code_name="XZZX"
code_name="optimal"
code_name="random"

if (code_name=="random"):
    realizations=60
else:
    realizations=1

# set physical error probabilities
error_probability_min, error_probability_max = 0.05, 0.5
error_probabilities = np.linspace(error_probability_min, error_probability_max, 18)
# set max_runs for each probability



def deform_matsvecs(code,decoder,error_model):

    hadamard_mat=np.zeros((sizes[code_index],sizes[code_index]))
    XYperm_mat,ZYperm_mat=hadamard_mat,hadamard_mat
    nrows, ncols=hadamard_mat.shape
    hadamard_vec=np.zeros(np.prod((nrows,ncols)))

    for col, row in np.ndindex(nrows,ncols):
        if(np.random.rand(1,1))<pH:
            hadamard_mat[col,row]=1

    for col,row in np.ndindex(nrows,ncols):
        hadamard_vec[(col+row*ncols)]=hadamard_mat[col,row]

    XYperm_vec=np.zeros(np.prod((nrows,ncols)))
    for col, row in np.ndindex(nrows,ncols):
        if(np.random.rand(1,1))<pXY:
            XYperm_mat[col,row]=1

    for col, row in np.ndindex(nrows,ncols):
        XYperm_vec[(col+row*ncols)]=XYperm_mat[col,row]

    ZYperm_vec=np.zeros(np.prod((nrows,ncols)))
    for col, row in np.ndindex(nrows,ncols):
        if(np.random.rand(1,1))<pZY:
            ZYperm_mat[col,row]=1

    for col, row in np.ndindex(nrows,ncols):
        ZYperm_vec[(col+row*ncols)]=ZYperm_mat[col,row]
    return hadamard_mat,hadamard_vec,XYperm_mat,XYperm_vec,ZYperm_mat,ZYperm_vec



for biasstr in biasstr_list:
    timestr = time.strftime("%Y%m%d-%H%M%S ")   #record current date and time
    dirname="./data/"+timestr+layout_name+code_name+'_bias='+biasstr
    os.mkdir(dirname)     

    for bias in bias_list:
        error_model = BiasedDepolarizingErrorModel(bias,biasstr)
        chi_val=10
        decoder = _rotatedplanarmpsdecoder_def.RotatedPlanarMPSDecoder_def(chi=chi_val)

        # print run parameters
        print('layout:', layout_name)
        print('codes_and_size:', [code.label for code in codes_and_size])
        print('Error model:', error_model.label)
        print('Decoder:', decoder.label)
        print('Error probabilities:', error_probabilities)
        # print('Maximum runs:', max_runs)

        pL_list_rand =np.zeros((len(codes_and_size),realizations,len(error_probabilities)))
        std_list_rand=np.zeros((len(codes_and_size),realizations,len(error_probabilities)))

        pL_list =np.zeros((len(codes_and_size),len(error_probabilities)))
        std_list=np.zeros((len(codes_and_size),len(error_probabilities)))        
        
        log_pL_list =np.zeros((len(codes_and_size),len(error_probabilities)))
        log_std_list=np.zeros((len(codes_and_size),len(error_probabilities)))
        
        def _rotate_q_index(index, code):
            """Convert code site index in format (x, y) to tensor network q-node index in format (r, c)"""
            site_x, site_y = index  # qubit index in (x, y)
            site_r, site_c = code.site_bounds[1] - site_y, site_x  # qubit index in (r, c)
            return code.site_bounds[0] - site_c + site_r, site_r + site_c  # q-node index in (r, c)

        for code_index,code in enumerate(codes_and_size):           
            if code_name=="random":
                pH=0.5 
                pXY=0
                pZY=0
                max_runs =500

            #  (0,2)-----(1,2)-----(2,2)
            #    |         |         |
            #    |         |         |
            #    |         |         |
            #  (0,1)-----(1,1)-----(2,1)
            #    |         |         |
            #    |         |         |
            #    |         |         |
            #  (0,0)-----(1,0)-----(2,0)

                for realization_index in range(realizations):

                    hadamard_mat,hadamard_vec,XYperm_mat,XYperm_vec,ZYperm_mat,ZYperm_vec= deform_matsvecs(code,decoder,error_model)

                    p=mp.Pool()
                    func=partial(parallel_step_p,code,hadamard_mat,hadamard_vec,XYperm_mat,XYperm_vec,ZYperm_mat,ZYperm_vec,error_model, decoder, max_runs)
                    result=p.map(func, error_probabilities)
                    #print(result)
                    p.close()
                    p.join()
                    for i in range(len(result)):
                        pL_list_rand[code_index][realization_index][i]=result[i][0]
                        std_list_rand[code_index][realization_index][i]=result[i][1]
                    
                pL_list[code_index] = np.sum(pL_list_rand[code_index],axis=0)/realizations
                std_list[code_index] = np.sqrt(np.sum(vsquare(std_list_rand[code_index]),axis=0))/realizations

                for i in range(len(pL_list[code_index])):
                    log_pL_list[code_index][i]=-np.log(pL_list[code_index][i])
                    log_std_list[code_index][i]=std_list[code_index][i]/(pL_list[code_index][i]*np.log(10))

            else:
                if code_name=="XZZX":
                    max_runs =1000

                    pXY, pZY=0,0
                    nrows, ncols=hadamard_mat.shape

                    for col, row in np.ndindex(nrows,ncols):
                        if (row+col)%2==0:
                            hadamard_mat[col,row]=1

                    hadamard_vec=np.zeros(np.prod((nrows,ncols)))
                    for col, row in np.ndindex(nrows,ncols):
                        hadamard_vec[(col+row*ncols)]=hadamard_mat[col,row]
                    
                    # for i in range(np.prod(hadamard_mat.shape)):
                    #     if i%2==0:
                    #         hadamard_vec[i]=1

                    rand_XY_coords= random_coords(XYperm_mat.shape,round(np.prod(XYperm_mat.shape)*pXY))
                    for col, row in rand_XY_coords:
                        XYperm_mat[col,row]=0
                    XYperm_vec=np.zeros(np.prod(XYperm_mat.shape))
                    for i,j in np.ndindex(XYperm_mat.shape):
                        XYperm_vec[(i+j*XYperm_mat.shape[1])]=XYperm_mat[i,j]

                    rand_ZY_coords= random_coords(ZYperm_mat.shape,round(np.prod(ZYperm_mat.shape)*pZY))
                    for col, row in rand_ZY_coords:
                        ZYperm_mat[col,row]=0
                    ZYperm_vec=np.zeros(np.prod(ZYperm_mat.shape))
                    for i,j in np.ndindex(ZYperm_mat.shape):
                        ZYperm_vec[(i+j*ZYperm_mat.shape[1])]=ZYperm_mat[i,j]


                if code_name=="optimal": #for odd dy
                    max_runs =2000
                    pXY, pZY=0,0

                    nrows, ncols=hadamard_mat.shape

                    for col, row in np.ndindex(hadamard_mat.shape):
                        if row%2==0:
                            if row%4==0:
                                hadamard_mat[row,range(0,ncols-1)]=1
                            else:
                                hadamard_mat[row,range(1,ncols)]=1
                        else:
                            if row%4==1:
                                hadamard_mat[row,ncols-1]=1
                            elif row%4==3:
                                hadamard_mat[row,0]=1

                    hadamard_vec=np.zeros(np.prod((nrows,ncols)))
                    for col, row in np.ndindex(nrows,ncols):
                        hadamard_vec[(col+row*ncols)]=hadamard_mat[col,row]


                    rand_XY_coords= random_coords(XYperm_mat.shape,round(np.prod(XYperm_mat.shape)*pXY))
                    for col, row in rand_XY_coords:
                        XYperm_mat[col,row]=0
                    XYperm_vec=np.zeros(np.prod(XYperm_mat.shape))
                    for i,j in np.ndindex(XYperm_mat.shape):
                        XYperm_vec[(i+j*XYperm_mat.shape[1])]=XYperm_mat[i,j]

                    rand_ZY_coords= random_coords(ZYperm_mat.shape,round(np.prod(ZYperm_mat.shape)*pZY))
                    for col, row in rand_ZY_coords:
                        ZYperm_mat[col,row]=0
                    ZYperm_vec=np.zeros(np.prod(ZYperm_mat.shape))
                    for i,j in np.ndindex(ZYperm_mat.shape):
                        ZYperm_vec[(i+j*ZYperm_mat.shape[1])]=ZYperm_mat[i,j]

                # hadamard_vec=np.zeros(np.prod(hadamard_mat.shape))

                # for i,j in np.ndindex(hadamard_mat.shape):
                #     if hadamard_mat[i,j]==1:
                #         hadamard_vec[(i+j*hadamard_mat.shape[1])]=1

                p=mp.Pool()
                func=partial(parallel_step_p,code,hadamard_mat,hadamard_vec,XYperm_mat,XYperm_vec,ZYperm_mat,ZYperm_vec,error_model, decoder, max_runs)
                result=p.map(func, error_probabilities)
                print(result)
                p.close()
                p.join()
                
                for i in range(len(result)):
                    pL_list[code_index][i]=result[i][0]   
                    std_list[code_index][i]=result[i][1]
                    log_pL_list[code_index][i]=-np.log(pL_list[code_index][i])
                    log_std_list[code_index][i]=std_list[code_index][i]/(pL_list[code_index][i]*np.log(10))

            
        from itertools import cycle
        plt.figure(figsize=(20,10))
        lines = ["-",":","--","-."]
        linecycler = cycle(lines)
        plt.title('TND at bias='+str(bias)+' and xi='+str(chi_val))
        for sizes_index,size in enumerate(sizes):
            plt.errorbar(error_probabilities,pL_list[sizes_index],std_list[sizes_index])
        plt.xlabel('p')
        plt.ylabel('$p_L$')
        plt.legend(sizes) 
        plt.savefig(dirname+"/threshold_plot_bias_"+str(bias)+".pdf")

        from itertools import cycle
        plt.figure(figsize=(20,10))
        lines = ["-",":","--","-."]
        linecycler = cycle(lines)
        plt.title('TND failure rate scaling at bias='+str(bias)+' and xi='+str(chi_val))
        for sizes_index,size in enumerate(sizes):
            plt.errorbar(error_probabilities,log_pL_list[sizes_index],log_std_list[sizes_index])
        plt.xlabel('p')
        plt.ylabel('$-log(p_L)$')
        plt.legend(sizes) 
        plt.savefig(dirname+"/log_scaling_plot_bias_"+str(bias)+".pdf")
