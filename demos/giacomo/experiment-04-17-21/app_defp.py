"""
This module contains functions to generate and merge stabilizer code simulation data.
"""
import collections
import itertools
import json
import logging
import statistics
import time
import numpy as np
from qecsim import paulitools as pt
from qecsim.error import QecsimError
from qecsim.model import DecodeResult
from random import shuffle
from qecsim.model import ErrorModel,cli_description
from qecsim import paulitools as pt

def deform_matsvecs(code,decoder,error_model,perm_rates,code_name,layout):
        
    seed_sequence = np.random.SeedSequence()
    rng = np.random.default_rng(seed_sequence) 

    if layout=='planar':
      
        error = error_model.generate(code,0.3,rng)
        syndrome = pt.bsp(error,code.stabilizers.T)
        sample_pauli = decoder.sample_recovery(code,syndrome)
        perm_mat_sample=np.zeros((2*sample_pauli.code.size[0] - 1,2*sample_pauli.code.size[1] - 1),dtype=int)
        perm_mat=np.zeros(perm_mat_sample.shape,dtype=int)
        n_qubits =code.n_k_d[0]
        perm_vec=[]
        if code_name[:6]=='random':
            # print(perm_rates)
            for row,col in np.ndindex(perm_mat.shape):
                if (row%2==0 and col%2==0):
                    x=rng.choice((0,1,2,3,4,5),size=1,p=perm_rates) 
                    perm_mat[row,col]=x[0]
                    perm_vec.append(perm_mat[row,col])

            for row,col in np.ndindex(perm_mat.shape):
                if (row%2==1 and col%2==1):
                    x=rng.choice((0,1,2,3,4,5),size=1,p=perm_rates) 
                    perm_mat[row,col]=x[0]
                    perm_vec.append(perm_mat[row,col])

        elif code_name=='XZZX':
            for row,col in np.ndindex(perm_mat.shape):
                if (row%2==0 and col%2==0):
                    perm_mat[row,col]=1
                    perm_vec.append(perm_mat[row,col])

            for row,col in np.ndindex(perm_mat.shape):
                if (row%2==1 and col%2==1):
                    perm_vec.append(perm_mat[row,col])

        elif code_name[:6]=='spiral':
            d=perm_mat.shape[0]
            for row,col in np.ndindex(perm_mat.shape):
                if (row%2==0 and col%2==0):
                    if (row ==0 and col in range(0,d,4)) or (row==d-1 and col in range(2,d-1,4)) or (row%2==1 and col%2==1):
                        perm_mat[row,col]=1
                    perm_vec.append(perm_mat[row,col])

            for row,col in np.ndindex(perm_mat.shape):
                if (row%2==1 and col%2==1):
                    if (row ==0 and col in range(0,d,4)) or (row==d-1 and col in range(2,d-1,4)) or (row%2==1 and col%2==1):
                        perm_mat[row,col]=1
                    perm_vec.append(perm_mat[row,col])

        elif code_name=='CSS' or code_name=='XY':
            for row,col in np.ndindex(perm_mat.shape):
                if (row%2==0 and col%2==0):
                    perm_vec.append(perm_mat[row,col])
            for row,col in np.ndindex(perm_mat.shape):
                if (row%2==1 and col%2==1):
                    perm_vec.append(perm_mat[row,col])

    elif layout=='rotated':

        perm_mat=np.zeros((code.site_bounds[0]+1,code.site_bounds[1]+1),dtype=int)
        nrows, ncols=perm_mat.shape
        perm_vec=np.zeros(np.prod((nrows,ncols)))

        if code_name[:6]=='random':
            # print(perm_rates)
            for row,col in np.ndindex(nrows,ncols):
                x=rng.choice((0,1,2,3,4,5),size=1,p=perm_rates) 
                perm_mat[row,col]=x[0]
                perm_vec[(row+col*ncols)]=perm_mat[row,col]

        elif code_name=='rot_XZZX':
            for row,col in np.ndindex(nrows,ncols):
                if (row+col)%2==0:
                    perm_mat[row,col]=1

            perm_vec=np.zeros(np.prod((nrows,ncols)))
            for row,col in np.ndindex(nrows,ncols):
                perm_vec[(row+col*ncols)]=perm_mat[row,col]

        elif code_name[:6]=='rot_spiral':
            for row,col in np.ndindex(nrows,ncols):
                if row%2==0:
                    if row%4==0:
                        perm_mat[row,range(0,ncols-1)]=1
                    else:
                        perm_mat[row,range(1,ncols)]=1
                else:
                    if row%4==1:
                        perm_mat[row,ncols-1]=1
                    elif row%4==3:
                        perm_mat[row,0]=1

            perm_vec=np.zeros(np.prod((nrows,ncols)))
            for row,col in np.ndindex(nrows,ncols):
                perm_vec[(row+col*ncols)]=perm_mat[row,col]

        elif code_name=='rotXZ' or code_name=='rotXY':
            for row,col in np.ndindex(nrows,ncols):
                perm_mat[col,row]=0
                perm_vec[(row+col*ncols)]=perm_mat[row,col]

    return perm_mat,perm_vec

def permute_error_Pauli(error_Pauli,perm_vec):  
#XYZ,ZYX,XZY,YXZ,YZX,ZXY
    n_qubits=len(error_Pauli)
    for i in range(n_qubits):
        #if perm_vec[i]==0: XYZ
        if perm_vec[i]==1: #ZYX
            if error_Pauli[i]=='X':
                error_Pauli[i]='Z'
            elif error_Pauli[i]=='Z':
                error_Pauli[i]='X'

        elif perm_vec[i]==2: #XZY
            if error_Pauli[i]=='Y':
                error_Pauli[i]='Z'
            elif error_Pauli[i]=='Z':
                error_Pauli[i]='Y'
    
        elif perm_vec[i]==3: #YXZ
            if error_Pauli[i]=='X':
                error_Pauli[i]='Y'
            elif error_Pauli[i]=='Y':
                error_Pauli[i]='X'

        elif perm_vec[i]==4: #XYZ->YZX Schrodinger
            if error_Pauli[i]=='X':
                error_Pauli[i]='Z'
            elif error_Pauli[i]=='Y':
                error_Pauli[i]='X'
            elif error_Pauli[i]=='Z':
                error_Pauli[i]='Y'                    

        elif perm_vec[i]==5: #XYZ->ZXY Schrodinger
            if error_Pauli[i]=='X':
                error_Pauli[i]='Y'
            elif error_Pauli[i]=='Y':
                error_Pauli[i]='Z'
            elif error_Pauli[i]=='Z':
                error_Pauli[i]='X'               

    step_error=pt.pauli_to_bsf(''.join(error_Pauli))                                        

    return step_error


logger = logging.getLogger(__name__)

def run_once_defp(code,error_model,decoder,error_probability,perm_rates,perm_mat,perm_vec,code_name,layout,rng=None):
    # validate parameters
    if not (0 <= error_probability <= 1):
        raise ValueError('Error probability must be in [0,1].')
    # defaults
    rng = np.random.default_rng() if rng is None else rng

    return _run_once_defp('ideal',code,1,error_model,decoder,error_probability,perm_rates,perm_mat,perm_vec,code_name,layout,0.0,rng)

def _run_once_defp(mode,code,time_steps,error_model,decoder,error_probability,perm_rates,perm_mat,perm_vec,code_name,layout,measurement_error_probability,rng):
    """Implements run_once and run_once_ftp functions"""
    # assumptions
    assert (mode == 'ideal' and time_steps == 1) or mode == 'ftp'
    if code_name[:6]=='random':
        perm_mat,perm_vec= deform_matsvecs(code,decoder,error_model,perm_rates,code_name,layout)

    # generate step_error,step_syndrome and step_measurement_error for each time step
    n_qubits = code.n_k_d[0]
    
    for _ in range(time_steps):
        # hadamard_mat,hadamard_vec,XYperm_mat,XYperm_vec,ZYperm_mat,ZYperm_vec= deform_matsvecs(code,decoder,error_model)
        step_errors,step_syndromes,step_measurement_errors = [],[],[]

        rng = np.random.default_rng() if rng is None else rng
        error_Pauli = rng.choice(('I','X','Y','Z'),size=n_qubits,p=error_model.probability_distribution(error_probability))
        step_error=permute_error_Pauli(error_Pauli,perm_vec)

        step_errors.append(step_error)
        # step_syndrome: stabilizers that do not commute with the error
        step_syndrome = pt.bsp(step_error,code.stabilizers.T)
        step_syndromes.append(step_syndrome)
        # step_measurement_error: random syndrome bit flips based on measurement_error_probability
        if measurement_error_probability:
            step_measurement_error = rng.choice(
                (0,1),
                size=step_syndrome.shape,
                p=(1 - measurement_error_probability,measurement_error_probability)
            )
        else:
            step_measurement_error = np.zeros(step_syndrome.shape,dtype=int)
        step_measurement_errors.append(step_measurement_error)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('run: step_errors={}'.format(step_errors))
        logger.debug('run: step_syndromes={}'.format(step_syndromes))
        logger.debug('run: step_measurement_errors={}'.format(step_measurement_errors))

    # error: sum of errors at each time step
    error = np.bitwise_xor.reduce(step_errors)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('run: error={}'.format(error))

    # syndrome: apply measurement_error at times t-1 and t to syndrome at time t
    syndrome = []
    for t in range(time_steps):
        syndrome.append(step_measurement_errors[t - 1] ^ step_syndromes[t] ^ step_measurement_errors[t])
    # convert syndrome to 2d numpy array
    syndrome = np.array(syndrome)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('run: syndrome={}'.format(syndrome))

    # decoding: boolean or best match recovery operation based on decoder
    ctx = {'error_model': error_model,'error_probability': error_probability,'error': error,
           'step_errors': step_errors,'measurement_error_probability': measurement_error_probability,
           'step_measurement_errors': step_measurement_errors}
    # convert syndrome to 1d if mode is 'ideal'
    if mode == 'ideal':  # convert syndrome to 1d and call decode
        decoding = decoder.decode(code,perm_mat,syndrome[0],**ctx)
    if mode == 'ftp':  # call decode_ftp
        decoding = decoder.decode_ftp(code,time_steps,syndrome,**ctx)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('run: decoding={}'.format(decoding))

    # if decoding is not DecodeResult,convert to DecodeResult
    if not isinstance(decoding,DecodeResult):
        # decoding is recovery,so wrap in DecodeResult
        decoding = DecodeResult(recovery=decoding)  # raises error if recovery is None
    # extract outcomes from decoding
    success = decoding.success
    logical_commutations = decoding.logical_commutations
    custom_values = decoding.custom_values
    # if recovery specified,resolve success and logical_commutations
    if decoding.recovery is not None:
        # recovered code
        recovered = decoding.recovery[1] ^ error
        max_coset_p = decoding.recovery[0]
        # success checks
        commutes_with_stabilizers = np.all(pt.bsp(recovered,code.stabilizers.T) == 0)
        if not commutes_with_stabilizers:
            log_data = {  # enough data to recreate issue
                # models
                'code': repr(code),'error_model': repr(error_model),'decoder': repr(decoder),
                # variables
                'error': pt.pack(error),'recovery': pt.pack(decoding.recovery),
                # step variables
                'step_errors': [pt.pack(v) for v in step_errors],
                'step_measurement_errors': [pt.pack(v) for v in step_measurement_errors],
            }
            logger.warning('RECOVERY DOES NOT RETURN TO CODESPACE: {}'.format(json.dumps(log_data,sort_keys=True)))
        resolved_logical_commutations = pt.bsp(recovered,code.logicals.T)
        commutes_with_logicals = np.all(resolved_logical_commutations == 0)
        resolved_success = commutes_with_stabilizers and commutes_with_logicals
        # fill in unspecified outcomes
        success = resolved_success if success is None else success
        logical_commutations = resolved_logical_commutations if logical_commutations is None else logical_commutations

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('run: success={}'.format(success))
        logger.debug('run: logical_commutations={!r}'.format(logical_commutations))
        logger.debug('run: custom_values={!r}'.format(custom_values))

    data = {
        'error_weight': pt.bsf_wt(np.array(step_errors)),
        'success': bool(success),
        'max_coset_p': max_coset_p, 
        'logical_commutations': logical_commutations,
        'custom_values': custom_values,
    }

    return data


def _run_defp(mode,code,time_steps,error_model,decoder,error_probability,perm_rates,code_name,layout,measurement_error_probability,
         max_runs=None,max_failures=None,random_seed=None):
    """Implements run and run_ftp functions"""

    # assumptions
    assert (mode == 'ideal' and time_steps == 1) or mode == 'ftp'

    # derived defaults
    if max_runs is None and max_failures is None:
        max_runs = 1

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('run: code={},time_steps={},error_model={},decoder={},error_probability={},'
                     'measurement_error_probability={} max_runs={},max_failures={},random_seed={}.'
                     .format(code,time_steps,error_model,decoder,error_probability,
                             measurement_error_probability,max_runs,max_failures,random_seed))

    wall_time_start = time.perf_counter()

    runs_data = {
        'code': code.label,
        'n_k_d': code.n_k_d,
        'time_steps': time_steps,
        'error_model': error_model.label,
        'decoder': decoder.label,
        'error_probability': error_probability,
        'measurement_error_probability': measurement_error_probability,
        'n_run': 0,
        'n_success': 0,
        'n_fail': 0,
        'n_logical_commutations': None,
        'custom_totals': None,
        'error_weight_total': 0,
        'error_weight_pvar': 0.0,
        'logical_failure_rate_samples': 0.0,
        'logical_failure_rate_samples_errorbar': 0.0,
        'logical_failure_rate': 0.0,
        'logical_failure_rate_errorbar': 0.0,
        'physical_error_rate': 0.0,
        'wall_time': 0.0,
    }

    # if random_seed is None,unpredictable entropy is pulled from the OS,which we log for reproducibility
    seed_sequence = np.random.SeedSequence(random_seed)
    logger.info('run: np.random.SeedSequence.entropy={}'.format(seed_sequence.entropy))
    rng = np.random.default_rng(seed_sequence)

    array_sum_keys = ('n_logical_commutations','custom_totals',)  # list of array sum keys
    array_val_keys = ('logical_commutations','custom_values',)  # list of array value keys
    error_weights = []  # list of error_weight from current run
    success_list = np.zeros(max_runs)
    max_coset_p_list = np.zeros(max_runs)

    perm_mat,perm_vec= deform_matsvecs(code,decoder,error_model,perm_rates,code_name,layout)

    while ((max_runs is None or runs_data['n_run'] < max_runs)
           and (max_failures is None or runs_data['n_fail'] < max_failures)):
        # run simulation
        data = _run_once_defp(mode,code,time_steps,error_model,decoder,error_probability,perm_rates,perm_mat,perm_vec,code_name,layout,
                         measurement_error_probability,rng)
        # increment run counts
        success_list[runs_data['n_run']]     = data['success']
        max_coset_p_list[runs_data['n_run']] =data['max_coset_p'] 
        runs_data['n_run'] += 1
        if data['success']:
            runs_data['n_success'] += 1
        else:
            runs_data['n_fail'] += 1
        # sum arrays
        for array_sum_key,array_val_key in zip(array_sum_keys,array_val_keys):
            array_sum = runs_data[array_sum_key]  # extract sum
            array_val = data[array_val_key]  # extract val
            if runs_data['n_run'] == 1 and array_val is not None:  # first run,so initialize sum,if val not None
                array_sum = np.zeros_like(array_val)
            if array_sum is None and array_val is None:  # both None
                array_sum = None
            elif (array_sum is None or array_val is None) or (array_sum.shape != array_val.shape):  # mismatch
                raise QecsimError(
                    'Mismatch between {} values to sum: {},{}'.format(array_val_key,array_sum,array_val))
            else:  # match,so sum
                array_sum = array_sum + array_val
            runs_data[array_sum_key] = array_sum  # update runs_data
        # append error weight
        error_weights.append(data['error_weight'])

    #error bar in logical failure rate
    #print(max_coset_p_list)
    #print(max_coset_p_list.mean())
    runs_data['logical_failure_rate_samples_errorbar'] = success_list.std()/np.sqrt(max_runs)
    runs_data['logical_failure_rate_errorbar']         = max_coset_p_list.std()/np.sqrt(max_runs)
    runs_data['logical_failure_rate']                  = 1 - max_coset_p_list.mean()
    #print(runs_data['logical_failure_rate']) 
    # error weight statistics
    runs_data['error_weight_total'] = sum(error_weights)
    runs_data['error_weight_pvar'] = statistics.pvariance(error_weights)

    # rate statistics
    _add_rate_statistics(runs_data)

    # convert sum arrays to tuples if not None
    for array_sum_key in array_sum_keys:
        if runs_data[array_sum_key] is not None:
            runs_data[array_sum_key] = tuple(runs_data[array_sum_key].tolist())

    # record wall_time
    runs_data['wall_time'] = time.perf_counter() - wall_time_start

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('run: aggregated_data={}'.format(runs_data))

    return [[runs_data['logical_failure_rate'],runs_data['logical_failure_rate_errorbar']],
            [runs_data['logical_failure_rate_samples'],runs_data['logical_failure_rate_samples_errorbar']]]
            


def run_defp(code,error_model,decoder,error_probability,perm_rates,code_name,layout,max_runs=None,max_failures=None,random_seed=None):
    """
    Execute stabilizer code error-decode-recovery (ideal) simulation many times and return aggregated runs data.

    See :func:`run_once` for details of a single run.

    Notes:

    * The simulation is run one or more times as determined by ``max_runs`` and ``max_failures``:

        * If ``max_runs`` specified,stop after ``max_runs`` runs.
        * If ``max_failures`` specified,stop after ``max_failures`` failures.
        * If ``max_runs`` and ``max_failures`` unspecified,run once.

    * The returned data is in the following format:

    ::

        {
            'code': '5-qubit',                     # given code.label
            'n_k_d': (5,1,3),                    # given code.n_k_d
            'time_steps': 1,                       # always 1 for ideal simulation
            'error_model': 'Depolarizing',         # given error_model.label
            'decoder': 'Naive',                    # given decoder.label
            'error_probability': 0.0,              # given error_probability
            'measurement_error_probability': 0.0    # always 0.0 for ideal simulation
            'n_run': 0,                            # count of runs
            'n_success': 0,                        # count of successful recovery
            'n_fail': 0,                           # count of failed recovery
            'n_logical_commutations': (0,0),      # count of logical commutations (tuple)
            'custom_totals': None,                 # sum of custom values (tuple)
            'error_weight_total': 0,               # sum of error_weight over n_run runs
            'error_weight_pvar': 0.0,              # pvariance of error_weight over n_run runs
            'logical_failure_rate': 0.0,           # n_fail / n_run
            'physical_error_rate': 0.0,            # error_weight_total / n_k_d[0] / time_steps / n_run
            'wall_time': 0.0,                      # wall-time for run in fractional seconds
        }

    :param code: Stabilizer code.
    :type code: StabilizerCode
    :param error_model: Error model.
    :type error_model: ErrorModel
    :param decoder: Decoder.
    :type decoder: Decoder
    :param error_probability: Error probability.
    :type error_probability: float
    :param max_runs: Maximum number of runs. (default=None or 1 if max_failures unspecified,unrestricted=None)
    :type max_runs: int
    :param max_failures: Maximum number of failures. (default=None,unrestricted=None)
    :type max_failures: int
    :param random_seed: Error generation random seed. (default=None,unseeded=None)
    :type random_seed: int
    :return: Aggregated runs data.
    :rtype: dict
    :raises ValueError: if error_probability is not in [0,1].
    """

    # validate parameters
    if not (0 <= error_probability <= 1):
        raise ValueError('Error probability must be in [0,1].')
    return _run_defp('ideal',code,1,error_model,decoder,error_probability,perm_rates,code_name,layout,0.0,max_runs,max_failures,random_seed)


def _add_rate_statistics(runs_data):
    """Add `logical_failure_rate` and `physical_error_rate` as defined in :func:`_run`."""
    # extract data
    time_steps = runs_data['time_steps']
    n_run = runs_data['n_run']
    n_fail = runs_data['n_fail']
    error_weight_total = runs_data['error_weight_total']
    code_n_qubits = runs_data['n_k_d'][0]
    # add rate statistics
    runs_data['logical_failure_rate_samples'] = n_fail / n_run
    runs_data['physical_error_rate'] = error_weight_total / code_n_qubits / time_steps / n_run


def merge(*data_list):
    """
    Merge any number of lists of aggregated runs data.

    Notes:

    * The runs data is in the format specified in :func:`run` and :func:`fun_ftp`.
    * Merged data is grouped by: `(code,n_k_d,error_model,decoder,error_probability,time_steps,
      measurement_error_probability)`.
    * The following scalar values are summed: `n_run`,`n_success`,`n_fail`,`error_weight_total`,`wall_time`.
    * The following array values are summed: `n_logical_commutations`,`custom_totals`.
    * The following values are recalculated: `logical_failure_rate`,`physical_error_rate`.
    * The following values are *not* currently recalculated: `error_weight_pvar`.

    :param data_list: List of aggregated runs data.
    :type data_list: list of dict
    :return: Merged list of aggregated runs data.
    :rtype: list of dict
    :raises ValueError: if there is a mismatch between array values to be summed.
    """
    # define group keys,value keys and zero values
    grp_keys = ('code','n_k_d','error_model','decoder','error_probability','time_steps',
                'measurement_error_probability')
    scalar_val_keys = ('n_run','n_fail','n_success','error_weight_total','wall_time')
    scalar_zero_vals = (0,0,0,0,0.0)
    array_val_keys = ('n_logical_commutations','custom_totals',)
    # map of groups to sums (use ordered dict to preserve order as much as possible).
    grps_to_scalar_sums = collections.OrderedDict()
    grps_to_array_sums = {}
    # iterate through single list from given data lists
    for runs_data in itertools.chain(*data_list):
        # define defaults,create new data with defaults overwritten by data
        # support for 0.10 and 0.15 files:
        defaults_0_16 = {'time_steps': 1,'measurement_error_probability': 0.0}
        # support for pre-1.0b6 files:
        defaults_1_0b6 = {'n_logical_commutations': None,'custom_totals': None}
        runs_data = dict(itertools.chain(defaults_0_16.items(),defaults_1_0b6.items(),runs_data.items()))
        # extract group from data (note: force lists to tuples so group_id is hashable)
        group_id = tuple(tuple(v) if isinstance(v,list) else v for v in (runs_data[k] for k in grp_keys))
        # scalars: e.g. (10,6,4,256,10.34)
        scalar_vals = tuple(runs_data[k] for k in scalar_val_keys)  # extract from data
        scalar_sums = grps_to_scalar_sums.get(group_id,scalar_zero_vals)  # get sums (or zeros if not found)
        scalar_sums = tuple(sum(x) for x in zip(scalar_vals,scalar_sums))  # update sums
        grps_to_scalar_sums[group_id] = scalar_sums  # put sums
        # arrays: e.g. ((2,5),(3,8,2),None)
        # arrays: extract from data as tuple of None and tuples
        array_vals = tuple(None if runs_data[k] is None else tuple(runs_data[k]) for k in array_val_keys)
        try:  # get sums and update
            array_sums = []
            for array_sum,array_val in zip(grps_to_array_sums[group_id],array_vals):  # sum and value tuples in pairs
                if array_sum is None and array_val is None:  # Both None
                    array_sums.append(None)
                elif (array_sum is None or array_val is None) or (len(array_sum) != len(array_val)):  # Mismatch
                    raise ValueError('Mismatch between array values to sum: {},{}'.format(array_sum,array_val))
                else:  # matching length,so sum
                    array_sums.append(tuple(sum(x) for x in zip(array_sum,array_val)))
            array_sums = tuple(array_sums)
        except KeyError:  # set sums from values if not found
            array_sums = array_vals
        grps_to_array_sums[group_id] = array_sums  # put sums
    # flatten grps_to_scalar_sums and grps_to_array_sums
    merged_data_list = [dict(zip(grp_keys + scalar_val_keys + array_val_keys,
                                 group_id + scalar_sums + grps_to_array_sums[group_id]))
                        for group_id,scalar_sums in grps_to_scalar_sums.items()]
    # add rate statistics
    for runs_data in merged_data_list:
        _add_rate_statistics(runs_data)
    return merged_data_list
