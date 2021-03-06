import itertools
import logging
import math
import sys

import numpy as np
import pytest

from qecsim import app
from qecsim.error import QecsimError
from qecsim.model import ErrorModel, Decoder, DecodeResult
from qecsim.models.basic import FiveQubitCode
from qecsim.models.basic import SteaneCode
from qecsim.models.color import Color666Code
from qecsim.models.color import Color666MPSDecoder
from qecsim.models.generic import BiasedDepolarizingErrorModel
from qecsim.models.generic import BiasedYXErrorModel
from qecsim.models.generic import BitFlipErrorModel
from qecsim.models.generic import BitPhaseFlipErrorModel
from qecsim.models.generic import CenterSliceErrorModel
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.generic import NaiveDecoder
from qecsim.models.generic import PhaseFlipErrorModel
from qecsim.models.planar import PlanarCMWPMDecoder
from qecsim.models.planar import PlanarCode
from qecsim.models.planar import PlanarMPSDecoder
from qecsim.models.planar import PlanarMWPMDecoder
from qecsim.models.planar import PlanarRMPSDecoder
from qecsim.models.planar import PlanarYDecoder
from qecsim.models.rotatedplanar import RotatedPlanarCode
from qecsim.models.rotatedplanar import RotatedPlanarMPSDecoder
from qecsim.models.rotatedplanar import RotatedPlanarRMPSDecoder
from qecsim.models.rotatedplanar import RotatedPlanarSMWPMDecoder
from qecsim.models.rotatedtoric import RotatedToricCode
from qecsim.models.rotatedtoric import RotatedToricSMWPMDecoder
from qecsim.models.toric import ToricCode
from qecsim.models.toric import ToricMWPMDecoder

# enable debug logging to ensure full coverage
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


class _FixedErrorModel(ErrorModel):
    def __init__(self, error):
        self.error = error

    def generate(self, code, probability, rng=None):
        return self.error

    @property
    def label(self):
        return 'fixed'


class _FixedDecoder(Decoder):
    def __init__(self, decoding):
        self.decoding = decoding

    def decode(self, code, syndrome, **kwargs):
        return self.decoding

    @property
    def label(self):
        return 'fixed'


@pytest.mark.parametrize('code, error_model, decoder', [
    # each code with each valid decoder
    (Color666Code(5), DepolarizingErrorModel(), Color666MPSDecoder(chi=8)),
    (FiveQubitCode(), DepolarizingErrorModel(), NaiveDecoder()),
    (PlanarCode(5, 5), DepolarizingErrorModel(), PlanarCMWPMDecoder()),
    (PlanarCode(5, 5), DepolarizingErrorModel(), PlanarMPSDecoder(chi=6)),
    (PlanarCode(5, 5), DepolarizingErrorModel(), PlanarMWPMDecoder()),
    (PlanarCode(5, 5), DepolarizingErrorModel(), PlanarRMPSDecoder(chi=6)),
    (PlanarCode(4, 5), BitPhaseFlipErrorModel(), PlanarYDecoder()),
    (RotatedPlanarCode(7, 7), DepolarizingErrorModel(), RotatedPlanarMPSDecoder(chi=8)),
    (RotatedPlanarCode(7, 7), DepolarizingErrorModel(), RotatedPlanarRMPSDecoder(chi=8)),
    (RotatedPlanarCode(7, 7), BiasedDepolarizingErrorModel(100), RotatedPlanarSMWPMDecoder()),
    (RotatedToricCode(6, 6), BiasedDepolarizingErrorModel(100), RotatedToricSMWPMDecoder()),
    (SteaneCode(), DepolarizingErrorModel(), NaiveDecoder()),
    (ToricCode(5, 5), DepolarizingErrorModel(), ToricMWPMDecoder()),
    # each generic noise model
    (PlanarCode(5, 5), BiasedDepolarizingErrorModel(10), PlanarMPSDecoder(chi=6)),
    (PlanarCode(5, 5), BiasedYXErrorModel(10), PlanarMPSDecoder(chi=6)),
    (PlanarCode(5, 5), BitFlipErrorModel(), PlanarMPSDecoder(chi=6)),
    (PlanarCode(5, 5), BitPhaseFlipErrorModel(), PlanarMPSDecoder(chi=6)),
    (PlanarCode(5, 5), CenterSliceErrorModel((0.2, 0.8, 0), 0.5), PlanarMPSDecoder(chi=6)),
    (PlanarCode(5, 5), DepolarizingErrorModel(), PlanarMPSDecoder(chi=6)),
    (PlanarCode(5, 5), PhaseFlipErrorModel(), PlanarMPSDecoder(chi=6)),
])
def test_run_once(code, error_model, decoder):
    error_probability = 0.15
    data = app.run_once(code, error_model, decoder, error_probability)  # no error raised
    expected_key_cls = {'error_weight': int, 'success': bool, 'logical_commutations': np.ndarray,
                        'custom_values': np.ndarray}
    assert data.keys() == expected_key_cls.keys(), 'data={} has missing/extra keys'
    for key, cls in expected_key_cls.items():
        assert data[key] is None or type(data[key]) == cls, 'data[{}]={} is not of type={}'.format(key, data[key], cls)


def test_run_once_seeded():
    code = PlanarCode(5, 5)
    error_model = DepolarizingErrorModel()
    decoder = PlanarMWPMDecoder()
    error_probability = 0.15
    data1 = app.run_once(code, error_model, decoder, error_probability, rng=np.random.default_rng(5))
    data2 = app.run_once(code, error_model, decoder, error_probability, rng=np.random.default_rng(5))
    assert data1['error_weight'] == data2['error_weight']
    assert data1['success'] == data2['success']
    assert np.array_equal(data1['logical_commutations'], data2['logical_commutations'])
    assert np.array_equal(data1['custom_values'], data2['custom_values'])


@pytest.mark.parametrize('code, error, decoding, expected_data', [
    # identity
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     {'success': True, 'logical_commutations': np.array([0, 0]), 'custom_values': None}),
    # logical X failure
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     PlanarCode(2, 2).new_pauli().logical_x().to_bsf(),
     {'success': False, 'logical_commutations': np.array([0, 1]), 'custom_values': None}),
    # logical Z failure
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     PlanarCode(2, 2).new_pauli().logical_z().to_bsf(),
     {'success': False, 'logical_commutations': np.array([1, 0]), 'custom_values': None}),
    # identity via decode-result
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(recovery=PlanarCode(2, 2).new_pauli().to_bsf()),
     {'success': True, 'logical_commutations': np.array([0, 0]), 'custom_values': None}),
    # logical X failure via decode-result
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(recovery=PlanarCode(2, 2).new_pauli().logical_x().to_bsf()),
     {'success': False, 'logical_commutations': np.array([0, 1]), 'custom_values': None}),
    # logical Z failure via decode-result
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(recovery=PlanarCode(2, 2).new_pauli().logical_z().to_bsf()),
     {'success': False, 'logical_commutations': np.array([1, 0]), 'custom_values': None}),
    # identity but override success
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(success=False, recovery=PlanarCode(2, 2).new_pauli().to_bsf()),
     {'success': False, 'logical_commutations': np.array([0, 0]), 'custom_values': None}),
    # identity but override logical_commutations
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(logical_commutations=np.array([1, 1]), recovery=PlanarCode(2, 2).new_pauli().to_bsf()),
     {'success': True, 'logical_commutations': np.array([1, 1]), 'custom_values': None}),
    # identity but override success and logical_commutations
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(success=False, logical_commutations=np.array([1, 1]), recovery=PlanarCode(2, 2).new_pauli().to_bsf()),
     {'success': False, 'logical_commutations': np.array([1, 1]), 'custom_values': None}),
    # identity but override success (no recovery)
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(success=False),
     {'success': False, 'logical_commutations': None, 'custom_values': None}),
    # identity but override success and logical_commutations (no recovery)
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(success=False, logical_commutations=np.array([1, 1])),
     {'success': False, 'logical_commutations': np.array([1, 1]), 'custom_values': None}),
    # identity via decode-result (with custom_values)
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(recovery=PlanarCode(2, 2).new_pauli().to_bsf(), custom_values=np.array([1])),
     {'success': True, 'logical_commutations': np.array([0, 0]), 'custom_values': np.array([1])}),
    # identity via decode-result (with custom_values)
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(recovery=PlanarCode(2, 2).new_pauli().to_bsf(), custom_values=np.array([0, 3])),
     {'success': True, 'logical_commutations': np.array([0, 0]), 'custom_values': np.array([0, 3])}),
])
def test_run_once_override(code, error, decoding, expected_data):
    # test execution paths returning different data
    data = app.run_once(code, _FixedErrorModel(error), _FixedDecoder(decoding), 0.0)
    print(data)
    assert data['success'] == expected_data['success']
    assert np.array_equal(data['logical_commutations'], expected_data['logical_commutations'])
    assert np.array_equal(data['custom_values'], expected_data['custom_values'])


@pytest.mark.parametrize('error_probability', [
    -0.1, 1.1,
])
def test_run_once_invalid_parameters(error_probability):
    with pytest.raises(ValueError) as exc_info:
        app.run_once(FiveQubitCode(), DepolarizingErrorModel(), NaiveDecoder(), error_probability)
    print(exc_info)


@pytest.mark.parametrize('code, time_steps, error_model, decoder, error_probability, measurement_error_probability', [
    # each code with each valid decoder
    (RotatedPlanarCode(7, 7), 7, BitPhaseFlipErrorModel(), RotatedPlanarSMWPMDecoder(), 0.05, None),
    (RotatedToricCode(6, 6), 6, BitPhaseFlipErrorModel(), RotatedToricSMWPMDecoder(), 0.05, None),
    # with optional measurement_error_probability
    (RotatedToricCode(6, 6), 6, BitPhaseFlipErrorModel(), RotatedToricSMWPMDecoder(), 0.05, 0.01),
])
def test_run_once_ftp(code, time_steps, error_model, decoder, error_probability, measurement_error_probability):
    data = app.run_once_ftp(code, time_steps, error_model, decoder, error_probability)
    expected_key_cls = {'error_weight': int, 'success': bool, 'logical_commutations': np.ndarray,
                        'custom_values': np.ndarray}
    assert data.keys() == expected_key_cls.keys(), 'data={} has missing/extra keys'
    for key, cls in expected_key_cls.items():
        assert data[key] is None or type(data[key]) == cls, 'data[{}]={} is not of type={}'.format(key, data[key], cls)


def test_run_once_ftp_seeded():
    code = RotatedToricCode(4, 4)
    time_steps = 4
    error_model = BitPhaseFlipErrorModel()
    decoder = RotatedToricSMWPMDecoder()
    error_probability = 0.15
    data1 = app.run_once_ftp(code, time_steps, error_model, decoder, error_probability, rng=np.random.default_rng(5))
    data2 = app.run_once_ftp(code, time_steps, error_model, decoder, error_probability, rng=np.random.default_rng(5))
    assert data1['error_weight'] == data2['error_weight']
    assert data1['success'] == data2['success']
    assert np.array_equal(data1['logical_commutations'], data2['logical_commutations'])
    assert np.array_equal(data1['custom_values'], data2['custom_values'])


@pytest.mark.parametrize('time_steps, error_probability, measurement_error_probability', [
    (0, 0.1, None),  # invalid time_steps
    (3, -0.1, None),  # invalid error_probability
    (3, 1.1, None),  # invalid error_probability
    (3, 0.1, -0.1),  # invalid measurement_error_probability
    (3, 0.1, 1.1),  # invalid measurement_error_probability
])
def test_run_once_ftp_invalid_parameters(time_steps, error_probability, measurement_error_probability):
    with pytest.raises(ValueError) as exc_info:
        app.run_once_ftp(RotatedToricCode(6, 6), time_steps, BitPhaseFlipErrorModel(), RotatedToricSMWPMDecoder(),
                         error_probability, measurement_error_probability)
    print(exc_info)


@pytest.mark.parametrize('code, error_model, decoder', [
    (Color666Code(5), DepolarizingErrorModel(), Color666MPSDecoder(chi=8)),
    (FiveQubitCode(), DepolarizingErrorModel(), NaiveDecoder()),
])
def test_run(code, error_model, decoder):
    error_probability = 0.15
    max_runs = 2
    data = app.run(code, error_model, decoder, error_probability, max_runs)  # no error raised
    expected_keys = {'code', 'n_k_d', 'error_model', 'decoder', 'error_probability', 'time_steps',
                     'measurement_error_probability', 'n_run', 'n_success', 'n_fail', 'n_logical_commutations',
                     'custom_totals', 'error_weight_total', 'error_weight_pvar', 'logical_failure_rate',
                     'physical_error_rate', 'wall_time'}
    assert data.keys() == expected_keys, 'data={} has missing/extra keys'
    assert data['n_run'] == max_runs, 'n_run does not equal requested max_runs (data={}).'.format(data)
    assert data['n_success'] + data['n_fail'] == max_runs, (
        'n_success + n_fail does not equal requested max_runs (data={}).'.format(data))
    assert data['n_success'] >= 0, 'n_success is negative (data={}).'.format(data)
    assert data['n_fail'] >= 0, 'n_fail is negative (data={}).'.format(data)
    assert data['n_fail'] <= sum(data['n_logical_commutations']), (
        'n_fail exceeds n_logical_commutations (data={}).'.format(data))
    assert data['logical_failure_rate'] == data['n_fail'] / data['n_run']


@pytest.mark.parametrize('max_runs, max_failures', [
    (None, None),
    (10, None),
    (None, 2),
    (10, 2),
])
def test_run_count(max_runs, max_failures):
    code = FiveQubitCode()
    error_model = BitPhaseFlipErrorModel()
    decoder = NaiveDecoder()
    error_probability = 0.05
    data = app.run(code, error_model, decoder, error_probability,
                   max_runs=max_runs, max_failures=max_failures)  # no error raised
    assert {'n_run', 'n_fail'} <= data.keys(), 'data={} missing count keys'
    if max_runs is None and max_failures is None:
        assert data['n_run'] == 1, 'n_run does not equal 1 when max_runs and max_failures unspecified'
    if max_runs is not None:
        assert data['n_run'] <= max_runs, ('n_run is not <= requested max_runs (data={}).'.format(data))
    if max_failures is not None:
        assert data['n_fail'] <= max_failures, ('n_fail is not <= requested max_failures (data={}).'.format(data))


def test_run_seeded():
    code = PlanarCode(5, 5)
    error_model = DepolarizingErrorModel()
    decoder = PlanarMPSDecoder()
    error_probability = 0.101
    max_runs = 5
    random_seed = 5
    data1 = app.run(code, error_model, decoder, error_probability, max_runs=max_runs, random_seed=random_seed)
    data2 = app.run(code, error_model, decoder, error_probability, max_runs=max_runs, random_seed=random_seed)
    # remove wall_time from data
    for data in (data1, data2):
        del data['wall_time']
    assert data1 == data2, 'Identically seeded runs are not the same. '


@pytest.mark.parametrize('code, error, decoding, max_runs, expected_data', [
    # identity
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     5,
     {'n_fail': 0, 'n_logical_commutations': (0, 0), 'custom_totals': None}),
    # logical X failure
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     PlanarCode(2, 2).new_pauli().logical_x().to_bsf(),
     5,
     {'n_fail': 5, 'n_logical_commutations': (0, 5), 'custom_totals': None}),
    # logical Z failure
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     PlanarCode(2, 2).new_pauli().logical_z().to_bsf(),
     8,
     {'n_fail': 8, 'n_logical_commutations': (8, 0), 'custom_totals': None}),
    # identity but override success
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(success=False, recovery=PlanarCode(2, 2).new_pauli().to_bsf()),
     7,
     {'n_fail': 7, 'n_logical_commutations': (0, 0), 'custom_totals': None}),
    # identity but override logical_commutations
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(logical_commutations=np.array([1, 1]), recovery=PlanarCode(2, 2).new_pauli().to_bsf()),
     3,
     {'n_fail': 0, 'n_logical_commutations': (3, 3), 'custom_totals': None}),
    # identity but override success and logical_commutations
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(success=False, logical_commutations=np.array([1, 1]), recovery=PlanarCode(2, 2).new_pauli().to_bsf()),
     4,
     {'n_fail': 4, 'n_logical_commutations': (4, 4), 'custom_totals': None}),
    # identity but override success (no recovery)
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(success=False),
     6,
     {'n_fail': 6, 'n_logical_commutations': None, 'custom_totals': None}),
    # identity but override success and logical_commutations (no recovery)
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(success=False, logical_commutations=np.array([1, 1])),
     2,
     {'n_fail': 2, 'n_logical_commutations': (2, 2), 'custom_totals': None}),
    # identity via decode-result (with custom_values)
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(recovery=PlanarCode(2, 2).new_pauli().to_bsf(), custom_values=np.array([1])),
     3,
     {'n_fail': 0, 'n_logical_commutations': (0, 0), 'custom_totals': (3,)}),
    # identity via decode-result (with custom_values)
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(recovery=PlanarCode(2, 2).new_pauli().to_bsf(), custom_values=np.array([1, 2])),
     5,
     {'n_fail': 0, 'n_logical_commutations': (0, 0), 'custom_totals': (5, 10)}),
    # identity via decode-result (with custom_values)
    (PlanarCode(2, 2),
     PlanarCode(2, 2).new_pauli().to_bsf(),
     DecodeResult(recovery=PlanarCode(2, 2).new_pauli().to_bsf(), custom_values=np.array([1.1, 2.2])),
     2,
     {'n_fail': 0, 'n_logical_commutations': (0, 0), 'custom_totals': (2.2, 4.4)}),
])
def test_run_override(code, error, decoding, max_runs, expected_data):
    # test n_fail and n_logical_commutations when returning different data
    data = app.run(code, _FixedErrorModel(error), _FixedDecoder(decoding), 0.0, max_runs=max_runs)
    print(data)
    assert data['n_fail'] == expected_data['n_fail']
    assert data['n_logical_commutations'] == expected_data['n_logical_commutations']
    assert data['custom_totals'] == expected_data['custom_totals']


@pytest.mark.parametrize('decoding1, decoding2', [
    (DecodeResult(success=True, logical_commutations=None),
     DecodeResult(success=True, logical_commutations=np.array([1, 0]))),
    (DecodeResult(success=True, logical_commutations=np.array([1, 1, 0])),
     DecodeResult(success=True, logical_commutations=np.array([1, 0]))),
])
def test_run_invalid_override(decoding1, decoding2):
    decodings = itertools.cycle((decoding1, decoding2))

    class _CycleDecoder(Decoder):

        def decode(self, code, syndrome, **kwargs):
            return next(decodings)

        @property
        def label(self):
            return 'Cycle'

    # should raise error due to inconsistent logical_commutations
    with pytest.raises(QecsimError):
        app.run(FiveQubitCode(), BitFlipErrorModel(), _CycleDecoder(), 0.0, max_runs=5)


@pytest.mark.parametrize('error_probability', [
    -0.1, 1.1,
])
def test_run_invalid_parameters(error_probability):
    with pytest.raises(ValueError) as exc_info:
        app.run(FiveQubitCode(), DepolarizingErrorModel(), NaiveDecoder(), error_probability, max_runs=2)
    print(exc_info)


@pytest.mark.parametrize('code, error_model, decoder, error_probability', [
    (FiveQubitCode(), DepolarizingErrorModel(), NaiveDecoder(), 0.05),
    (FiveQubitCode(), DepolarizingErrorModel(), NaiveDecoder(), 0.1),
    (FiveQubitCode(), DepolarizingErrorModel(), NaiveDecoder(), 0.15),
])
def test_run_physical_error_rate(code, error_model, decoder, error_probability):
    max_runs = 100  # Need to repeat many times to ensure physical_error_rate is close to error_probability
    data = app.run(code, error_model, decoder, error_probability, max_runs)  # no error raised
    for key in ('error_probability', 'physical_error_rate', 'error_weight_pvar'):
        assert key in data, 'data={} does not contain key={}'.format(data, key)
    e_prob = data['error_probability']
    p_rate = data['physical_error_rate']
    p_var = data['error_weight_pvar'] / (data['n_k_d'][0] ** 2)  # physical_error_rate_pvar (power of 2 is correct)
    p_std = math.sqrt(p_var)  # physical_error_rate_std
    assert p_rate - p_std < e_prob < p_rate + p_std, (
        'physical_error_rate={} is not within 1 std={} of error_probability={}'.format(p_rate, p_std, e_prob))


@pytest.mark.parametrize('code, time_steps, error_model, decoder', [
    (RotatedPlanarCode(7, 7), 7, BitPhaseFlipErrorModel(), RotatedPlanarSMWPMDecoder()),
    (RotatedToricCode(6, 6), 6, BitPhaseFlipErrorModel(), RotatedToricSMWPMDecoder()),
])
def test_run_ftp(code, time_steps, error_model, decoder):
    error_probability = 0.15
    max_runs = 2
    data = app.run_ftp(code, time_steps, error_model, decoder, error_probability, max_runs=max_runs)
    expected_keys = {'code', 'n_k_d', 'error_model', 'decoder', 'error_probability', 'time_steps',
                     'measurement_error_probability', 'n_run', 'n_success', 'n_fail', 'n_logical_commutations',
                     'custom_totals', 'error_weight_total', 'error_weight_pvar', 'logical_failure_rate',
                     'physical_error_rate', 'wall_time'}
    assert data.keys() == expected_keys, 'data={} has missing/extra keys'
    assert data['n_run'] == max_runs, ('n_run does not equal requested max_runs (data={}).'.format(data))
    assert data['n_success'] + data['n_fail'] == max_runs, (
        'n_success + n_fail does not equal requested max_runs (data={}).'.format(data))
    assert data['n_success'] >= 0, 'n_success is negative (data={}).'.format(data)
    assert data['n_fail'] >= 0, 'n_fail is negative (data={}).'.format(data)
    assert data['logical_failure_rate'] == data['n_fail'] / data['n_run']


@pytest.mark.parametrize('max_runs, max_failures', [
    (None, None),
    (10, None),
    (None, 2),
    (10, 2),
])
def test_run_ftp_count(max_runs, max_failures):
    code = RotatedToricCode(6, 6)
    time_steps = 6
    error_model = BitPhaseFlipErrorModel()
    decoder = RotatedToricSMWPMDecoder()
    error_probability = 0.05
    data = app.run_ftp(code, time_steps, error_model, decoder, error_probability,
                       max_runs=max_runs, max_failures=max_failures)  # no error raised
    assert {'n_run', 'n_fail'} <= data.keys(), 'data={} missing count keys'
    if max_runs is None and max_failures is None:
        assert data['n_run'] == 1, 'n_run does not equal 1 when max_runs and max_failures unspecified'
    if max_runs is not None:
        assert data['n_run'] <= max_runs, ('n_run is not <= requested max_runs (data={}).'.format(data))
    if max_failures is not None:
        assert data['n_fail'] <= max_failures, ('n_fail is not <= requested max_failures (data={}).'.format(data))


@pytest.mark.parametrize('time_steps, error_probability, measurement_error_probability, expected', [
    (2, 0.1, None, 0.1),  # time_steps > 1, default to error_probability
    (1, 0.1, None, 0.0),  # time_steps = 1, default to 0
    (2, 0.1, 0.05, 0.05),  # time_steps > 1, override
    (1, 0.1, 0.05, 0.05)  # time_steps = 1, override
])
def test_run_ftp_measurement_error_probability_defaults(time_steps, error_probability, measurement_error_probability,
                                                        expected):
    code = RotatedToricCode(4, 4)
    error_model = BitPhaseFlipErrorModel()
    decoder = RotatedToricSMWPMDecoder()
    max_runs = 2
    data = app.run_ftp(code, time_steps, error_model, decoder, error_probability, measurement_error_probability,
                       max_runs=max_runs)
    assert data['measurement_error_probability'] == expected


def test_run_ftp_seeded():
    code = RotatedToricCode(4, 4)
    time_steps = 4
    error_model = BitPhaseFlipErrorModel()
    decoder = RotatedToricSMWPMDecoder()
    error_probability = 0.15
    max_runs = 5
    random_seed = 5
    data1 = app.run_ftp(code, time_steps, error_model, decoder, error_probability, max_runs=max_runs,
                        random_seed=random_seed)
    data2 = app.run_ftp(code, time_steps, error_model, decoder, error_probability, max_runs=max_runs,
                        random_seed=random_seed)
    # remove wall_time from data
    for data in (data1, data2):
        del data['wall_time']
    assert data1 == data2, 'Identically seeded runs are not the same. '


@pytest.mark.parametrize('time_steps, error_probability, measurement_error_probability', [
    (0, 0.1, None),  # invalid time_steps
    (3, -0.1, None),  # invalid error_probability
    (3, 1.1, None),  # invalid error_probability
    (3, 0.1, -0.1),  # invalid measurement_error_probability
    (3, 0.1, 1.1),  # invalid measurement_error_probability
])
def test_run_ftp_invalid_parameters(time_steps, error_probability, measurement_error_probability):
    with pytest.raises(ValueError) as exc_info:
        app.run_ftp(RotatedToricCode(6, 6), time_steps, BitPhaseFlipErrorModel(), RotatedToricSMWPMDecoder(),
                    error_probability, measurement_error_probability, max_runs=2)
    print(exc_info)


@pytest.mark.parametrize('code, time_steps, error_model, decoder, error_probability', [
    (RotatedPlanarCode(3, 3), 3, DepolarizingErrorModel(), RotatedPlanarSMWPMDecoder(), 0.05),
    (RotatedPlanarCode(3, 3), 3, DepolarizingErrorModel(), RotatedPlanarSMWPMDecoder(), 0.1),
    (RotatedPlanarCode(3, 3), 3, DepolarizingErrorModel(), RotatedPlanarSMWPMDecoder(), 0.15),
])
def test_run_ftp_physical_error_rate(code, time_steps, error_model, decoder, error_probability):
    max_runs = 100  # Need to repeat many times to ensure physical_error_rate is close to error_probability
    data = app.run_ftp(code, time_steps, error_model, decoder, error_probability, max_runs=max_runs)
    for key in ('error_probability', 'physical_error_rate', 'error_weight_pvar'):
        assert key in data, 'data={} does not contain key={}'.format(data, key)
    e_prob = data['error_probability']
    p_rate = data['physical_error_rate']
    p_var = data['error_weight_pvar'] / (data['n_k_d'][0] ** 2)  # physical_error_rate_pvar (power of 2 is correct)
    p_std = math.sqrt(p_var)  # physical_error_rate_std
    assert p_rate - p_std < e_prob < p_rate + p_std, (
        'physical_error_rate={} is not within 1 std={} of error_probability={}'.format(p_rate, p_std, e_prob))


@pytest.mark.parametrize('data, expected', [
    # (([{'code': '5-qubit', 'n_run': 10}], [{'code': '5-qubit', 'n_run': 20}]), [{'code': '5-qubit', 'n_run': 30}]),
    # single data set is unchanged (data sets from merge)
    (([{'n_k_d': (450, 2, 15), 'physical_error_rate': 0.09496, 'n_run': 10000, 'decoder': 'Toric MWPM',
        'error_weight_total': 427320, 'error_model': 'Bit-flip', 'n_success': 8310, 'n_fail': 1690,
        'code': 'Toric 15x15', 'wall_time': 591.4239950589836, 'logical_failure_rate': 0.169,
        'error_probability': 0.095}],
      ),
     [{'n_k_d': (450, 2, 15), 'physical_error_rate': 0.09496, 'n_run': 10000, 'decoder': 'Toric MWPM',
       'error_weight_total': 427320, 'error_model': 'Bit-flip', 'n_success': 8310, 'n_fail': 1690,
       'code': 'Toric 15x15', 'wall_time': 591.4239950589836, 'logical_failure_rate': 0.169,
       'error_probability': 0.095, 'time_steps': 1, 'measurement_error_probability': 0.0,
       'n_logical_commutations': None, 'custom_totals': None}]
     ),
    # multiple same-group data sets are merged (data sets from merge)
    (([{'n_k_d': (450, 2, 15), 'physical_error_rate': 0.09496, 'n_run': 10000, 'decoder': 'Toric MWPM',
        'error_weight_total': 427320, 'error_model': 'Bit-flip', 'n_success': 8310, 'n_fail': 1690,
        'code': 'Toric 15x15', 'wall_time': 591.4239950589836, 'logical_failure_rate': 0.169,
        'error_probability': 0.095}],
      [{'error_probability': 0.095, 'code': 'Toric 15x15', 'n_k_d': (450, 2, 15), 'n_fail': 1720,
        'error_weight_total': 428968, 'physical_error_rate': 0.09532622222222223, 'decoder': 'Toric MWPM',
        'error_model': 'Bit-flip', 'n_run': 10000, 'n_success': 8280, 'logical_failure_rate': 0.172,
        'wall_time': 789.6659666132182}],
      ),
     [{'error_probability': 0.095, 'code': 'Toric 15x15', 'n_k_d': (450, 2, 15), 'n_fail': 3410,
       'error_weight_total': 856288, 'physical_error_rate': 0.09514311111111111, 'decoder': 'Toric MWPM',
       'error_model': 'Bit-flip', 'n_run': 20000, 'n_success': 16590, 'logical_failure_rate': 0.1705,
       'wall_time': 1381.0899616722018, 'time_steps': 1, 'measurement_error_probability': 0.0,
       'n_logical_commutations': None, 'custom_totals': None}]
     ),
    # multiple different-group data sets are not merged (data sets from merge)
    (([{'n_k_d': (450, 2, 15), 'physical_error_rate': 0.09496, 'n_run': 10000, 'decoder': 'Toric MWPM',
        'error_weight_total': 427320, 'error_model': 'Bit-flip', 'n_success': 8310, 'n_fail': 1690,
        'code': 'Toric 15x15', 'wall_time': 591.4239950589836, 'logical_failure_rate': 0.169,
        'error_probability': 0.095}],
      [{'error_probability': 0.095, 'code': 'Toric 15x15', 'n_k_d': (450, 2, 15), 'n_fail': 1720,
        'error_weight_total': 428968, 'physical_error_rate': 0.09532622222222223, 'decoder': 'Toric MWPM',
        'error_model': 'Bit-flip', 'n_run': 10000, 'n_success': 8280, 'logical_failure_rate': 0.172,
        'wall_time': 789.6659666132182}],
      [{'logical_failure_rate': 0.1172, 'decoder': 'Toric MWPM', 'wall_time': 60594.737112408504,
        'n_k_d': (2450, 2, 35), 'n_success': 8828, 'code': 'Toric 35x35', 'n_run': 10000, 'n_fail': 1172,
        'error_weight_total': 2326883, 'physical_error_rate': 0.09497481632653061,
        'error_probability': 0.095, 'error_model': 'Bit-flip'}],
      ),
     [{'error_probability': 0.095, 'code': 'Toric 15x15', 'n_k_d': (450, 2, 15), 'n_fail': 3410,
       'error_weight_total': 856288, 'physical_error_rate': 0.09514311111111111, 'decoder': 'Toric MWPM',
       'error_model': 'Bit-flip', 'n_run': 20000, 'n_success': 16590, 'logical_failure_rate': 0.1705,
       'wall_time': 1381.0899616722018, 'time_steps': 1, 'measurement_error_probability': 0.0,
       'n_logical_commutations': None, 'custom_totals': None},
      {'logical_failure_rate': 0.1172, 'decoder': 'Toric MWPM', 'wall_time': 60594.737112408504,
       'n_k_d': (2450, 2, 35), 'n_success': 8828, 'code': 'Toric 35x35', 'n_run': 10000, 'n_fail': 1172,
       'error_weight_total': 2326883, 'physical_error_rate': 0.09497481632653061, 'error_probability': 0.095,
       'error_model': 'Bit-flip', 'time_steps': 1, 'measurement_error_probability': 0.0,
       'n_logical_commutations': None, 'custom_totals': None}]
     ),
    # error_weight_pvar ignored, n_k_d in list form converted to tuple (data sets from run)
    (([{'code': 'Toric 3x3', 'decoder': 'Toric MWPM', 'error_model': 'Bit-flip', 'error_probability': 0.1,
        'error_weight_pvar': 1.8983999999999999, 'error_weight_total': 196, 'logical_failure_rate': 0.24,
        'n_fail': 24, 'n_k_d': [18, 2, 3], 'n_run': 100, 'n_success': 76,
        'physical_error_rate': 0.1088888888888889, 'wall_time': 0.2560051280015614}],
      [{'code': 'Toric 3x3', 'decoder': 'Toric MWPM', 'error_model': 'Bit-flip', 'error_probability': 0.1,
        'error_weight_pvar': 1.7996, 'error_weight_total': 202, 'logical_failure_rate': 0.33,
        'n_fail': 33, 'n_k_d': [18, 2, 3], 'n_run': 100, 'n_success': 67,
        'physical_error_rate': 0.11222222222222221, 'wall_time': 0.24756049499774235}],
      ),
     [{'code': 'Toric 3x3', 'decoder': 'Toric MWPM', 'error_model': 'Bit-flip', 'error_probability': 0.1,
       'error_weight_total': 398, 'logical_failure_rate': 0.285, 'n_fail': 57, 'n_k_d': (18, 2, 3), 'n_run': 200,
       'n_success': 143, 'physical_error_rate': 0.11055555555555556, 'wall_time': 0.5035656229993037,
       'time_steps': 1, 'measurement_error_probability': 0.0, 'n_logical_commutations': None, 'custom_totals': None}]
     ),
    # multiple same-group (including measurement_error_probability) data sets are merged
    (([{'error_probability': 0.2, 'code': 'Steane', 'n_run': 5, 'error_model': 'Phase-flip',
        'error_weight_pvar': 0.24000000000000002, 'physical_error_rate': 0.08571428571428572,
        'n_k_d': (7, 1, 3), 'wall_time': 0.020358342910185456, 'n_fail': 0, 'n_success': 5,
        'logical_failure_rate': 0.0, 'time_steps': 2, 'measurement_error_probability': 0.01,
        'error_weight_total': 6, 'decoder': 'Naive'}],
      [{'n_run': 5, 'time_steps': 2, 'measurement_error_probability': 0.01, 'error_model': 'Phase-flip',
        'physical_error_rate': 0.2, 'decoder': 'Naive', 'n_fail': 2, 'n_k_d': (7, 1, 3),
        'wall_time': 0.01997815491631627, 'logical_failure_rate': 0.4, 'error_probability': 0.2,
        'error_weight_total': 14, 'error_weight_pvar': 1.04, 'n_success': 3, 'code': 'Steane'}],
      ),
     [{'n_run': 10, 'time_steps': 2, 'measurement_error_probability': 0.01, 'error_model': 'Phase-flip',
       'physical_error_rate': 0.14285714285714285, 'decoder': 'Naive', 'n_fail': 2, 'n_k_d': (7, 1, 3),
       'wall_time': 0.04033649782650173, 'logical_failure_rate': 0.2, 'error_probability': 0.2,
       'error_weight_total': 20, 'n_success': 8, 'code': 'Steane', 'n_logical_commutations': None,
       'custom_totals': None}]
     ),
    # multiple different-group (including measurement_error_probability) data sets are not merged
    (([{'n_success': 5, 'physical_error_rate': 0.02857142857142857, 'wall_time': 0.019910499919205904,
        'error_probability': 0.05, 'error_model': 'Phase-flip', 'error_weight_pvar': 0.16000000000000003,
        'n_run': 5, 'error_weight_total': 2, 'n_fail': 0, 'time_steps': 2,
        'measurement_error_probability': 0.01, 'n_k_d': (7, 1, 3), 'code': 'Steane', 'decoder': 'Naive',
        'logical_failure_rate': 0.0}],
      [{'error_weight_pvar': 0, 'code': 'Steane', 'decoder': 'Naive', 'n_run': 5, 'n_k_d': (7, 1, 3),
        'physical_error_rate': 0.0, 'error_weight_total': 0, 'n_fail': 0,
        'time_steps': 1, 'measurement_error_probability': 0.0, 'error_probability': 0.05,
        'wall_time': 0.013491070130839944, 'logical_failure_rate': 0.0, 'n_success': 5,
        'error_model': 'Phase-flip'}],
      ),
     [{"code": "Steane", "decoder": "Naive", "error_model": "Phase-flip", "error_probability": 0.05,
       "error_weight_total": 2, "logical_failure_rate": 0.0, "time_steps": 2,
       "measurement_error_probability": 0.01, "n_fail": 0, "n_k_d": (7, 1, 3), "n_run": 5, "n_success": 5,
       "physical_error_rate": 0.02857142857142857, "wall_time": 0.019910499919205904, 'n_logical_commutations': None,
       "custom_totals": None},
      {"code": "Steane", "decoder": "Naive", "error_model": "Phase-flip", "error_probability": 0.05,
       "error_weight_total": 0, "logical_failure_rate": 0.0, "time_steps": 1,
       "measurement_error_probability": 0.0, "n_fail": 0, "n_k_d": (7, 1, 3), "n_run": 5, "n_success": 5,
       "physical_error_rate": 0.0, "wall_time": 0.013491070130839944, "n_logical_commutations": None,
       "custom_totals": None}]
     ),
    # logical commutations and custom
    (([{"code": "Rotated planar 13x13", "decoder": "Rotated planar RMPS (chi=16, mode=c)",
        "error_model": "Depolarizing", "error_probability": 0.3, "error_weight_pvar": 48.36000000000001,
        "error_weight_total": 508, "logical_failure_rate": 0.8, "measurement_error_probability": 0.0,
        "custom_totals": [2], "n_fail": 8, "n_k_d": [169, 1, 13], "n_logical_commutations": [5, 4], "n_run": 10,
        "n_success": 2, "physical_error_rate": 0.30059171597633133, "time_steps": 1, "wall_time": 4.250106001000001}],
      [{"code": "Rotated planar 13x13", "decoder": "Rotated planar RMPS (chi=16, mode=c)",
        "error_model": "Depolarizing", "error_probability": 0.3, "error_weight_pvar": 65.04,
        "error_weight_total": 267, "logical_failure_rate": 1.0, "measurement_error_probability": 0.0,
        "custom_totals": [1], "n_fail": 5, "n_k_d": [169, 1, 13], "n_logical_commutations": [3, 2], "n_run": 5,
        "n_success": 0, "physical_error_rate": 0.31597633136094677, "time_steps": 1, "wall_time": 2.029527459}],
      ),
     [{"code": "Rotated planar 13x13", "decoder": "Rotated planar RMPS (chi=16, mode=c)",
       "error_model": "Depolarizing", "error_probability": 0.3,
       "error_weight_total": 775, "logical_failure_rate": 0.8666666666666667, "measurement_error_probability": 0.0,
       "custom_totals": (3,), "n_fail": 13, "n_k_d": (169, 1, 13), "n_logical_commutations": (8, 6), "n_run": 15,
       "n_success": 2, "physical_error_rate": 0.3057199211045365, "time_steps": 1, "wall_time": 6.279633460000001}]
     ),
])
def test_merge(data, expected):
    actual = app.merge(*data)

    print()
    print('EXPECTED=')
    for r in expected:
        for k, v in sorted(r.items()):
            print(k, v)
    print()
    print('ACTUAL=')
    for r in actual:
        for k, v in sorted(r.items()):
            print(k, v)
    print()

    assert actual == expected, 'Merged data=\n{}\ndoes not match expected\ndata=\n{}'.format(actual, expected)


@pytest.mark.parametrize('data', [
    # inconsistent logical commutations (first: (5, 4), second: None)
    ([{"code": "Rotated planar 13x13", "decoder": "Rotated planar RMPS (chi=16, mode=c)",
       "error_model": "Depolarizing", "error_probability": 0.3, "error_weight_pvar": 48.36000000000001,
       "error_weight_total": 508, "logical_failure_rate": 0.8, "measurement_error_probability": 0.0,
       "custom_totals": None, "n_fail": 8, "n_k_d": (169, 1, 13), "n_logical_commutations": (5, 4), "n_run": 10,
       "n_success": 2, "physical_error_rate": 0.30059171597633133, "time_steps": 1, "wall_time": 4.250106001000001}],
     [{"code": "Rotated planar 13x13", "decoder": "Rotated planar RMPS (chi=16, mode=c)",
       "error_model": "Depolarizing", "error_probability": 0.3, "error_weight_pvar": 65.04,
       "error_weight_total": 267, "logical_failure_rate": 1.0, "measurement_error_probability": 0.0,
       "custom_totals": None, "n_fail": 5, "n_k_d": (169, 1, 13), "n_logical_commutations": None, "n_run": 5,
       "n_success": 0, "physical_error_rate": 0.31597633136094677, "time_steps": 1, "wall_time": 2.029527459}],
     ),
    # inconsistent logical commutations (first: None, second: (3, 2))
    ([{"code": "Rotated planar 13x13", "decoder": "Rotated planar RMPS (chi=16, mode=c)",
       "error_model": "Depolarizing", "error_probability": 0.3, "error_weight_pvar": 48.36000000000001,
       "error_weight_total": 508, "logical_failure_rate": 0.8, "measurement_error_probability": 0.0,
       "custom_totals": None, "n_fail": 8, "n_k_d": (169, 1, 13), "n_logical_commutations": None, "n_run": 10,
       "n_success": 2, "physical_error_rate": 0.30059171597633133, "time_steps": 1, "wall_time": 4.250106001000001}],
     [{"code": "Rotated planar 13x13", "decoder": "Rotated planar RMPS (chi=16, mode=c)",
       "error_model": "Depolarizing", "error_probability": 0.3, "error_weight_pvar": 65.04,
       "error_weight_total": 267, "logical_failure_rate": 1.0, "measurement_error_probability": 0.0,
       "custom_totals": None, "n_fail": 5, "n_k_d": (169, 1, 13), "n_logical_commutations": (3, 2), "n_run": 5,
       "n_success": 0, "physical_error_rate": 0.31597633136094677, "time_steps": 1, "wall_time": 2.029527459}],
     ),
    # inconsistent logical commutations (first: (5, 4), second: (3, 2, 1))
    ([{"code": "Rotated planar 13x13", "decoder": "Rotated planar RMPS (chi=16, mode=c)",
       "error_model": "Depolarizing", "error_probability": 0.3, "error_weight_pvar": 48.36000000000001,
       "error_weight_total": 508, "logical_failure_rate": 0.8, "measurement_error_probability": 0.0,
       "custom_totals": None, "n_fail": 8, "n_k_d": (169, 1, 13), "n_logical_commutations": (5, 4), "n_run": 10,
       "n_success": 2, "physical_error_rate": 0.30059171597633133, "time_steps": 1, "wall_time": 4.250106001000001}],
     [{"code": "Rotated planar 13x13", "decoder": "Rotated planar RMPS (chi=16, mode=c)",
       "error_model": "Depolarizing", "error_probability": 0.3, "error_weight_pvar": 65.04,
       "error_weight_total": 267, "logical_failure_rate": 1.0, "measurement_error_probability": 0.0,
       "custom_totals": None, "n_fail": 5, "n_k_d": (169, 1, 13), "n_logical_commutations": (3, 2, 1), "n_run": 5,
       "n_success": 0, "physical_error_rate": 0.31597633136094677, "time_steps": 1, "wall_time": 2.029527459}],
     ),
    # inconsistent custom (first: (2,), second: None)
    ([{"code": "Rotated planar 13x13", "decoder": "Rotated planar RMPS (chi=16, mode=c)",
       "error_model": "Depolarizing", "error_probability": 0.3, "error_weight_pvar": 48.36000000000001,
       "error_weight_total": 508, "logical_failure_rate": 0.8, "measurement_error_probability": 0.0,
       "custom_totals": (2,), "n_fail": 8, "n_k_d": (169, 1, 13), "n_logical_commutations": (5, 4), "n_run": 10,
       "n_success": 2, "physical_error_rate": 0.30059171597633133, "time_steps": 1, "wall_time": 4.250106001000001}],
     [{"code": "Rotated planar 13x13", "decoder": "Rotated planar RMPS (chi=16, mode=c)",
       "error_model": "Depolarizing", "error_probability": 0.3, "error_weight_pvar": 65.04,
       "error_weight_total": 267, "logical_failure_rate": 1.0, "measurement_error_probability": 0.0,
       "custom_totals": None, "n_fail": 5, "n_k_d": (169, 1, 13), "n_logical_commutations": None, "n_run": 5,
       "n_success": 0, "physical_error_rate": 0.31597633136094677, "time_steps": 1, "wall_time": 2.029527459}],
     ),
])
def test_merge_invalid(data):
    with pytest.raises(ValueError):
        app.merge(*data)
