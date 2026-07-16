import numpy as np
import pandas as pd
import pytest
from src.analysis import (
    check_semantic_exact,
    diagnose_jit_warmup,
    check_early_stop_diagnostic,
    compute_S_ij,
    correlation_analysis
)
from concurrent.futures import TimeoutError

def test_check_semantic_exact_success():
    assert check_semantic_exact("x + y", "y + x # ground truth") is True
    assert check_semantic_exact("x + y", "x - y") is False

def test_check_semantic_exact_sympify_error():
    # Invalid syntax should raise SympifyError/SyntaxError, not return None
    with pytest.raises(Exception):
        check_semantic_exact("x + * y", "x + y")

def test_check_semantic_exact_timeout(monkeypatch):
    # Mock sympy.simplify to hang
    import time
    def mock_simplify(*args, **kwargs):
        time.sleep(2.0)
    monkeypatch.setattr("sympy.simplify", mock_simplify)
    
    assert check_semantic_exact("x", "x", timeout_s=0.1) is None

def test_diagnose_jit_warmup():
    df = pd.DataFrame({
        'dataset': ['d1', 'd1', 'd1', 'd1'],
        'constraints': ['baseline', 'baseline', 'baseline', 'C1'],
        'wall_clock_s': [10.0, 2.0, 2.1, 1.0],
        'seed': [1, 2, 3, 1]
    })
    diag_df = diagnose_jit_warmup(df)
    assert diag_df.loc[0, 'jit_warmup_suspect'] == True
    assert diag_df.loc[1, 'jit_warmup_suspect'] == False
    
def test_check_early_stop_diagnostic():
    df = pd.DataFrame({
        'dataset': ['d1']*4,
        'constraints': ['baseline', 'baseline', 'C1', 'C1'],
        'recovered': [True, True, True, True],
        # Baseline has 0/2 timeout rate
        # C1 has 2/2 timeout rate -> anomalous
        'timeout_hit': [False, False, True, True]
    })
    diagnostics = check_early_stop_diagnostic(df)
    assert "d1/C1" in diagnostics
    assert diagnostics["d1/C1"]["rate"] == 1.0
    assert diagnostics["d1/C1"]["baseline_rate"] == 0.0

def test_compute_S_ij():
    df = pd.DataFrame({
        'dataset': ['d1']*6,
        'constraints': ['C1', 'C1', 'C2', 'C2', 'C1+C2', 'C1+C2'],
        'seed': [1, 2, 1, 2, 1, 2],
        'wall_clock_s': [
            10.0, 10.0,  # C1
            5.0,  5.0,   # C2
            2.0,  np.nan # C1+C2 -> seed 2 is censored
        ]
    })
    # S_1: min(10, 5)/2 = 2.5
    # S_2: min(10, 5)/NaN = NaN
    # Result S_ij should be NaN because n_valid < 3
    S_df = compute_S_ij(df)
    assert len(S_df) == 0
    
    # If we add more valid seeds
    df = pd.DataFrame({
        'dataset': ['d1']*9,
        'constraints': ['C1']*3 + ['C2']*3 + ['C1+C2']*3,
        'seed': [1, 2, 3]*3,
        'wall_clock_s': [
            10, 10, 10,
            5,  5,  5,
            2,  2,  2
        ]
    })
    S_df = compute_S_ij(df)
    assert len(S_df) == 1
    assert S_df.iloc[0]['S_ij'] == 2.5
    assert S_df.iloc[0]['n_valid_seeds'] == 3

def test_correlation_analysis():
    # 24 points: 4 datasets x 6 pairs
    datasets = ['d1', 'd2', 'd3', 'd4']
    pairs = ['C1,C2', 'C1,C3', 'C1,C4', 'C2,C3', 'C2,C4', 'C3,C4']
    
    m_records = []
    s_records = []
    
    for i, p in enumerate(pairs):
        c_i, c_j = p.split(',')
        m_ij = float(i) / 10.0 + 0.1 # 0.1 to 0.6
        for d in datasets:
            m_records.append({'pair': p, 'c_i': c_i, 'c_j': c_j, 'M_ij': m_ij})
            # Perfect correlation
            s_records.append({'dataset': d, 'pair': p, 'c_i': c_i, 'c_j': c_j, 'S_ij': m_ij * 2.0})
            
    M_df = pd.DataFrame(m_records).drop_duplicates(subset=['pair'])
    S_df = pd.DataFrame(s_records)
    
    results = correlation_analysis(M_df, S_df)
    
    assert np.isclose(results['pearson_r'], 1.0)
    assert np.isclose(results['ols_r2'], 1.0)
    
    # Check FDR p-values (6 p-values)
    assert len(results['fdr_p_values']) == 6
    # Perfect correlation, S_ij > 1, so t-test vs 1.0 should give small p-values for larger S_ij
    
    # Sensitivity exclusions count verification
    assert not np.isnan(results['sens_no_c1'])
    assert not np.isnan(results['sens_hard'])
    assert not np.isnan(results['sens_c4'])
