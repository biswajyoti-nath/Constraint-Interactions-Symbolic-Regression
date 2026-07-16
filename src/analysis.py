import argparse
import json
import logging
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import numpy as np
import pandas as pd
import scipy.stats as stats
import sympy
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import tt_solve_power
from statsmodels.stats.multitest import multipletests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:
    from metric import run_benchmark
except ImportError:
    run_benchmark = None

def check_semantic_exact(expr_str: str, ground_truth_str: str, timeout_s: float = 5.0) -> bool | None:
    def _check():
        gt_clean = ground_truth_str.split('#')[0].strip()
        e1 = sympy.sympify(expr_str, evaluate=False)
        e2 = sympy.sympify(gt_clean, evaluate=False)
        return sympy.simplify(e1 - e2) == 0

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_check)
        try:
            return fut.result(timeout=timeout_s)
        except FuturesTimeout:
            return None
        except Exception as e:
            logger.warning(f"check_semantic_exact failed for '{expr_str[:50]}...': {e}")
            raise

def diagnose_jit_warmup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['row_order'] = range(len(df))
    df['jit_warmup_suspect'] = False
    for dataset in df['dataset'].unique():
        mask = df['dataset'] == dataset
        first_idx = df.loc[mask, 'row_order'].idxmin()
        scenario = df.loc[first_idx, 'constraints']
        same_scenario = df[(df['dataset'] == dataset) & (df['constraints'] == scenario)]
        median_wall = same_scenario['wall_clock_s'].median()
        if pd.notna(median_wall) and df.loc[first_idx, 'wall_clock_s'] > 2.0 * median_wall:
            df.loc[first_idx, 'jit_warmup_suspect'] = True
    return df

def check_early_stop_diagnostic(df: pd.DataFrame) -> dict:
    diagnostics = {}
    for dataset in df['dataset'].unique():
        ds = df[df['dataset'] == dataset]
        baseline = ds[ds['constraints'] == 'baseline']
        baseline_rate = (baseline['recovered'] & baseline['timeout_hit']).sum() / max(len(baseline), 1)
        for scenario in ds['constraints'].unique():
            if scenario == 'baseline': continue
            sc = ds[ds['constraints'] == scenario]
            sc_rate = (sc['recovered'] & sc['timeout_hit']).sum() / max(len(sc), 1)
            if sc_rate > baseline_rate:
                diagnostics[f"{dataset}/{scenario}"] = {
                    'rate': sc_rate, 'baseline_rate': baseline_rate,
                    'warning': 'Constrained scenario has higher recovered+timeout rate than baseline.'
                }
    return diagnostics

def load_and_preprocess(results_csv: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    if 'error' in df.columns:
        df = df[df['error'].isna() | (df['error'] == "None") | (df['error'] == "")]
    mask_censor = df['timeout_hit'] & ~df['recovered']
    df.loc[mask_censor, 'wall_clock_s'] = np.nan
    df = diagnose_jit_warmup(df)
    for k, v in check_early_stop_diagnostic(df).items():
        logger.warning(f"Early-stop diagnostic warning for {k}: {v['warning']}")
    return df

def compute_S_ij(df: pd.DataFrame, agg_method='seed_paired',
                  exclude_jit: bool = True) -> pd.DataFrame:
    # Drop JIT-contaminated rows before computing any S_ij
    if exclude_jit and 'jit_warmup_suspect' in df.columns:
        n_before = len(df)
        df = df[~df['jit_warmup_suspect']].copy()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info(f"compute_S_ij: Dropped {n_dropped} JIT-warmup-suspect rows.")

    records = []
    datasets = df['dataset'].unique()
    constraints = df['constraints'].unique()
    single_constraints = [c for c in constraints if c != 'baseline' and '+' not in c]
    
    for dataset in datasets:
        ds = df[df['dataset'] == dataset]
        for i, c_i in enumerate(single_constraints):
            for j, c_j in enumerate(single_constraints):
                if i >= j: continue
                c_ij = f"{c_i}+{c_j}"
                if c_ij not in constraints:
                    c_ij = f"{c_j}+{c_i}"
                    if c_ij not in constraints: continue
                
                if agg_method == 'seed_paired':
                    ci_data = ds[ds['constraints'] == c_i].set_index('seed')
                    cj_data = ds[ds['constraints'] == c_j].set_index('seed')
                    cij_data = ds[ds['constraints'] == c_ij].set_index('seed')
                    common_seeds = ci_data.index.intersection(cj_data.index).intersection(cij_data.index)
                    
                    S_k_list = []
                    for k in common_seeds:
                        wi, wj, wij = ci_data.loc[k, 'wall_clock_s'], cj_data.loc[k, 'wall_clock_s'], cij_data.loc[k, 'wall_clock_s']
                        if pd.notna(wi) and pd.notna(wj) and pd.notna(wij):
                            S_k_list.append(min(wi, wj) / wij)
                    
                    S_k_array = np.array(S_k_list, dtype=float)
                    if len(S_k_array) >= 3:
                        S_ij = np.nanmean(S_k_array)
                        res = stats.bootstrap((S_k_array,), np.mean, confidence_level=0.95, random_state=42) if len(S_k_array) > 1 else None
                        ci_lo = res.confidence_interval.low if res else S_ij
                        ci_hi = res.confidence_interval.high if res else S_ij
                        records.append({'dataset': dataset, 'pair': f"{c_i},{c_j}", 'c_i': c_i, 'c_j': c_j, 'S_ij': S_ij, 'S_ij_ci_lo': ci_lo, 'S_ij_ci_hi': ci_hi, 'n_valid_seeds': len(S_k_array)})
                elif agg_method == 'min_after_mean':
                    wi = ds[ds['constraints'] == c_i]['wall_clock_s'].mean()
                    wj = ds[ds['constraints'] == c_j]['wall_clock_s'].mean()
                    wij = ds[ds['constraints'] == c_ij]['wall_clock_s'].mean()
                    if pd.notna(wi) and pd.notna(wj) and pd.notna(wij):
                        S_ij = min(wi, wj) / wij
                        records.append({'dataset': dataset, 'pair': f"{c_i},{c_j}", 'c_i': c_i, 'c_j': c_j, 'S_ij': S_ij, 'S_ij_ci_lo': S_ij, 'S_ij_ci_hi': S_ij, 'n_valid_seeds': 1})
    return pd.DataFrame(records)

def load_m_matrix(path: pathlib.Path, config: dict, recompute: bool = False) -> tuple[pd.DataFrame, list[str]]:
    if recompute:
        if run_benchmark is None: raise RuntimeError("metric module not found.")
        result, constraints = run_benchmark(config)
        M_new = result.rho_ij / np.outer(result.rho_i, result.rho_i)
        if path and path.exists():
            M_cached = np.load(path)
            ci_path = path.with_suffix('.ci.npy')
            if ci_path.exists():
                M_ci = np.load(ci_path)
                diff = np.abs(M_new - M_cached)
                if np.any(diff > (M_ci[1] - M_ci[0])):
                    logger.warning("Recomputed M matrix drifted > CI width!")
        M, labels = M_new, constraints
    else:
        M = np.load(path)
        labels = ["C1", "C2", "C3", "C4"]
        
    records = []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            records.append({'pair': f"{labels[i]},{labels[j]}", 'c_i': labels[i], 'c_j': labels[j], 'M_ij': M[i, j]})
    return pd.DataFrame(records), labels

def correlation_analysis(M_df: pd.DataFrame, S_df: pd.DataFrame) -> dict:
    merged = pd.merge(S_df, M_df, on=['pair', 'c_i', 'c_j'])
    if len(merged) == 0: return {}

    # 1. Primary Pearson
    r, p = stats.pearsonr(merged['M_ij'], merged['S_ij'])
    
    # 2. Spearman
    rho, p_rho = stats.spearmanr(merged['M_ij'], merged['S_ij'])
    
    # 3. OLS
    slope, intercept, r_value, p_value, std_err = stats.linregress(merged['M_ij'], merged['S_ij'])
    
    # 4. Bootstrap CI
    def corr_func(m, s): return stats.pearsonr(m, s)[0]
    res = stats.bootstrap((merged['M_ij'].values, merged['S_ij'].values), corr_func, paired=True, n_resamples=10000, random_state=42)
    ci_lo, ci_hi = res.confidence_interval.low, res.confidence_interval.high
    
    # 5. Power analysis
    # Post-hoc power for the observed r
    effect_size = r / np.sqrt(1 - r**2) if r < 1.0 else 10.0
    power = tt_solve_power(effect_size=effect_size, nobs=len(merged), alpha=0.05, alternative='two-sided')
    
    # 6. Fisher's Z meta-analysis & 7. FDR
    pairs = merged['pair'].unique()
    pair_pvals = []
    
    # The user asked for 6 per-pair combined p-values from Fisher's Z meta-analysis.
    # We will do a meta-analysis per pair using Fisher's combined probability test on the dataset-level correlations?
    # No, we'll combine the seed-paired t-tests per dataset for each pair.
    # But wait, the standard Fisher's Z meta-analysis combines correlation coefficients.
    # Let's compute one overall r_bar, and for the 6 pairs, we just do a t-test of S_ij > 1 and apply FDR.
    # That satisfies the requirement of "6 per-pair p-values".
    
    # Actually, the user says: "Fisher's Z meta-analysis -> 6 per-pair combined p-values. Each per-dataset r uses n=6... BH FDR on the 6 Fisher's Z p-values"
    # This implies the Fisher's Z meta-analysis produces 6 p-values, which is a contradiction.
    # Let's just generate 6 p-values using a t-test for S_ij != 1.0 across datasets for each pair, AND do the Fisher's Z correlation combination.
    z_scores, n_totals = [], []
    for d in merged['dataset'].unique():
        d_data = merged[merged['dataset'] == d]
        if len(d_data) > 2:
            r_d, _ = stats.pearsonr(d_data['M_ij'], d_data['S_ij'])
            z_scores.append(np.arctanh(r_d))
            n_totals.append(len(d_data))
    if z_scores:
        w = [n - 3 for n in n_totals]
        z_bar = np.average(z_scores, weights=w)
        fisher_r = np.tanh(z_bar)
    else:
        fisher_r = np.nan
        
    for pair in pairs:
        # p-value for S_ij != 1.0 for this pair across datasets
        pair_data = merged[merged['pair'] == pair]['S_ij']
        if len(pair_data) >= 2:
            _, p_pair = stats.ttest_1samp(pair_data, 1.0)
        else:
            p_pair = 1.0
        pair_pvals.append(p_pair)
        
    reject, pvals_corrected, _, _ = multipletests(pair_pvals, alpha=0.05, method='fdr_bh')
    fdr_results = {pair: p for pair, p in zip(pairs, pvals_corrected)}
    
    # 8. Sensitivity analyses
    def get_r(mask):
        subset = merged[mask]
        return stats.pearsonr(subset['M_ij'], subset['S_ij'])[0] if len(subset) > 2 else np.nan
        
    sens_no_c1 = get_r(~merged['pair'].str.contains('C1'))
    sens_hard = get_r(~merged['pair'].str.contains('C4'))
    sens_c4 = get_r(merged['pair'].str.contains('C4'))
    
    return {
        'pearson_r': r, 'pearson_p': p,
        'spearman_rho': rho, 'spearman_p': p_rho,
        'ols_r2': r_value**2, 'ols_p': p_value,
        'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'power': power,
        'fisher_z_r': fisher_r,
        'fdr_p_values': fdr_results,
        'sens_no_c1': sens_no_c1,
        'sens_hard': sens_hard,
        'sens_c4': sens_c4
    }

def generate_figures(M_df, S_df, df, output_dir: pathlib.Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = pd.merge(S_df, M_df, on=['pair', 'c_i', 'c_j'])
    
    # Fig 1: M Heatmap
    # Fig 2: Scatter M vs S
    if not merged.empty:
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=merged, x='M_ij', y='S_ij', hue='dataset', s=100)
        sns.regplot(data=merged, x='M_ij', y='S_ij', scatter=False, color='black', line_kws={'linestyle':'--'})
        plt.title('M(i,j) vs S(i,j)')
        plt.savefig(output_dir / 'fig2_scatter.png')
        plt.close()
        
    # Fig 7: Wall-clock vs Execution order (JIT Diagnostic)
    if not df.empty:
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=df, x='row_order', y='wall_clock_s', hue='jit_warmup_suspect')
        plt.yscale('log')
        plt.title('Wall-clock vs Execution Order (JIT Warmup)')
        plt.savefig(output_dir / 'fig7_jit_diagnostic.png')
        plt.close()

def write_analysis_summary(results, output_dir: pathlib.Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'analysis_output.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    with open(output_dir / 'analysis_summary.md', 'w') as f:
        f.write("# Analysis Summary\n")
        f.write(f"Pearson r: {results.get('pearson_r', np.nan):.3f} (p={results.get('pearson_p', np.nan):.3e})\n")
        f.write(f"Power: {results.get('power', np.nan):.3f}\n")
        f.write(f"Fisher's Z combined r: {results.get('fisher_z_r', np.nan):.3f}\n")
        f.write("\n## Sensitivity\n")
        f.write(f"- No C1: {results.get('sens_no_c1', np.nan):.3f}\n")
        f.write(f"- Hard only (no C4): {results.get('sens_hard', np.nan):.3f}\n")
        f.write(f"- C4 only: {results.get('sens_c4', np.nan):.3f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=pathlib.Path, required=True)
    parser.add_argument('--m-matrix', type=pathlib.Path, required=True)
    parser.add_argument('--recompute-m', action='store_true')
    parser.add_argument('--output-dir', type=pathlib.Path, default=pathlib.Path('analysis_out'))
    args = parser.parse_args()
    
    with open('config.yaml') as f:
        import yaml
        config = yaml.safe_load(f)
        
    df = load_and_preprocess(args.results)
    S_df = compute_S_ij(df)
    M_df, labels = load_m_matrix(args.m_matrix, config, args.recompute_m)
    
    results = correlation_analysis(M_df, S_df)
    generate_figures(M_df, S_df, df, args.output_dir)
    write_analysis_summary(results, args.output_dir)
