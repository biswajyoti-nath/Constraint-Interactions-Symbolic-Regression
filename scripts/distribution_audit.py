import pathlib
import sys
import yaml
import numpy as np
import pandas as pd
import sympy
import json
from collections import defaultdict
from pysr import PySRRegressor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.experiments import LOADERS, DATASET_FEATURE_NAMES, build_pysr_kwargs
from src.constraints import build_constraints
from src.expr_generator import GrammarGenerator
from src.metric import DensityEstimator

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # We enforce C1a only as per config
    config["constraints"]["structural"]["enforce_c1a_only"] = True
    constraints = build_constraints(config)
    labels = ["C1a", "C2", "C3", "C4"]

    # 1. Compute Monte Carlo rho (using a smaller N=1000 for speed in the audit script, 
    # since we just need order-of-magnitude estimates)
    print("Computing Monte Carlo densities (N=1000)...")
    gen = GrammarGenerator("config.yaml")
    estimator = DensityEstimator(gen, max_depth=config["grammar"]["max_depth"])
    mc_result = estimator.estimate(constraints, N=1000)
    mc_rho = mc_result.rho_i

    # 2. Run PySR Baseline (1 iteration) to get Generation 1 proxy
    dataset_name = "polynomial"
    seeds = [42, 123, 456, 789, 1024]
    all_hof_exprs = []

    print(f"\nRunning 1-iteration PySR on '{dataset_name}' for {len(seeds)} seeds...")
    for seed in seeds:
        X_train, y_train, _, _, _ = LOADERS[dataset_name](seed=seed)
        features = DATASET_FEATURE_NAMES[dataset_name]
        
        pysr_kwargs = build_pysr_kwargs(config, [])
        pysr_kwargs.update({
            "niterations": 1,
            "deterministic": True,
            "parallelism": "serial",
            "random_state": seed,
            "verbosity": 0,
            "progress": False,
            "temp_equation_file": True
        })
        
        # Remove early stopping for this audit to ensure it runs a full iteration
        if "early_stop_condition" in pysr_kwargs:
            del pysr_kwargs["early_stop_condition"]
            
        model = PySRRegressor(**pysr_kwargs)
        model.fit(X_train, y_train, variable_names=features)
        
        hof = model.equations_
        if isinstance(hof, list):
            hof = hof[0]
            
        for _, row in hof.iterrows():
            try:
                expr = row.get("sympy_format", sympy.sympify(row["equation"]))
                all_hof_exprs.append(expr)
            except Exception:
                pass

    n_hof = len(all_hof_exprs)
    print(f"\nExtracted {n_hof} Hall-of-Fame expressions across all seeds.")

    # 3. Compute empirical rho from HoF expressions
    empirical_counts = np.zeros(len(constraints))
    for expr in all_hof_exprs:
        for i, c_fn in enumerate(constraints):
            try:
                if c_fn(expr):
                    empirical_counts[i] += 1
            except Exception:
                pass
                
    empirical_rho = empirical_counts / n_hof

    # 4. Compare and output
    results = []
    print(f"\n{'Constraint':<10} | {'MC Rho':<10} | {'Empirical Rho':<15} | {'Ratio (Emp/MC)':<15}")
    print("-" * 55)
    for i, label in enumerate(labels):
        mc_val = mc_rho[i]
        emp_val = empirical_rho[i]
        ratio = emp_val / mc_val if mc_val > 0 else np.nan
        
        print(f"{label:<10} | {mc_val:<10.4f} | {emp_val:<15.4f} | {ratio:<15.2f}x")
        results.append({
            "constraint": label,
            "mc_rho": float(mc_val),
            "empirical_rho": float(emp_val),
            "ratio": float(ratio)
        })

    out_file = pathlib.Path("data/distribution_audit.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "seeds": seeds,
            "n_hof_expressions": n_hof,
            "results": results
        }, f, indent=2)
        
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
