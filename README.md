# Constraint-Interaction Metric for Symbolic Regression

## What This Does
This project provides a comprehensive framework to quantify how different search constraints interact in Symbolic Regression (SR). Specifically, it calculates a **constraint-interaction metric** $M(i,j)$ to determine if pairs of constraints (like depth limits, operator whitelists, or dimensional constraints) act synergistically ($M>1$), redundantly ($M<1$), or independently ($M \approx 1$).

This is part of a TIH IITG Summer Research project by Biswajyoti Nath and Subinoy Nath.

## Quick Start
Get up and running with the environment in under 5 minutes.

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repo-url>
   cd Constraint-Interaction-SymREG
   ```

2. **Set up the Python environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Install Julia (Required by PySR)**:
   PySR will automatically download its internal Julia dependencies on the first run. You can initialize it by running:
   ```bash
   python3 -c "import pysr; pysr.install()"
   ```

4. **Test the expression generator**:
   ```bash
   python3 src/expr_generator.py
   ```

## Project Structure

```text
Constraint-Interaction-SymREG/
├── config.yaml            # Central configuration for constraints, PySR, and data
├── requirements.txt       # Pinned Python dependencies
├── docs/                  # Documentation
│   └── research_plan/     # LaTeX sources and PDF of the research plan
├── src/                   # Source code
│   └── expr_generator.py  # Grammar-based random expression generator
├── data/                  # Output directory for generated datasets and results
├── tests/                 # Unit tests
└── figures/               # Output directory for generated plots
```

## Key Concepts

- **Grammar-Based Expression Generator**: We use a recursive generator that constructs abstract syntax trees (ASTs) strictly adhering to a defined set of binary/unary operators and terminal variables/constants.
- **Monte Carlo Density Estimation**: We generate hundreds of thousands of random valid expressions to estimate $\rho(C)$, which is the baseline probability that an expression satisfies constraint $C$.
- **Interaction Coefficient $M(i,j)$**: Calculated as $\frac{\rho(C_i \wedge C_j)}{\rho(C_i)\rho(C_j)}$. It shows how combining two constraints restricts the search space compared to what statistical independence would predict.

## Common Tasks

### Configuring the Grammar
All parameters are controlled in `config.yaml`. To change the allowed variables or operators, edit the `grammar` section:
```yaml
grammar:
  variables: ["x", "y", "z"]
  operators:
    binary: ["+", "-", "*", "/"]
```

### Generating Random Expressions
You can generate random mathematical expressions using the `GrammarGenerator` class:
```python
from src.expr_generator import GrammarGenerator

generator = GrammarGenerator()
expr = generator.generate_sympy(max_depth=4)
print(expr)
```

## Troubleshooting

- **Julia not found / PySR initialization fails**: Ensure that `pysr.install()` completes successfully. If you have a system-wide Julia installation, it might conflict with PySR's managed version. Make sure to run `python3 -c "import pysr; pysr.install()"` in the active virtual environment.
- **SympifyError during generation**: The expression generator occasionally produces syntactically invalid strings or math domain errors (like division by zero). The `generate_sympy` method automatically retries up to `max_retries` times to avoid these errors.
