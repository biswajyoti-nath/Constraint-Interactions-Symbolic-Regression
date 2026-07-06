import random
import yaml
from sympy import sympify, SympifyError
import pathlib

class GrammarGenerator:
    """Generates random mathematical expressions based on a specified grammar.

    This class parses the central `config.yaml` to extract the allowed variables,
    constants, binary operators, and unary operators. It then recursively builds
    mathematical expressions (ASTs) up to a maximum depth.

    Attributes:
        config (dict): The loaded grammar configuration dictionary.
        binary_ops (list[str]): List of allowed binary operators (e.g., '+', '-').
        unary_ops (list[str]): List of allowed unary operators (e.g., 'sin', 'cos').
        variables (list[str]): List of allowed terminal variables (e.g., 'x', 'y').
        constants_min (int): Minimum integer value for random constants.
        constants_max (int): Maximum integer value for random constants.
        max_depth (int): Maximum allowed depth for the generated expression tree.
    """

    def __init__(self, config_path="config.yaml"):
        """Initialize the GrammarGenerator.

        Args:
            config_path (str): The file path to the YAML configuration file.
                Defaults to "config.yaml".
        """
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["grammar"]
        
        self.binary_ops = self.config["operators"]["binary"]
        self.unary_ops = self.config["operators"]["unary"]
        self.variables = self.config["variables"]
        self.constants_min = self.config["constants"]["min"]
        self.constants_max = self.config["constants"]["max"]
        self.max_depth = self.config["max_depth"]

    def _generate_recursive(self, depth):
        """Recursively generate an expression string up to the given depth.

        At depth 0, or with a 20% probability at higher depths, this returns
        a leaf node (variable or constant). Otherwise, it randomly chooses
        between a unary or binary operator and recurses.

        Args:
            depth (int): The remaining depth allowed for the expression tree.

        Returns:
            str: A string representation of the generated mathematical expression.
        """
        # Base case: if we hit depth 0, we must return a leaf (variable or constant)
        if depth == 0:
            return self._random_leaf()
        
        # Determine whether to branch or return a leaf earlier
        # To avoid overwhelmingly bushy trees, we introduce a chance to return a leaf
        # proportional to the depth (optional logic, but here we just give it a 20% chance)
        if random.random() < 0.2:
            return self._random_leaf()
            
        choice = random.choice(["binary", "unary"])
        if choice == "binary" and len(self.binary_ops) > 0:
            op = random.choice(self.binary_ops)
            left = self._generate_recursive(depth - 1)
            right = self._generate_recursive(depth - 1)
            return f"({left} {op} {right})"
        elif choice == "unary" and len(self.unary_ops) > 0:
            op = random.choice(self.unary_ops)
            inner = self._generate_recursive(depth - 1)
            return f"{op}({inner})"
        else:
            # Fallback if ops list is empty
            return self._random_leaf()

    def _random_leaf(self):
        """Generate a random leaf node (variable or constant).

        Returns:
            str: A string representing either a variable name or a numeric constant.
        """
        if random.random() < 0.5 and len(self.variables) > 0:
            return random.choice(self.variables)
        else:
            # Generate random constant, avoid zero or negatives if not allowed, but here we allow min to max
            val = random.randint(self.constants_min, self.constants_max)
            # 50% chance of being negative if the range is strictly positive (for variety)
            if val > 0 and random.random() < 0.5:
                val = -val
            return str(val)

    def generate(self, max_depth=None):
        """Generate a random mathematical expression as a string.

        Args:
            max_depth (int, optional): The maximum depth of the expression tree.
                If None, uses the max_depth specified in the configuration.

        Returns:
            str: The randomly generated mathematical expression.
        """
        if max_depth is None:
            max_depth = self.max_depth
            
        return self._generate_recursive(max_depth)

    def generate_sympy(self, max_depth=None, max_retries=10):
        """Generate a valid SymPy expression.
        
        Repeatedly calls `generate()` until a string is produced that successfully
        parses into a SymPy expression without syntax or mathematical domain errors.

        Args:
            max_depth (int, optional): The maximum depth of the expression tree.
            max_retries (int): The maximum number of generation attempts before giving up.
                Defaults to 10.

        Returns:
            sympy.Expr or None: A parsed SymPy expression object, or None if all
                retries failed.
        """
        for _ in range(max_retries):
            expr_str = self.generate(max_depth)
            try:
                # sympify evaluates the string safely
                expr_sympy = sympify(expr_str, evaluate=False)
                # Ensure no division by zero or immediate nan eval happens if we simplify
                # evaluate=False keeps the structure intact.
                return expr_sympy
            except (SympifyError, TypeError, ValueError, ZeroDivisionError):
                continue
        return None

if __name__ == "__main__":
    # Test the generator
    generator = GrammarGenerator()
    print("Generating sample expressions:")
    for i in range(5):
        expr = generator.generate_sympy(max_depth=4)
        print(f"Sample {i+1}: {expr}")
