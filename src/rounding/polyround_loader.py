#!/usr/bin/env python3
"""
Native Python support for PolyRound rounding in cu_sample.

This module provides utilities to:
1. Load rounding transforms computed by PolyRound
2. Verify transform files exist and are valid
3. Convert between different transform formats
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


class RoundingTransform:
    """
    Rounding transform data structure compatible with PolyRound output.
    
    The transform maps between original variables v and rounded variables y:
        v = shift + T * y
    
    Where:
        - A, b define the rounded polytope: A * y <= b
        - start is a feasible starting point in y-space
        - T is the transformation matrix
        - shift is the translation vector
    """
    
    def __init__(self, model_dir: str, model_name: str):
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        
        # Look for rounding subdirectory
        self.rounding_dir = self.model_dir / "rounding"
        if not self.rounding_dir.exists():
            # Fallback to root if not found
            self.rounding_dir = self.model_dir
            
        self.prefix = self.rounding_dir / f"{model_name}_rounding"
        
        self.A: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.start: Optional[np.ndarray] = None
        self.T: Optional[np.ndarray] = None
        self.shift: Optional[np.ndarray] = None
        
        # Additional legacy format support
        self.N: Optional[np.ndarray] = None  # nullspace (T in some formats)
        self.v0: Optional[np.ndarray] = None  # particular solution (shift)
        
    def verify_files(self) -> Dict[str, bool]:
        """Check which rounding files exist."""
        required_files = {
            "A": f"{self.prefix}_A.csv",
            "b": f"{self.prefix}_b.csv", 
            "start": f"{self.prefix}_start.csv",
            "T": f"{self.prefix}_T.csv",
            "shift": f"{self.prefix}_shift.csv",
        }
        
        status = {}
        for key, path in required_files.items():
            status[key] = Path(path).exists()
            
        return status
    
    def load(self, verbose: bool = True) -> bool:
        """
        Load rounding transform from CSV files.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load required files
            self.A = pd.read_csv(f"{self.prefix}_A.csv", header=None).values
            self.b = pd.read_csv(f"{self.prefix}_b.csv", header=None).values.flatten()
            self.start = pd.read_csv(f"{self.prefix}_start.csv", header=None).values.flatten()
            self.T = pd.read_csv(f"{self.prefix}_T.csv", header=None).values
            self.shift = pd.read_csv(f"{self.prefix}_shift.csv", header=None).values.flatten()
            
            if verbose:
                print(f"Loaded rounding transform for {self.model_name}")
                print(f"  A: {self.A.shape}")
                print(f"  b: {self.b.shape}")
                print(f"  start: {self.start.shape}")
                print(f"  T: {self.T.shape}")
                print(f"  shift: {self.shift.shape}")
                print(f"  Original dim: {self.T.shape[0]}")
                print(f"  Reduced dim: {self.T.shape[1]}")
                
            # Also load legacy format if available (N, v0)
            n_path = f"{self.prefix}_N.csv"
            if Path(n_path).exists():
                self.N = pd.read_csv(n_path, header=None).values
                if verbose:
                    print(f"  N (legacy): {self.N.shape}")
                    
            v0_path = f"{self.prefix}_v0.csv"
            if Path(v0_path).exists():
                self.v0 = pd.read_csv(v0_path, header=None).values.flatten()
                if verbose:
                    print(f"  v0 (legacy): {self.v0.shape}")
            
            return True
            
        except FileNotFoundError as e:
            if verbose:
                print(f"Error loading rounding files: {e}")
            return False
    
    def y_to_v(self, y: np.ndarray) -> np.ndarray:
        """Map from rounded y-space to original v-space."""
        if self.T is None or self.shift is None:
            raise ValueError("Transform not loaded. Call load() first.")
        return self.shift + self.T @ y
    
    def verify_feasibility(self, y: np.ndarray, tol: float = 1e-6) -> Tuple[bool, float]:
        """
        Check if y is feasible in the rounded polytope.
        
        Returns:
            (is_feasible, max_violation)
        """
        if self.A is None or self.b is None:
            raise ValueError("Polytope not loaded. Call load() first.")
            
        slack = self.b - self.A @ y
        max_violation = -np.min(slack)
        is_feasible = max_violation <= tol
        
        return is_feasible, max_violation
    
    def get_dimensions(self) -> Dict[str, int]:
        """Get dimension information."""
        dims = {}
        if self.T is not None:
            dims['original'] = self.T.shape[0]
            dims['reduced'] = self.T.shape[1]
        if self.A is not None:
            dims['inequalities'] = self.A.shape[0]
        return dims


def load_rounding_transform(model_name: str, model_dir: Optional[str] = None, 
                           verbose: bool = True) -> Optional[RoundingTransform]:
    """
    Convenience function to load a rounding transform.
    
    Args:
        model_name: Name of the model (e.g., "iAB_RBC_283")
        model_dir: Directory containing model files. If None, defaults to ../models/{model_name}
        verbose: Print loading information
        
    Returns:
        RoundingTransform object or None if loading failed
    """
    if model_dir is None:
        # Default: models directory relative to this script
        script_dir = Path(__file__).parent.parent.parent
        model_dir = script_dir / "models" / model_name
        
    rt = RoundingTransform(str(model_dir), model_name)
    
    if verbose:
        status = rt.verify_files()
        missing = [k for k, v in status.items() if not v]
        if missing:
            print(f"Warning: Missing files: {missing}")
    
    if rt.load(verbose=verbose):
        return rt
    else:
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python polyround_loader.py <model_name> [model_dir]")
        sys.exit(1)
        
    model_name = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    rt = load_rounding_transform(model_name, model_dir, verbose=True)
    
    if rt is not None:
        print("\nTransform loaded successfully!")
        print(f"Dimensions: {rt.get_dimensions()}")
        
        # Test feasibility of start point
        feasible, violation = rt.verify_feasibility(rt.start)
        print(f"\nStart point feasibility: {feasible}")
        if not feasible:
            print(f"  Max violation: {violation}")
    else:
        print("Failed to load rounding transform")
        sys.exit(1)
