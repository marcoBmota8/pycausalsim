"""
Structural Causal Models (SCM) for PyCausalSim.

Implements SCMs that can:
- Fit causal mechanisms from data
- Generate counterfactuals
- Simulate interventions
- Sample from observational and interventional distributions
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import warnings


class StructuralCausalModel:
    """
    Structural Causal Model for representing and simulating causal systems.
    
    An SCM consists of:
    - A set of endogenous variables
    - A set of exogenous (noise) variables
    - A set of structural equations (causal mechanisms)
    
    Parameters
    ----------
    graph : dict, optional
        Causal graph as adjacency list: {child: [parents]}
    noise_type : str
        Type of noise: 'gaussian', 'uniform', 'empirical'
        
    Examples
    --------
    >>> scm = StructuralCausalModel()
    >>> scm.fit(data)
    >>> counterfactuals = scm.counterfactual(
    ...     intervention={'price': 0.9 * current_price},
    ...     evidence={'traffic_source': 'paid'}
    ... )
    """
    
    def __init__(
        self,
        graph: Optional[Dict[str, List[str]]] = None,
        noise_type: str = 'gaussian'
    ):
        self.graph = graph or {}
        self.noise_type = noise_type
        
        # Fitted components
        self._mechanisms = {}  # var -> fitted mechanism
        self._noise_params = {}  # var -> noise parameters
        self._variable_order = []  # Topological order
        self._is_fitted = False
        self._data_stats = {}  # Variable statistics
    
    @property
    def variables(self) -> List[str]:
        """Return all variables in the model."""
        return list(self.graph.keys())
    
    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted
    
    def fit(self, data: pd.DataFrame) -> 'StructuralCausalModel':
        """
        Fit structural equations from data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Observational data
            
        Returns
        -------
        self
        """
        # Store data statistics
        for col in data.columns:
            self._data_stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max()
            }
        
        # Ensure all variables are in graph
        for col in data.columns:
            if col not in self.graph:
                self.graph[col] = []
        
        # Get topological order
        self._variable_order = self._topological_sort()
        
        # Fit mechanism for each variable
        for var in self._variable_order:
            parents = self.graph.get(var, [])
            mechanism, noise = self._fit_mechanism(data, var, parents)
            self._mechanisms[var] = mechanism
            self._noise_params[var] = noise
        
        self._is_fitted = True
        return self
    
    def _topological_sort(self) -> List[str]:
        """Return variables in topological order."""
        visited = set()
        order = []
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for parent in self.graph.get(node, []):
                dfs(parent)
            order.append(node)
        
        for var in self.graph:
            dfs(var)
        
        return order
    
    def _fit_mechanism(
        self,
        data: pd.DataFrame,
        var: str,
        parents: List[str]
    ) -> Tuple[Any, Dict]:
        """Fit causal mechanism for a variable."""
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        
        if not parents:
            # Root node: fit distribution
            values = data[var].values
            mechanism = {
                'type': 'root',
                'values': values, # If root, mechanism values are the noise values
                'mean': np.mean(values),
                'std': np.std(values)
            }
            noise = {
                'type': self.noise_type,
                'mean': np.mean(values),
                'std': np.std(values)
            }
            return mechanism, noise
        
        # Non-root: fit regression
        X = data[parents].values
        y = data[var].values
        
        # Try non-linear model first, fall back to linear
        try:
            model = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                random_state=42
            )
            model.fit(X, y)
            residuals = y - model.predict(X)
        except Exception:
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            residuals = y - model.predict(X)
        
        mechanism = {
            'type': 'regression',
            'model': model,
            'parents': parents
        }
        
        noise = {
            'type': self.noise_type,
            'values': residuals,
            'mean': np.mean(residuals),
            'std': np.std(residuals)
        }
        
        return mechanism, noise
    
    def predict(
        self,
        data: pd.DataFrame,
        target: str
    ) -> np.ndarray:
        """
        Predict target variable given other variables.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with parent variables
        target : str
            Variable to predict
            
        Returns
        -------
        np.ndarray
            Predicted values
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        mechanism = self._mechanisms.get(target)
        if mechanism is None:
            raise ValueError(f"Unknown variable: {target}")
        
        if mechanism['type'] == 'root':
            return np.full(len(data), mechanism['mean'])
        
        parents = mechanism['parents']
        X = data[parents].values
        return mechanism['model'].predict(X)
    
    def sample(
        self,
        n_samples: int = 1000,
        interventions: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Sample from the SCM (observational or interventional distribution).
        
        Parameters
        ----------
        n_samples : int
            Number of samples
        interventions : dict, optional
            Variables to intervene on: {var: value}
            
        Returns
        -------
        pd.DataFrame
            Sampled data
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        interventions = interventions or {}
        samples = {}
        
        # Sample in topological order
        for var in self._variable_order:
            if var in interventions:
                # Intervention: set to fixed value
                samples[var] = np.full(n_samples, interventions[var])
            else:
                mechanism = self._mechanisms[var]
                noise_params = self._noise_params[var]
                
                if mechanism['type'] == 'root':
                    # Root node: sample from marginal
                    if self.noise_type == 'gaussian':
                        samples[var] = np.random.normal(
                            mechanism['mean'],
                            mechanism['std'],
                            n_samples
                        )
                    elif self.noise_type == 'uniform':
                        samples[var] = np.random.uniform(
                            mechanism['mean'] - mechanism['std'] * 1.7,
                            mechanism['mean'] + mechanism['std'] * 1.7,
                            n_samples
                        )
                    elif self.noise_type == 'empirical': # If empirical, sample from bootstrap of the noise values. This is the same approach as in DoWhy.
                        samples[var] = np.random.choice(mechanism['values'], n_samples, replace=True)
                    else:
                        raise ValueError(f"Unknown noise type: {self.noise_type}")
                else:
                    # Non-root: apply mechanism + noise
                    parents = mechanism['parents']
                    X = np.column_stack([samples[p] for p in parents])
                    
                    predictions = mechanism['model'].predict(X)
                    
                    if self.noise_type == 'gaussian':
                        noise = np.random.normal(0, noise_params['std'], n_samples)
                    elif self.noise_type == 'uniform':
                        noise = np.random.uniform(
                            -noise_params['std'] * 1.7,
                            noise_params['std'] * 1.7,
                            n_samples
                        )
                    elif self.noise_type == 'empirical': # If empirical, sample from bootstrap of the noise values. This is the same approach as in DoWhy.
                        noise = np.random.choice(noise_params['values'], n_samples, replace=True) 
                    else:
                        raise ValueError(f"Unknown noise type: {self.noise_type}")
                    
                    samples[var] = predictions + noise
        
        return pd.DataFrame(samples)
    
    def counterfactual(
        self,
        intervention: Dict[str, float],
        evidence: Optional[Dict[str, Any]] = None,
        data: Optional[pd.DataFrame] = None,
        noise_scale: float = 0.0
    ) -> pd.DataFrame:
        """
        Generate counterfactual outcomes.
        
        Answers: "What would Y have been if we had set X to x?"
        
        Parameters
        ----------
        intervention : dict
            Interventions to apply: {var: value}
        evidence : dict, optional
            Conditions to filter data on
        data : pd.DataFrame, optional
            Data to generate counterfactuals for
        noise_scale : float
            Scale of additional noise (0 = deterministic)
            
        Returns
        -------
        pd.DataFrame
            Counterfactual data
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        if data is None:
            # Generate synthetic data
            data = self.sample(1000)
        
        # Apply evidence filter
        if evidence:
            mask = np.ones(len(data), dtype=bool)
            for var, val in evidence.items():
                mask &= (data[var] == val)
            data = data[mask].copy()
        
        # Generate counterfactuals
        cf_data = data.copy()
        
        # Apply interventions
        for var, value in intervention.items():
            cf_data[var] = value
        
        # Propagate through descendants in topological order
        intervened_vars = set(intervention.keys())
        
        for var in self._variable_order:
            if var in intervened_vars:
                continue
            
            mechanism = self._mechanisms[var]
            
            if mechanism['type'] == 'root':
                continue  # Keep original values for roots
            
            # Check if any parent was intervened on
            parents = mechanism['parents']
            affected = any(p in intervened_vars or self._is_descendant(p, intervened_vars)
                         for p in parents)
            
            if affected:
                # Recompute using new parent values
                X = cf_data[parents].values
                predictions = mechanism['model'].predict(X)
                
                if noise_scale > 0:
                    noise = np.random.normal(0, self._noise_params[var]['std'] * noise_scale, len(data))
                    predictions = predictions + noise
                
                cf_data[var] = predictions
        
        return cf_data
    
    def _is_descendant(self, var: str, ancestors: set) -> bool:
        """Check if var is a descendant of any variable in ancestors."""
        parents = self.graph.get(var, [])
        for p in parents:
            if p in ancestors or self._is_descendant(p, ancestors):
                return True
        return False
    
    def intervene(
        self,
        variable: str,
        value: float
    ) -> 'StructuralCausalModel':
        """
        Return a new SCM with an intervention applied.
        
        The do-operator: do(X = x)
        
        Parameters
        ----------
        variable : str
            Variable to intervene on
        value : float
            Value to set
            
        Returns
        -------
        StructuralCausalModel
            New SCM with intervention
        """
        # Create copy with modified graph (remove incoming edges)
        new_graph = {v: list(p) for v, p in self.graph.items()}
        new_graph[variable] = []  # Remove all parents
        
        new_scm = StructuralCausalModel(graph=new_graph, noise_type=self.noise_type)
        
        # Copy mechanisms except for intervened variable
        new_scm._mechanisms = dict(self._mechanisms)
        new_scm._noise_params = dict(self._noise_params)
        new_scm._data_stats = dict(self._data_stats)
        
        # Set intervened variable to constant
        new_scm._mechanisms[variable] = {
            'type': 'root',
            'mean': value,
            'std': 0
        }
        new_scm._noise_params[variable] = {'type': 'constant', 'std': 0}
        
        new_scm._variable_order = new_scm._topological_sort()
        new_scm._is_fitted = True
        
        return new_scm
    
    def ate(
        self,
        treatment: str,
        outcome: str,
        treatment_value: float,
        control_value: float,
        n_samples: int = 10000
    ) -> float:
        """
        Compute Average Treatment Effect.
        
        ATE = E[Y | do(T=t1)] - E[Y | do(T=t0)]
        
        Parameters
        ----------
        treatment : str
            Treatment variable
        outcome : str
            Outcome variable
        treatment_value : float
            Treatment condition value
        control_value : float
            Control condition value
        n_samples : int
            Number of samples for estimation
            
        Returns
        -------
        float
            Average treatment effect
        """
        # Sample under treatment
        treated = self.sample(n_samples, interventions={treatment: treatment_value})
        
        # Sample under control
        control = self.sample(n_samples, interventions={treatment: control_value})
        
        return treated[outcome].mean() - control[outcome].mean()
    
    def get_mechanism(self, variable: str) -> Dict:
        """Get the fitted mechanism for a variable."""
        return self._mechanisms.get(variable)
    
    def summary(self) -> str:
        """Return model summary."""
        lines = [
            "Structural Causal Model",
            "=" * 50,
            f"Variables: {len(self.variables)}",
            f"Fitted: {self._is_fitted}",
            "",
            "Causal Structure:",
        ]
        
        for var in self._variable_order:
            parents = self.graph.get(var, [])
            if parents:
                lines.append(f"  {var} <- {', '.join(parents)}")
            else:
                lines.append(f"  {var} (root)")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"StructuralCausalModel(variables={len(self.variables)}, fitted={self._is_fitted})"


class Variable:
    """
    Represents a variable in an SCM with explicit distribution.
    
    Examples
    --------
    >>> X = Variable("X", stats.norm(loc=0, scale=1))
    >>> Y = Variable("Y", lambda x: 2*x + stats.norm(0, 0.1).rvs())
    """
    
    def __init__(
        self,
        name: str,
        distribution: Any = None,
        parents: Optional[List['Variable']] = None,
        mechanism: Optional[callable] = None
    ):
        self.name = name
        self.distribution = distribution
        self.parents = parents or []
        self.mechanism = mechanism
    
    def sample(self, parent_values: Optional[Dict[str, np.ndarray]] = None, n: int = 1) -> np.ndarray:
        """Sample from the variable."""
        if self.mechanism and parent_values:
            # Apply mechanism with parent values
            return self.mechanism(parent_values)
        elif self.distribution:
            if hasattr(self.distribution, 'rvs'):
                return self.distribution.rvs(size=n)
            elif callable(self.distribution):
                return np.array([self.distribution() for _ in range(n)])
        
        return np.zeros(n)
    
    def __repr__(self) -> str:
        return f"Variable('{self.name}')"


# Alias for compatibility
SCM = StructuralCausalModel
