import numpy as np
from scipy import stats
import pandas as pd
from typing import Dict, Tuple, List, Union
import warnings

class ABTestAnalyzer:
    """
    A comprehensive tool for analyzing AB test results with multiple statistical approaches.
    """
    
    def __init__(self, control_data: Dict[str, np.ndarray], 
                 treatment_data: Dict[str, np.ndarray],
                 metric_types: Dict[str, str] = None):
        """
        Initialize with control and treatment data for multiple metrics.
        
        Parameters:
        control_data: Dictionary of metric names to numpy arrays for control group
        treatment_data: Dictionary of metric names to numpy arrays for treatment group
        metric_types: Dictionary of metric types (percentage, currency, count, etc.)
        """
        self.control = control_data
        self.treatment = treatment_data
        self.metric_types = metric_types
        self.metrics = list(control_data.keys())
        
    def run_full_analysis(self, confidence_level: float = 0.95) -> Dict:
        """Run all available analyses and return comprehensive results."""
        results = {}
        
        for metric in self.metrics:
            control_data = self.control[metric]
            treatment_data = self.treatment[metric]
            
            results[metric] = {
                'basic_stats': self._calculate_basic_stats(control_data, treatment_data),
                'confidence_intervals': self._calculate_confidence_intervals(
                    control_data, treatment_data, confidence_level),
                'hypothesis_test': self._run_hypothesis_test(control_data, treatment_data),
                'effect_size': self._calculate_effect_size(control_data, treatment_data),
                'power_analysis': self._perform_power_analysis(control_data, treatment_data),
                'sequential_analysis': self._perform_sequential_analysis(
                    control_data, treatment_data),
                'robustness_checks': self._perform_robustness_checks(
                    control_data, treatment_data)
            }
            
        return results
    
    def _calculate_basic_stats(self, control: np.ndarray, 
                             treatment: np.ndarray) -> Dict:
        """Calculate basic statistical measures."""
        return {
            'control': {
                'mean': np.mean(control),
                'median': np.median(control),
                'std': np.std(control),
                'sample_size': len(control)
            },
            'treatment': {
                'mean': np.mean(treatment),
                'median': np.median(treatment),
                'std': np.std(treatment),
                'sample_size': len(treatment)
            },
            'relative_difference': (np.mean(treatment) - np.mean(control)) / np.mean(control)
        }
    
    def _calculate_confidence_intervals(self, control: np.ndarray, 
                                     treatment: np.ndarray,
                                     confidence_level: float) -> Dict:
        """Calculate confidence intervals for difference between groups."""
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        
        # Calculate pooled standard error
        n1, n2 = len(control), len(treatment)
        pooled_se = np.sqrt(np.var(control)/n1 + np.var(treatment)/n2)
        
        # Calculate confidence interval for difference
        t_stat = stats.t.ppf((1 + confidence_level)/2, n1 + n2 - 2)
        margin_of_error = t_stat * pooled_se
        
        difference = treatment_mean - control_mean
        
        return {
            'difference': difference,
            'ci_lower': difference - margin_of_error,
            'ci_upper': difference + margin_of_error,
            'relative_difference': difference / control_mean
        }
    
    def _run_hypothesis_test(self, control: np.ndarray, 
                           treatment: np.ndarray) -> Dict:
        """Perform statistical hypothesis testing."""
        # T-test
        t_stat, p_value = stats.ttest_ind(treatment, control)
        
        # Mann-Whitney U test (non-parametric)
        mw_stat, mw_p_value = stats.mannwhitneyu(treatment, control, alternative='two-sided')
        
        return {
            't_test': {'statistic': t_stat, 'p_value': p_value},
            'mann_whitney': {'statistic': mw_stat, 'p_value': mw_p_value}
        }
    
    def _calculate_effect_size(self, control: np.ndarray, 
                             treatment: np.ndarray) -> Dict:
        """Calculate effect size metrics."""
        # Cohen's d
        pooled_std = np.sqrt((np.var(control) + np.var(treatment)) / 2)
        cohens_d = (np.mean(treatment) - np.mean(control)) / pooled_std
        
        return {
            'cohens_d': cohens_d,
            'interpretation': self._interpret_cohens_d(cohens_d)
        }
    
    def _perform_power_analysis(self, control: np.ndarray, 
                              treatment: np.ndarray) -> Dict:
        """Perform statistical power analysis."""
        effect_size = self._calculate_effect_size(control, treatment)['cohens_d']
        n1, n2 = len(control), len(treatment)
        
        # Calculate observed power
        from scipy.stats import norm
        alpha = 0.05  # conventional significance level
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = effect_size * np.sqrt(n1*n2/(n1 + n2)) - z_alpha
        power = norm.cdf(z_beta)
        
        return {
            'observed_power': power,
            'is_well_powered': power >= 0.8,  # conventional threshold
            'sample_size_recommendation': self._recommend_sample_size(effect_size)
        }
    
    def _perform_sequential_analysis(self, control: np.ndarray, 
                                   treatment: np.ndarray) -> Dict:
        """Perform sequential analysis to check for early stopping bias."""
        sample_proportions = np.linspace(0.1, 1.0, 10)
        sequential_results = []
        
        for prop in sample_proportions:
            n_samples = int(len(control) * prop)
            _, p_value = stats.ttest_ind(
                control[:n_samples], 
                treatment[:n_samples]
            )
            sequential_results.append({
                'proportion': prop,
                'p_value': p_value
            })
            
        return {
            'sequential_p_values': sequential_results,
            'potential_early_stopping_bias': self._check_early_stopping_bias(
                sequential_results)
        }
    
    def _perform_robustness_checks(self, control: np.ndarray, 
                                 treatment: np.ndarray) -> Dict:
        """Perform various robustness checks."""
        return {
            'normality_test': self._test_normality(control, treatment),
            'variance_test': self._test_variance_equality(control, treatment),
            'outlier_analysis': self._analyze_outliers(control, treatment),
            'sample_size_adequacy': self._check_sample_size_adequacy(
                control, treatment)
        }
    
    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size."""
        if abs(d) < 0.2:
            return "negligible effect"
        elif abs(d) < 0.5:
            return "small effect"
        elif abs(d) < 0.8:
            return "medium effect"
        else:
            return "large effect"
    
    @staticmethod
    def _recommend_sample_size(effect_size: float, 
                             desired_power: float = 0.8) -> int:
        """Recommend sample size based on effect size."""
        from scipy.stats import norm
        alpha = 0.05
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(desired_power)
        
        n = 2 * ((z_alpha + z_beta)/effect_size)**2
        return int(np.ceil(n))
    
    @staticmethod
    def _check_early_stopping_bias(sequential_results: List[Dict]) -> bool:
        """Check for potential early stopping bias in sequential analysis."""
        p_values = [result['p_value'] for result in sequential_results]
        return any(p < 0.05 for p in p_values[:-1]) and p_values[-1] >= 0.05
    
    @staticmethod
    def _test_normality(control: np.ndarray, treatment: np.ndarray) -> Dict:
        """Test for normality in both groups."""
        _, c_p_value = stats.normaltest(control)
        _, t_p_value = stats.normaltest(treatment)
        
        return {
            'control_normal': c_p_value >= 0.05,
            'treatment_normal': t_p_value >= 0.05,
            'control_p_value': c_p_value,
            'treatment_p_value': t_p_value
        }
    
    @staticmethod
    def _test_variance_equality(control: np.ndarray, 
                              treatment: np.ndarray) -> Dict:
        """Test for equality of variances."""
        _, p_value = stats.levene(control, treatment)
        return {
            'equal_variances': p_value >= 0.05,
            'p_value': p_value
        }
    
    @staticmethod
    def _analyze_outliers(control: np.ndarray, treatment: np.ndarray) -> Dict:
        """Analyze outliers using IQR method."""
        def get_outliers(data):
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            return len(outliers), outliers
            
        c_count, c_outliers = get_outliers(control)
        t_count, t_outliers = get_outliers(treatment)
        
        return {
            'control_outliers': {
                'count': c_count,
                'percentage': c_count/len(control) * 100
            },
            'treatment_outliers': {
                'count': t_count,
                'percentage': t_count/len(treatment) * 100
            }
        }
    
    @staticmethod
    def _check_sample_size_adequacy(control: np.ndarray, 
                                  treatment: np.ndarray) -> Dict:
        """Check if sample sizes are adequate for reliable analysis."""
        return {
            'adequate_sample_size': len(control) >= 30 and len(treatment) >= 30,
            'control_size': len(control),
            'treatment_size': len(treatment),
            'recommendation': "Sample size is adequate" 
                if len(control) >= 30 and len(treatment) >= 30
                else "Consider collecting more data"
        }

# Example usage
if __name__ == "__main__":
    # Sample data
    np.random.seed(42)
    
    # Create sample data for multiple metrics
    data = {
        'control': {
            'conversion_rate': np.random.normal(0.15, 0.02, 1000),
            'revenue': np.random.normal(100, 20, 1000),
            'engagement_time': np.random.normal(300, 60, 1000)
        },
        'treatment': {
            'conversion_rate': np.random.normal(0.16, 0.02, 1000),
            'revenue': np.random.normal(105, 20, 1000),
            'engagement_time': np.random.normal(310, 60, 1000)
        }
    }
    
    # Define metric types
    metric_types = {
        'conversion_rate': 'percentage',
        'revenue': 'currency',
        'engagement_time': 'time'
    }
    
    # Create analyzer instance
    analyzer = ABTestAnalyzer(data['control'], data['treatment'], metric_types)
    
    # Run full analysis
    results = analyzer.run_full_analysis()
    
    # Print summary of results
    for metric, analysis in results.items():
        print(f"\n=== {metric.upper()} ANALYSIS ===")
        print(f"Basic Stats:")
        print(f"Control Mean: {analysis['basic_stats']['control']['mean']:.4f}")
        print(f"Treatment Mean: {analysis['basic_stats']['treatment']['mean']:.4f}")
        print(f"Relative Difference: {analysis['basic_stats']['relative_difference']*100:.2f}%")
        
        print(f"\nStatistical Significance:")
        print(f"P-value (t-test): {analysis['hypothesis_test']['t_test']['p_value']:.4f}")
        
        print(f"\nEffect Size:")
        print(f"Cohen's d: {analysis['effect_size']['cohens_d']:.4f}")
        print(f"Interpretation: {analysis['effect_size']['interpretation']}")
        
        print(f"\nPower Analysis:")
        print(f"Observed Power: {analysis['power_analysis']['observed_power']:.4f}")
        print(f"Well Powered? {analysis['power_analysis']['is_well_powered']}")
