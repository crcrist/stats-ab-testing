# main.py

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import warnings

class ABTestAnalyzer:
    """A comprehensive tool for analyzing AB test results."""
    
    def __init__(self, control_data: Dict[str, np.ndarray], 
                 treatment_data: Dict[str, np.ndarray],
                 metric_types: Dict[str, str] = None):
        self.control = control_data
        self.treatment = treatment_data
        self.metric_types = metric_types
        self.metrics = list(control_data.keys())
    
    def _calculate_basic_stats(self, control: np.ndarray, treatment: np.ndarray) -> Dict:
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
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        
        n1, n2 = len(control), len(treatment)
        pooled_se = np.sqrt(np.var(control)/n1 + np.var(treatment)/n2)
        
        t_stat = stats.t.ppf((1 + confidence_level)/2, n1 + n2 - 2)
        margin_of_error = t_stat * pooled_se
        
        difference = treatment_mean - control_mean
        
        return {
            'difference': difference,
            'ci_lower': difference - margin_of_error,
            'ci_upper': difference + margin_of_error,
            'relative_difference': difference / control_mean
        }
    
    def _run_hypothesis_test(self, control: np.ndarray, treatment: np.ndarray) -> Dict:
        t_stat, p_value = stats.ttest_ind(treatment, control)
        return {
            't_test': {'statistic': t_stat, 'p_value': p_value}
        }
    
    def _calculate_effect_size(self, control: np.ndarray, treatment: np.ndarray) -> Dict:
        pooled_std = np.sqrt((np.var(control) + np.var(treatment)) / 2)
        cohens_d = (np.mean(treatment) - np.mean(control)) / pooled_std
        
        return {
            'cohens_d': cohens_d,
            'interpretation': self._interpret_cohens_d(cohens_d)
        }
    
    def _perform_power_analysis(self, control: np.ndarray, treatment: np.ndarray) -> Dict:
        effect_size = self._calculate_effect_size(control, treatment)['cohens_d']
        n1, n2 = len(control), len(treatment)
        
        alpha = 0.05
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = effect_size * np.sqrt(n1*n2/(n1 + n2)) - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return {
            'observed_power': power,
            'is_well_powered': power >= 0.8
        }
    
    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        if abs(d) < 0.2:
            return "negligible effect"
        elif abs(d) < 0.5:
            return "small effect"
        elif abs(d) < 0.8:
            return "medium effect"
        else:
            return "large effect"
    
    def run_full_analysis(self, confidence_level: float = 0.95) -> Dict:
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
                'power_analysis': self._perform_power_analysis(control_data, treatment_data)
            }
            
        return results
    
    def create_visualizations(self) -> None:
        """Create and save visualizations for the analysis."""
        # Set some nice default parameters
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        
        # Define colors
        control_color = '#1f77b4'  # Blue
        treatment_color = '#2ca02c'  # Green
        
        # Create visualizations for each metric
        for metric in self.metrics:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle(f'{metric.replace("_", " ").title()} Analysis', fontsize=14)
            
            # Subplot 1: Histograms
            ax1.hist(self.control[metric], alpha=0.5, label='Control', 
                    color=control_color, bins=30, density=True)
            ax1.hist(self.treatment[metric], alpha=0.5, label='Treatment', 
                    color=treatment_color, bins=30, density=True)
            ax1.set_title('Distribution Comparison')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Density')
            ax1.legend()
            
            # Subplot 2: Box Plots
            bp = ax2.boxplot([self.control[metric], self.treatment[metric]], 
                           labels=['Control', 'Treatment'], patch_artist=True)
            
            # Color the boxes
            bp['boxes'][0].set_facecolor(control_color)
            bp['boxes'][1].set_facecolor(treatment_color)
            for box in bp['boxes']:
                box.set_alpha(0.5)
            
            ax2.set_title('Box Plot Comparison')
            ax2.set_ylabel('Value')
            
            # Add means as points
            ax2.plot([1], [np.mean(self.control[metric])], 'o', 
                    color=control_color, label='Control Mean')
            ax2.plot([2], [np.mean(self.treatment[metric])], 'o', 
                    color=treatment_color, label='Treatment Mean')
            ax2.legend()
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f'{metric}_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create effect size visualization
        results = self.run_full_analysis()
        effect_sizes = []
        metrics_list = []
        
        for m, analysis in results.items():
            effect_sizes.append(analysis['effect_size']['cohens_d'])
            metrics_list.append(m)
        
        plt.figure()
        bars = plt.bar(metrics_list, effect_sizes)
        
        # Color bars based on effect size magnitude
        for bar, effect in zip(bars, effect_sizes):
            if abs(effect) < 0.2:
                bar.set_color('lightgray')
            elif abs(effect) < 0.5:
                bar.set_color('lightblue')
            elif abs(effect) < 0.8:
                bar.set_color('steelblue')
            else:
                bar.set_color('darkblue')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        plt.title("Effect Size Comparison Across Metrics")
        plt.ylabel("Cohen's d")
        plt.xlabel("Metrics")
        
        plt.xticks(rotation=45)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('effect_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_sample_data(seed=42):
    """Generate sample AB test data for demonstration."""
    np.random.seed(seed)
    
    sample_size = 1000
    
    data = {
        'control': {
            'conversion_rate': np.random.beta(15, 85, sample_size),
            'revenue': np.exp(np.random.normal(4.5, 0.5, sample_size)),
            'engagement_time': np.maximum(0, np.random.normal(300, 60, sample_size))
        },
        'treatment': {
            'conversion_rate': np.random.beta(17, 83, sample_size),
            'revenue': np.exp(np.random.normal(4.6, 0.5, sample_size)),
            'engagement_time': np.maximum(0, np.random.normal(310, 60, sample_size))
        }
    }
    
    return data

if __name__ == "__main__":
    # Generate sample data
    data = generate_sample_data()
    
    # Define metric types
    metric_types = {
        'conversion_rate': 'percentage',
        'revenue': 'currency',
        'engagement_time': 'time'
    }
    
    # Create analyzer instance
    analyzer = ABTestAnalyzer(data['control'], data['treatment'], metric_types)
    
    # Run analysis and create visualizations
    results = analyzer.run_full_analysis()
    analyzer.create_visualizations()
    
    # Print results
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
