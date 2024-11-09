import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import warnings
import seaborn as sns

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
    
    def create_enhanced_visualizations(self) -> None:
        """Create enhanced visualizations with confidence intervals and error bars."""
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        
        results = self.run_full_analysis()
        
        # Colors
        control_color = '#1f77b4'  # Blue
        treatment_color = '#2ca02c'  # Green
        
        for metric in self.metrics:
            # 1. Distribution and CI Plot
            fig = plt.figure(figsize=(15, 10))
            gs = plt.GridSpec(2, 2)
            
            # Distribution Plot
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[1, 1])
            
            fig.suptitle(f'{metric.replace("_", " ").title()} Analysis', fontsize=14)
            
            # Distribution with confidence intervals
            sns.kdeplot(data=self.control[metric], ax=ax1, color=control_color, 
                       label='Control', fill=True, alpha=0.3)
            sns.kdeplot(data=self.treatment[metric], ax=ax1, color=treatment_color, 
                       label='Treatment', fill=True, alpha=0.3)
            
            # Add mean lines with CI bands
            control_ci = results[metric]['confidence_intervals']
            
            ax1.axvline(np.mean(self.control[metric]), color=control_color, 
                       linestyle='--', alpha=0.8)
            ax1.axvline(np.mean(self.treatment[metric]), color=treatment_color, 
                       linestyle='--', alpha=0.8)
            
            ax1.set_title('Distribution Comparison with Means')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Density')
            ax1.legend()
            
            # Box Plot with Mean Points
            bp = ax2.boxplot([self.control[metric], self.treatment[metric]], 
                            labels=['Control', 'Treatment'], patch_artist=True)
            
            # Color boxes
            bp['boxes'][0].set_facecolor(control_color)
            bp['boxes'][1].set_facecolor(treatment_color)
            for box in bp['boxes']:
                box.set_alpha(0.5)
            
            # Add mean points with error bars
            control_stats = results[metric]['basic_stats']['control']
            treatment_stats = results[metric]['basic_stats']['treatment']
            
            ax2.errorbar([1], [control_stats['mean']], 
                        yerr=control_stats['std']/np.sqrt(control_stats['sample_size']),
                        fmt='o', color=control_color, capsize=5, label='Mean with SE')
            ax2.errorbar([2], [treatment_stats['mean']], 
                        yerr=treatment_stats['std']/np.sqrt(treatment_stats['sample_size']),
                        fmt='o', color=treatment_color, capsize=5)
            
            ax2.set_title('Box Plot with Mean and Standard Error')
            ax2.legend()
            
            # Relative Difference Plot
            diff_data = results[metric]['confidence_intervals']
            
            # Create bar for relative difference
            rel_diff = diff_data['relative_difference'] * 100  # Convert to percentage
            ax3.bar(['Relative Difference'], [rel_diff], 
                    yerr=[abs(rel_diff - diff_data['ci_lower']*100)], 
                    capsize=5, color='lightblue', alpha=0.6)
            
            # Add horizontal line at 0
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            
            # Add confidence interval band
            ax3.fill_between([-.25, .25], 
                            diff_data['ci_lower']*100, 
                            diff_data['ci_upper']*100, 
                            color='gray', alpha=0.2)
            
            ax3.set_title('Relative Difference with 95% CI')
            ax3.set_ylabel('Percentage Difference (%)')
            
            # Add p-value and effect size annotations
            p_value = results[metric]['hypothesis_test']['t_test']['p_value']
            effect_size = results[metric]['effect_size']['cohens_d']
            power = results[metric]['power_analysis']['observed_power']
            
            ax3.text(.5, .95, f'p-value: {p_value:.4f}\nEffect Size (d): {effect_size:.3f}\nPower: {power:.2f}', 
                    transform=ax3.transAxes, verticalalignment='top')
            
            plt.tight_layout()
            plt.savefig(f'{metric}_enhanced_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create Enhanced Effect Size Comparison
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        effect_sizes = []
        effect_errors = []
        metrics_list = []
        
        for m, analysis in results.items():
            effect_sizes.append(analysis['effect_size']['cohens_d'])
            # Calculate approximate SE for Cohen's d
            n1 = analysis['basic_stats']['control']['sample_size']
            n2 = analysis['basic_stats']['treatment']['sample_size']
            d = analysis['effect_size']['cohens_d']
            se_d = np.sqrt((n1 + n2)/(n1 * n2) + d**2/(2*(n1 + n2 - 2)))
            effect_errors.append(se_d)
            metrics_list.append(m)
        
        # Create bar plot with error bars
        bars = plt.bar(metrics_list, effect_sizes, yerr=effect_errors, capsize=5)
        
        # Color bars based on effect size magnitude and add confidence bands
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
        plt.title("Effect Size Comparison with Confidence Intervals")
        plt.ylabel("Cohen's d")
        plt.xlabel("Metrics")
        
        # Add interpretation guide
        plt.axhline(y=0.2, color='gray', linestyle='--', alpha=0.3)
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3)
        plt.text(len(metrics_list)-1, 0.2, 'Small Effect', ha='right', va='bottom')
        plt.text(len(metrics_list)-1, 0.5, 'Medium Effect', ha='right', va='bottom')
        plt.text(len(metrics_list)-1, 0.8, 'Large Effect', ha='right', va='bottom')
        
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('effect_sizes_enhanced.png', dpi=300, bbox_inches='tight')
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
    analyzer.create_enhanced_visualizations()
    
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
