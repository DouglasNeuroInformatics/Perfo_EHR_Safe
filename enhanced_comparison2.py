# enhanced_comparison.py
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from scipy.stats import ks_2samp, pearsonr, skew, kurtosis
import os
from typing import Dict, Tuple, List
from tqdm import tqdm

class EnhancedDataComparator:
    def __init__(self):
        # Create output directory for plots
        self.output_dir = 'comparison_plots'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data
        print("Loading data...")
        try:
            with open('synthetic_data.pkl', 'rb') as f:
                self.synthetic_encoded = pickle.load(f)
                
            with open('encoded_data.pkl', 'rb') as f:
                self.real_encoded = pickle.load(f)
                
            with open('normalized_data.pkl', 'rb') as f:
                self.normalized_data = pickle.load(f)
                
            self.n_real = len(self.real_encoded)
            self.n_synthetic = len(self.synthetic_encoded)
            self.n_features = self.real_encoded.shape[1]
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise
    
    def plot_feature_distributions(self):
        """Plot distributions for all features with statistical tests"""
        try:
            n_cols = 4
            n_rows = (self.n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
            if n_rows == 1:
                axes = [axes]  # Convert to list if only one row
            axes = np.array(axes).ravel()
            
            for i in range(self.n_features):
                # Get data
                real_feature = self.real_encoded[:, i]
                synth_feature = self.synthetic_encoded[:, i]
                
                # Perform KS test
                ks_stat, p_value = ks_2samp(real_feature, synth_feature)
                
                # Plot distributions
                sns.kdeplot(real_feature, ax=axes[i], label='Real', color='blue', alpha=0.5)
                sns.kdeplot(synth_feature, ax=axes[i], label='Synthetic', color='red', alpha=0.5)
                
                # Add statistics
                axes[i].set_title(f'Feature {i+1}\nKS p-value: {p_value:.3e}')
                axes[i].legend()
            
            # Remove empty subplots
            for i in range(self.n_features, len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_distributions.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error in plot_feature_distributions: {e}")
            plt.close()
    
    def plot_statistical_moments(self):
        """Compare statistical moments between real and synthetic data"""
        try:
            # Calculate moments
            moments = {
                'mean': (np.mean(self.real_encoded, axis=0), 
                        np.mean(self.synthetic_encoded, axis=0)),
                'std': (np.std(self.real_encoded, axis=0),
                       np.std(self.synthetic_encoded, axis=0)),
                'skew': (stats.skew(self.real_encoded, axis=0),
                        stats.skew(self.synthetic_encoded, axis=0)),
                'kurtosis': (stats.kurtosis(self.real_encoded, axis=0),
                            stats.kurtosis(self.synthetic_encoded, axis=0))
            }
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            axes = axes.ravel()
            
            # Plot each moment
            for i, (moment_name, (real_moment, synth_moment)) in enumerate(moments.items()):
                axes[i].scatter(real_moment, synth_moment, alpha=0.5)
                
                # Add diagonal line
                min_val = min(real_moment.min(), synth_moment.min())
                max_val = max(real_moment.max(), synth_moment.max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
                
                # Calculate and add R² value
                r2 = np.corrcoef(real_moment, synth_moment)[0, 1]**2
                
                axes[i].set_title(f'{moment_name.capitalize()} Comparison\nR² = {r2:.3f}')
                axes[i].set_xlabel('Real Data')
                axes[i].set_ylabel('Synthetic Data')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'statistical_moments.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error in plot_statistical_moments: {e}")
            plt.close()
    
    def plot_correlation_analysis(self):
        """Detailed correlation analysis"""
        try:
            # Compute correlation matrices
            real_corr = np.corrcoef(self.real_encoded.T)
            synth_corr = np.corrcoef(self.synthetic_encoded.T)
            
            # Plot correlation matrices
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
            
            # Real correlations
            sns.heatmap(real_corr, ax=ax1, cmap='coolwarm', center=0, vmin=-1, vmax=1)
            ax1.set_title('Real Data Correlations')
            
            # Synthetic correlations
            sns.heatmap(synth_corr, ax=ax2, cmap='coolwarm', center=0, vmin=-1, vmax=1)
            ax2.set_title('Synthetic Data Correlations')
            
            # Correlation differences
            diff_corr = np.abs(real_corr - synth_corr)
            sns.heatmap(diff_corr, ax=ax3, cmap='YlOrRd', vmin=0, vmax=1)
            ax3.set_title(f'Absolute Correlation Differences\nMean Diff: {np.mean(diff_corr):.3f}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'correlation_analysis.png'))
            plt.close()
            
            # Plot correlation preservation
            plt.figure(figsize=(8, 8))
            triu_indices = np.triu_indices(self.n_features, k=1)
            real_corr_triu = real_corr[triu_indices]
            synth_corr_triu = synth_corr[triu_indices]
            
            plt.scatter(real_corr_triu, synth_corr_triu, alpha=0.5)
            
            min_val = min(real_corr_triu.min(), synth_corr_triu.min())
            max_val = max(real_corr_triu.max(), synth_corr_triu.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Calculate and add R² value
            r2 = np.corrcoef(real_corr_triu, synth_corr_triu)[0, 1]**2
            
            plt.xlabel('Real Data Correlations')
            plt.ylabel('Synthetic Data Correlations')
            plt.title(f'Correlation Preservation\nR² = {r2:.3f}')
            
            plt.savefig(os.path.join(self.output_dir, 'correlation_preservation.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error in plot_correlation_analysis: {e}")
            plt.close()
    
    def generate_summary_report(self):
        """Generate and save detailed summary statistics"""
        try:
            # Calculate basic statistics
            stats_dict = {
                'Real_Mean': np.mean(self.real_encoded, axis=0),
                'Synthetic_Mean': np.mean(self.synthetic_encoded, axis=0),
                'Real_Std': np.std(self.real_encoded, axis=0),
                'Synthetic_Std': np.std(self.synthetic_encoded, axis=0),
                'Real_Skew': stats.skew(self.real_encoded, axis=0),
                'Synthetic_Skew': stats.skew(self.synthetic_encoded, axis=0),
                'Real_Kurtosis': stats.kurtosis(self.real_encoded, axis=0),
                'Synthetic_Kurtosis': stats.kurtosis(self.synthetic_encoded, axis=0)
            }
            
            # Calculate KS test statistics
            ks_stats = []
            ks_pvals = []
            for i in range(self.n_features):
                ks_stat, p_val = ks_2samp(self.real_encoded[:, i], 
                                        self.synthetic_encoded[:, i])
                ks_stats.append(ks_stat)
                ks_pvals.append(p_val)
            
            stats_dict['KS_Statistic'] = ks_stats
            stats_dict['KS_P_Value'] = ks_pvals
            
            # Create DataFrame
            stats_df = pd.DataFrame(stats_dict)
            
            # Calculate summary statistics for differences
            diff_stats = {
                'Mean_Abs_Diff': np.mean(np.abs(stats_dict['Real_Mean'] - 
                                              stats_dict['Synthetic_Mean'])),
                'Std_Abs_Diff': np.mean(np.abs(stats_dict['Real_Std'] - 
                                             stats_dict['Synthetic_Std'])),
                'Skew_Abs_Diff': np.mean(np.abs(stats_dict['Real_Skew'] - 
                                              stats_dict['Synthetic_Skew'])),
                'Kurtosis_Abs_Diff': np.mean(np.abs(stats_dict['Real_Kurtosis'] - 
                                                  stats_dict['Synthetic_Kurtosis'])),
                'Mean_KS_Stat': np.mean(ks_stats),
                'Mean_KS_P_Value': np.mean(ks_pvals)
            }
            
            # Save detailed report
            with open(os.path.join(self.output_dir, 'summary_report.txt'), 'w') as f:
                f.write("Summary Statistics Report\n")
                f.write("=======================\n\n")
                f.write("Feature-wise Statistics:\n")
                f.write("----------------------\n")
                f.write(stats_df.to_string())
                f.write("\n\nOverall Differences:\n")
                f.write("------------------\n")
                for metric, value in diff_stats.items():
                    f.write(f"{metric}: {value:.4f}\n")
            
            # Save DataFrames for further analysis
            with open(os.path.join(self.output_dir, 'summary_statistics.pkl'), 'wb') as f:
                pickle.dump({
                    'feature_stats': stats_df,
                    'overall_diffs': diff_stats
                }, f)
            
        except Exception as e:
            print(f"Error in generate_summary_report: {e}")
    
    def run_all_analyses(self):
        """Run all comparison analyses"""
        print("Generating feature distribution plots...")
        self.plot_feature_distributions()
        
        print("Analyzing statistical moments...")
        self.plot_statistical_moments()
        
        print("Performing correlation analysis...")
        self.plot_correlation_analysis()
        
        print("Generating summary report...")
        self.generate_summary_report()
        
        print(f"\nAnalysis complete! All results have been saved to: {self.output_dir}/")

def main():
    try:
        comparator = EnhancedDataComparator()
        comparator.run_all_analyses()
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()