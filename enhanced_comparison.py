# enhanced_comparison.py
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy
from scipy.stats import ks_2samp, pearsonr
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
        with open('synthetic_data.pkl', 'rb') as f:
            self.synthetic_encoded = pickle.load(f)
            
        with open('encoded_data.pkl', 'rb') as f:
            self.real_encoded = pickle.load(f)
            
        with open('normalized_data.pkl', 'rb') as f:
            self.normalized_data = pickle.load(f)
            
        self.n_real = len(self.real_encoded)
        self.n_synthetic = len(self.synthetic_encoded)
        self.n_features = self.real_encoded.shape[1]
    
    def plot_feature_distributions(self):
        """Plot distributions for all features with statistical tests"""
        n_cols = 4
        n_rows = (self.n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        axes = axes.ravel()
        
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
            axes[i].set_title(f'Feature {i+1}\nKS p-value: {p_value:.3f}')
            axes[i].legend()
        
        # Remove empty subplots
        for i in range(self.n_features, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_distributions.png'))
        plt.close()
    
    def plot_statistical_moments(self):
        """Compare statistical moments between real and synthetic data"""
        # Calculate moments
        moments = {
            'mean': (np.mean(self.real_encoded, axis=0), 
                    np.mean(self.synthetic_encoded, axis=0)),
            'std': (np.std(self.real_encoded, axis=0),
                   np.std(self.synthetic_encoded, axis=0)),
            'skew': (scipy.stats.skew(self.real_encoded, axis=0),
                    scipy.stats.skew(self.synthetic_encoded, axis=0)),
            'kurtosis': (scipy.stats.kurtosis(self.real_encoded, axis=0),
                        scipy.stats.kurtosis(self.synthetic_encoded, axis=0))
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()
        
        for i, (moment_name, (real_moment, synth_moment)) in enumerate(moments.items()):
            axes[i].scatter(real_moment, synth_moment, alpha=0.5)
            
            # Add diagonal line
            min_val = min(real_moment.min(), synth_moment.min())
            max_val = max(real_moment.max(), synth_moment.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
            
            axes[i].set_title(f'{moment_name.capitalize()} Comparison')
            axes[i].set_xlabel('Real Data')
            axes[i].set_ylabel('Synthetic Data')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'statistical_moments.png'))
        plt.close()
    
    def plot_correlation_analysis(self):
        """Detailed correlation analysis"""
        # Compute correlation matrices
        real_corr = np.corrcoef(self.real_encoded.T)
        synth_corr = np.corrcoef(self.synthetic_encoded.T)
        
        # Plot correlation matrices
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        sns.heatmap(real_corr, ax=ax1, cmap='coolwarm', center=0)
        ax1.set_title('Real Data Correlations')
        
        sns.heatmap(synth_corr, ax=ax2, cmap='coolwarm', center=0)
        ax2.set_title('Synthetic Data Correlations')
        
        sns.heatmap(np.abs(real_corr - synth_corr), ax=ax3, cmap='coolwarm')
        ax3.set_title('Absolute Correlation Differences')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_analysis.png'))
        plt.close()
        
        # Plot correlation preservation
        plt.figure(figsize=(8, 8))
        plt.scatter(real_corr[np.triu_indices(self.n_features, k=1)],
                   synth_corr[np.triu_indices(self.n_features, k=1)],
                   alpha=0.5)
        
        min_val = min(real_corr.min(), synth_corr.min())
        max_val = max(real_corr.max(), synth_corr.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Real Data Correlations')
        plt.ylabel('Synthetic Data Correlations')
        plt.title('Correlation Preservation')
        
        plt.savefig(os.path.join(self.output_dir, 'correlation_preservation.png'))
        plt.close()
    
    def plot_dimensionality_reduction_analysis(self):
        """Enhanced dimensionality reduction analysis"""
        # Combine data
        combined_data = np.vstack([self.real_encoded, self.synthetic_encoded])
        labels = ['Real'] * self.n_real + ['Synthetic'] * self.n_synthetic
        
        # PCA
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(combined_data)
        
        # t-SNE with different perplexities
        perplexities = [5, 30, 50]
        tsne_results = []
        
        for perp in perplexities:
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
            tsne_results.append(tsne.fit_transform(combined_data))
        
        # Plot PCA (2D and 3D)
        fig = plt.figure(figsize=(15, 5))
        
        # 2D PCA
        ax1 = fig.add_subplot(121)
        for label, color in zip(['Real', 'Synthetic'], ['blue', 'red']):
            mask = np.array(labels) == label
            ax1.scatter(pca_result[mask, 0], pca_result[mask, 1],
                       label=label, alpha=0.5, color=color)
        ax1.set_title('PCA (2D)')
        ax1.legend()
        
        # 3D PCA
        ax2 = fig.add_subplot(122, projection='3d')
        for label, color in zip(['Real', 'Synthetic'], ['blue', 'red']):
            mask = np.array(labels) == label
            ax2.scatter(pca_result[mask, 0], pca_result[mask, 1], pca_result[mask, 2],
                       label=label, alpha=0.5, color=color)
        ax2.set_title('PCA (3D)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pca_analysis.png'))
        plt.close()
        
        # Plot t-SNE results
        fig, axes = plt.subplots(1, len(perplexities), figsize=(20, 5))
        
        for i, (perp, tsne_result) in enumerate(zip(perplexities, tsne_results)):
            for label, color in zip(['Real', 'Synthetic'], ['blue', 'red']):
                mask = np.array(labels) == label
                axes[i].scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                              label=label, alpha=0.5, color=color)
            axes[i].set_title(f't-SNE (perplexity={perp})')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'tsne_analysis.png'))
        plt.close()
    
    def plot_density_comparison(self):
        """2D density comparison for pairs of features"""
        n_pairs = min(6, (self.n_features * (self.n_features - 1)) // 2)
        feature_pairs = [(i, i+1) for i in range(0, 2*n_pairs, 2)]
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        for i, (f1, f2) in enumerate(feature_pairs):
            # Real data
            sns.kdeplot(
                x=self.real_encoded[:, f1],
                y=self.real_encoded[:, f2],
                ax=axes[i],
                levels=5,
                color='blue',
                alpha=0.5,
                label='Real'
            )
            
            # Synthetic data
            sns.kdeplot(
                x=self.synthetic_encoded[:, f1],
                y=self.synthetic_encoded[:, f2],
                ax=axes[i],
                levels=5,
                color='red',
                alpha=0.5,
                label='Synthetic'
            )
            
            axes[i].set_title(f'Features {f1+1} vs {f2+1}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'density_comparison.png'))
        plt.close()
    
    def generate_summary_report(self):
        """Generate and save summary statistics"""
        report = {
            'Feature Statistics': pd.DataFrame({
                'Real_Mean': np.mean(self.real_encoded, axis=0),
                'Synthetic_Mean': np.mean(self.synthetic_encoded, axis=0),
                'Real_Std': np.std(self.real_encoded, axis=0),
                'Synthetic_Std': np.std(self.synthetic_encoded, axis=0),
                'KS_Statistic': [ks_2samp(self.real_encoded[:, i],
                                        self.synthetic_encoded[:, i])[0]
                                for i in range(self.n_features)],
                'KS_P_Value': [ks_2samp(self.real_encoded[:, i],
                                      self.synthetic_encoded[:, i])[1]
                                for i in range(self.n_features)]
            })
        }
        
        # Save report
        with open(os.path.join(self.output_dir, 'summary_report.pkl'), 'wb') as f:
            pickle.dump(report, f)
        
        # Save readable version
        with open(os.path.join(self.output_dir, 'summary_report.txt'), 'w') as f:
            f.write("Summary Statistics Report\n")
            f.write("=======================\n\n")
            f.write(report['Feature Statistics'].to_string())
    
    def run_all_analyses(self):
        """Run all comparison analyses"""
        print("Generating feature distribution plots...")
        self.plot_feature_distributions()
        
        print("Analyzing statistical moments...")
        self.plot_statistical_moments()
        
        print("Performing correlation analysis...")
        self.plot_correlation_analysis()
        
        print("Running dimensionality reduction analysis...")
        self.plot_dimensionality_reduction_analysis()
        
        print("Generating density comparisons...")
        self.plot_density_comparison()
        
        print("Generating summary report...")
        self.generate_summary_report()
        
        print(f"\nAnalysis complete! All plots have been saved to: {self.output_dir}/")

def main():
    comparator = EnhancedDataComparator()
    comparator.run_all_analyses()

if __name__ == "__main__":
    main()