"""
Diagnostic Analysis Script

Analyzes training diagnostics to identify:
1. Gradient flow issues (vanishing/exploding gradients)
2. Attention pattern evolution
3. Component-level performance bottlenecks
4. Training dynamics and convergence
5. Improvement opportunities
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import argparse


class DiagnosticAnalyzer:
    """Analyze training diagnostics."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.gradient_dir = self.log_dir / "gradients"
        self.attention_dir = self.log_dir / "attention"
        self.component_dir = self.log_dir / "components"
        self.output_dir = self.log_dir / "analysis"
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_gradients(self):
        """Analyze gradient flow across training."""
        print("\n" + "="*80)
        print("GRADIENT ANALYSIS")
        print("="*80)
        
        # Load all gradient files
        gradient_files = sorted(self.gradient_dir.glob("gradients_step_*.json"))
        if not gradient_files:
            print("No gradient files found")
            return
        
        print(f"Found {len(gradient_files)} gradient snapshots")
        
        # Collect data
        steps = []
        component_norms = {
            'fusion_cross_attention': [],
            'hierarchical_fusion_proj': [],
            'celestial_to_target_attention': [],
            'wave_patching_composer': [],
            'target_patching_composer': [],
            'graph_combiner': [],
            'stochastic_learner': [],
            'decoder': []
        }
        
        for grad_file in gradient_files:
            with open(grad_file) as f:
                data = json.load(f)
            
            steps.append(data['step'])
            
            for comp_name in component_norms.keys():
                if comp_name in data['component_stats']:
                    component_norms[comp_name].append(
                        data['component_stats'][comp_name]['total_norm']
                    )
                else:
                    component_norms[comp_name].append(0.0)
        
        # Create DataFrame
        df = pd.DataFrame({'step': steps})
        for comp_name, norms in component_norms.items():
            df[comp_name] = norms
        
        # Save to CSV
        csv_path = self.output_dir / "gradient_norms.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved gradient norms to: {csv_path}")
        
        # Plot gradient evolution
        plt.figure(figsize=(15, 8))
        for comp_name, norms in component_norms.items():
            if sum(norms) > 0:
                plt.plot(steps, norms, label=comp_name, marker='o', markersize=3)
        
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.title('Component-wise Gradient Flow')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plot_path = self.output_dir / "gradient_flow.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved gradient plot to: {plot_path}")
        plt.close()
        
        # Identify issues
        print("\nGradient Flow Analysis:")
        for comp_name, norms in component_norms.items():
            if not norms or sum(norms) == 0:
                continue
            
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            max_norm = max(norms)
            min_norm = min([n for n in norms if n > 0], default=0)
            
            print(f"\n{comp_name}:")
            print(f"  Mean norm: {mean_norm:.6f}")
            print(f"  Std norm: {std_norm:.6f}")
            print(f"  Max norm: {max_norm:.6f}")
            print(f"  Min norm: {min_norm:.6f}")
            
            # Check for issues
            if mean_norm < 1e-5:
                print(f"  ⚠️  WARNING: Vanishing gradients (mean < 1e-5)")
            elif mean_norm > 10:
                print(f"  ⚠️  WARNING: Large gradients (mean > 10)")
            
            if max_norm > 100:
                print(f"  ⚠️  WARNING: Exploding gradients detected (max > 100)")
        
        return df
    
    def analyze_attention(self):
        """Analyze attention patterns."""
        print("\n" + "="*80)
        print("ATTENTION ANALYSIS")
        print("="*80)
        
        attention_files = sorted(self.attention_dir.glob("attention_step_*.json"))
        if not attention_files:
            print("No attention files found")
            return
        
        print(f"Found {len(attention_files)} attention snapshots")
        
        # Analyze fusion attention
        fusion_entropy = []
        steps = []
        
        for attn_file in attention_files:
            with open(attn_file) as f:
                data = json.load(f)
            
            steps.append(data['step'])
            
            if 'fusion_attention' in data:
                fusion_entropy.append(data['fusion_attention']['entropy'])
        
        # Plot fusion attention entropy
        if fusion_entropy:
            plt.figure(figsize=(12, 6))
            plt.plot(steps, fusion_entropy, marker='o', markersize=4)
            plt.xlabel('Training Step')
            plt.ylabel('Attention Entropy')
            plt.title('Hierarchical Fusion Attention Entropy (Higher = More Distributed)')
            plt.grid(True, alpha=0.3)
            
            plot_path = self.output_dir / "fusion_attention_entropy.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved fusion attention plot to: {plot_path}")
            plt.close()
            
            print(f"\nFusion Attention Analysis:")
            print(f"  Mean entropy: {np.mean(fusion_entropy):.4f}")
            print(f"  Entropy trend: {'Increasing' if fusion_entropy[-1] > fusion_entropy[0] else 'Decreasing'}")
        
        # Analyze C→T attention
        print("\nC→T Attention Analysis:")
        
        # Get last attention snapshot
        with open(attention_files[-1]) as f:
            last_data = json.load(f)
        
        if 'c2t_attention' in last_data:
            for target_name, target_stats in last_data['c2t_attention'].items():
                print(f"\n{target_name}:")
                print(f"  Mean attention: {target_stats['mean']:.6f}")
                print(f"  Attention spread (std): {target_stats['std']:.6f}")
                print(f"  Max attention: {target_stats['max']:.6f}")
                if 'top_5_celestial' in target_stats:
                    print(f"  Top 5 celestial bodies: {target_stats['top_5_celestial']}")
    
    def analyze_components(self):
        """Analyze component-level performance."""
        print("\n" + "="*80)
        print("COMPONENT ANALYSIS")
        print("="*80)
        
        component_files = sorted(self.component_dir.glob("profile_step_*.json"))
        if not component_files:
            print("No component profiles found")
            return
        
        print(f"Found {len(component_files)} component profiles")
        
        # Get last profile
        with open(component_files[-1]) as f:
            profile_data = json.load(f)
        
        print("\nParameter Breakdown:")
        total_params = 0
        total_trainable = 0
        
        for comp_name, comp_stats in profile_data.get('parameter_breakdown', {}).items():
            params = comp_stats['total_params']
            trainable = comp_stats['trainable_params']
            total_params += params
            total_trainable += trainable
            
            print(f"\n{comp_name}:")
            print(f"  Total parameters: {params:,}")
            print(f"  Trainable parameters: {trainable:,}")
            print(f"  Percentage of total: {params/max(1, total_params)*100:.2f}%")
        
        print(f"\nTotal Model Parameters: {total_params:,}")
        print(f"Total Trainable Parameters: {total_trainable:,}")
        
        # Enhanced config info
        if 'config' in profile_data:
            print("\nModel Configuration:")
            for key, value in profile_data['config'].items():
                print(f"  {key}: {value}")
    
    def generate_improvement_recommendations(self, gradient_df):
        """Generate improvement recommendations based on diagnostics."""
        print("\n" + "="*80)
        print("IMPROVEMENT RECOMMENDATIONS")
        print("="*80)
        
        recommendations = []
        
        # Analyze gradient patterns
        for comp_name in gradient_df.columns:
            if comp_name == 'step':
                continue
            
            norms = gradient_df[comp_name].values
            if sum(norms) == 0:
                continue
            
            mean_norm = np.mean(norms)
            
            # Check for vanishing gradients
            if mean_norm < 1e-5:
                recommendations.append({
                    'component': comp_name,
                    'issue': 'Vanishing gradients',
                    'severity': 'HIGH',
                    'suggestion': f'Consider: (1) Increase learning rate for {comp_name}, '
                                 f'(2) Add skip connections, (3) Use different initialization'
                })
            
            # Check for exploding gradients
            elif mean_norm > 10:
                recommendations.append({
                    'component': comp_name,
                    'issue': 'Large gradients',
                    'severity': 'MEDIUM',
                    'suggestion': f'Consider: (1) Reduce learning rate for {comp_name}, '
                                 f'(2) Increase gradient clipping threshold'
                })
            
            # Check for stability
            std_norm = np.std(norms)
            if std_norm > mean_norm * 2:
                recommendations.append({
                    'component': comp_name,
                    'issue': 'Unstable gradients',
                    'severity': 'MEDIUM',
                    'suggestion': f'Gradient variance is high. Consider: (1) Reduce batch size, '
                                 f'(2) Use gradient accumulation, (3) Add normalization layers'
                })
        
        # Print recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. [{rec['severity']}] {rec['component']}")
                print(f"   Issue: {rec['issue']}")
                print(f"   Suggestion: {rec['suggestion']}")
        else:
            print("\n✅ No major issues detected. Training appears healthy!")
        
        # Save recommendations
        rec_file = self.output_dir / "recommendations.json"
        with open(rec_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"\nRecommendations saved to: {rec_file}")
        
        return recommendations
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE DIAGNOSTIC REPORT")
        print("="*80)
        
        # Analyze all components
        gradient_df = self.analyze_gradients()
        self.analyze_attention()
        self.analyze_components()
        
        # Generate recommendations
        if gradient_df is not None:
            self.generate_improvement_recommendations(gradient_df)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Results saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.iterdir()):
            print(f"  - {file.name}")


def main():
    parser = argparse.ArgumentParser(description='Analyze training diagnostics')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Directory containing diagnostic logs')
    
    args = parser.parse_args()
    
    analyzer = DiagnosticAnalyzer(args.log_dir)
    analyzer.generate_report()


if __name__ == '__main__':
    main()
