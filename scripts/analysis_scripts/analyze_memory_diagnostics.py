"""
Memory Diagnostics Analyzer
Analyzes backward pass memory diagnostics to identify OOM root causes

Usage:
    python analyze_memory_diagnostics.py [path_to_diagnostics.json]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import argparse


class MemoryDiagnosticsAnalyzer:
    """Analyzes memory diagnostics to find OOM causes."""
    
    def __init__(self, diagnostics_path: Path):
        self.diagnostics_path = diagnostics_path
        self.data: Optional[Dict[str, Any]] = None
        self.snapshots: List[Dict[str, Any]] = []
        self.findings: List[Dict[str, Any]] = []
        
    def load_diagnostics(self) -> bool:
        """Load diagnostics from JSON file."""
        try:
            with open(self.diagnostics_path, 'r') as f:
                self.data = json.load(f)
            
            self.snapshots = self.data.get('snapshots', [])
            print(f"✅ Loaded {len(self.snapshots)} snapshots from {self.diagnostics_path}")
            return True
            
        except FileNotFoundError:
            print(f"❌ File not found: {self.diagnostics_path}")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            return False
    
    def analyze_memory_growth_patterns(self) -> Dict[str, Any]:
        """Analyze memory growth patterns across stages."""
        print("\n" + "="*80)
        print("ANALYZING MEMORY GROWTH PATTERNS")
        print("="*80)
        
        # Group by stage
        by_stage = defaultdict(list)
        for snapshot in self.snapshots:
            stage = snapshot.get('stage', 'unknown')
            by_stage[stage].append(snapshot)
        
        # Calculate statistics for each stage
        stage_stats = {}
        for stage, snapshots_list in by_stage.items():
            allocated = [s.get('cuda_allocated_mb', 0) for s in snapshots_list]
            reserved = [s.get('cuda_reserved_mb', 0) for s in snapshots_list]
            
            if allocated:
                stage_stats[stage] = {
                    'count': len(snapshots_list),
                    'avg_allocated': sum(allocated) / len(allocated),
                    'max_allocated': max(allocated),
                    'min_allocated': min(allocated),
                    'avg_reserved': sum(reserved) / len(reserved) if reserved else 0,
                    'max_reserved': max(reserved) if reserved else 0,
                }
        
        # Print stage statistics
        print("\nMemory usage by stage:")
        print(f"{'Stage':<30} {'Count':<8} {'Avg MB':<10} {'Max MB':<10}")
        print("-" * 80)
        
        for stage, stats in sorted(stage_stats.items()):
            print(f"{stage:<30} {stats['count']:<8} "
                  f"{stats['avg_allocated']:>9.1f} {stats['max_allocated']:>9.1f}")
        
        # Analyze growth between stages
        print("\nMemory growth between consecutive stages:")
        print(f"{'From Stage':<25} {'To Stage':<25} {'Growth (MB)':<15}")
        print("-" * 80)
        
        prev_snapshot = None
        max_growth = 0
        max_growth_transition = None
        
        for snapshot in self.snapshots:
            if prev_snapshot and prev_snapshot.get('batch_idx') == snapshot.get('batch_idx'):
                growth = snapshot.get('cuda_allocated_mb', 0) - prev_snapshot.get('cuda_allocated_mb', 0)
                
                if abs(growth) > 10:  # Only show significant changes
                    from_stage = prev_snapshot.get('stage', 'unknown')
                    to_stage = snapshot.get('stage', 'unknown')
                    print(f"{from_stage:<25} {to_stage:<25} {growth:>14.1f}")
                    
                    if growth > max_growth:
                        max_growth = growth
                        max_growth_transition = (from_stage, to_stage)
            
            prev_snapshot = snapshot
        
        if max_growth_transition:
            finding = {
                'type': 'max_memory_growth',
                'severity': 'critical' if max_growth > 1000 else 'warning',
                'description': f"Largest memory growth: {max_growth:.1f}MB during {max_growth_transition[0]} -> {max_growth_transition[1]}",
                'growth_mb': max_growth,
                'transition': max_growth_transition,
            }
            self.findings.append(finding)
            
            print(f"\n⚠️  FINDING: {finding['description']}")
        
        return stage_stats
    
    def analyze_backward_pass_specifically(self) -> Dict[str, Any]:
        """Focus analysis on backward pass behavior."""
        print("\n" + "="*80)
        print("BACKWARD PASS SPECIFIC ANALYSIS")
        print("="*80)
        
        backward_data = {
            'pre_backward_snapshots': [],
            'post_backward_snapshots': [],
            'growth_per_batch': [],
        }
        
        # Find pre/post backward pairs
        for i in range(len(self.snapshots) - 1):
            curr = self.snapshots[i]
            next_snap = self.snapshots[i + 1]
            
            if (curr.get('stage') == 'pre_backward' and 
                next_snap.get('stage') == 'post_backward' and
                curr.get('batch_idx') == next_snap.get('batch_idx')):
                
                backward_data['pre_backward_snapshots'].append(curr)
                backward_data['post_backward_snapshots'].append(next_snap)
                
                # Calculate growth
                growth = (next_snap.get('cuda_allocated_mb', 0) - 
                         curr.get('cuda_allocated_mb', 0))
                backward_data['growth_per_batch'].append({
                    'batch_idx': curr.get('batch_idx'),
                    'epoch': curr.get('epoch'),
                    'growth_mb': growth,
                    'pre_allocated': curr.get('cuda_allocated_mb', 0),
                    'post_allocated': next_snap.get('cuda_allocated_mb', 0),
                    'params_with_grad': next_snap.get('params_with_grad', 0),
                    'total_gradient_norm': next_snap.get('total_gradient_norm', 0),
                    'nan_grads': next_snap.get('params_with_nan_grad', 0),
                    'inf_grads': next_snap.get('params_with_inf_grad', 0),
                })
        
        if backward_data['growth_per_batch']:
            print(f"\nAnalyzed {len(backward_data['growth_per_batch'])} backward passes")
            
            growths = [b['growth_mb'] for b in backward_data['growth_per_batch']]
            avg_growth = sum(growths) / len(growths)
            max_growth = max(growths)
            min_growth = min(growths)
            
            print(f"\nBackward pass memory growth statistics:")
            print(f"  Average growth: {avg_growth:.1f}MB")
            print(f"  Maximum growth: {max_growth:.1f}MB")
            print(f"  Minimum growth: {min_growth:.1f}MB")
            
            # Find problematic batches
            problematic = [b for b in backward_data['growth_per_batch'] if b['growth_mb'] > avg_growth * 2]
            
            if problematic:
                print(f"\n⚠️  Found {len(problematic)} batches with excessive memory growth:")
                print(f"{'Batch':<8} {'Epoch':<8} {'Growth (MB)':<15} {'Gradient Norm':<15} {'NaN/Inf'}")
                print("-" * 80)
                
                for batch in problematic[:10]:  # Show top 10
                    nan_inf = f"{batch['nan_grads']}/{batch['inf_grads']}"
                    print(f"{batch['batch_idx']:<8} {batch['epoch']:<8} "
                          f"{batch['growth_mb']:>14.1f} {batch['total_gradient_norm']:>14.4f} {nan_inf:>10}")
                
                finding = {
                    'type': 'excessive_backward_growth',
                    'severity': 'critical',
                    'description': f"{len(problematic)} batches have >2x average memory growth during backward",
                    'avg_growth_mb': avg_growth,
                    'problematic_batches': len(problematic),
                }
                self.findings.append(finding)
            
            # Check for gradient issues
            nan_batches = [b for b in backward_data['growth_per_batch'] if b['nan_grads'] > 0]
            inf_batches = [b for b in backward_data['growth_per_batch'] if b['inf_grads'] > 0]
            
            if nan_batches:
                print(f"\n⚠️  WARNING: {len(nan_batches)} batches had NaN gradients!")
                finding = {
                    'type': 'nan_gradients',
                    'severity': 'critical',
                    'description': f"NaN gradients detected in {len(nan_batches)} batches",
                    'affected_batches': len(nan_batches),
                }
                self.findings.append(finding)
            
            if inf_batches:
                print(f"\n⚠️  WARNING: {len(inf_batches)} batches had Inf gradients!")
                finding = {
                    'type': 'inf_gradients',
                    'severity': 'critical',
                    'description': f"Inf gradients detected in {len(inf_batches)} batches",
                    'affected_batches': len(inf_batches),
                }
                self.findings.append(finding)
        
        else:
            print("\n❌ No backward pass data found in diagnostics")
        
        return backward_data
    
    def detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        print("\n" + "="*80)
        print("MEMORY LEAK DETECTION")
        print("="*80)
        
        leaks = self.data.get('potential_leaks', [])
        
        if leaks:
            print(f"\n⚠️  {len(leaks)} potential memory leaks detected:")
            for leak in leaks:
                print(f"\n  Type: {leak['type']}")
                print(f"  Description: {leak['description']}")
                if 'growth_rate_mb_per_step' in leak:
                    print(f"  Growth rate: {leak['growth_rate_mb_per_step']:.2f}MB per step")
                if 'variation' in leak:
                    print(f"  Variation: {leak['variation']} tensors")
                
                self.findings.append({
                    'type': 'memory_leak',
                    'severity': 'critical',
                    'description': leak['description'],
                    'details': leak,
                })
        else:
            print("\n✅ No obvious memory leaks detected")
        
        return leaks
    
    def analyze_tensor_accumulation(self) -> Dict[str, Any]:
        """Analyze tensor count over time."""
        print("\n" + "="*80)
        print("TENSOR ACCUMULATION ANALYSIS")
        print("="*80)
        
        tensor_counts = [s.get('num_tensors', 0) for s in self.snapshots]
        grad_tensor_counts = [s.get('num_tensors_with_grad', 0) for s in self.snapshots]
        
        if tensor_counts:
            print(f"\nTensor statistics:")
            print(f"  Min tensors: {min(tensor_counts)}")
            print(f"  Max tensors: {max(tensor_counts)}")
            print(f"  Average tensors: {sum(tensor_counts) / len(tensor_counts):.1f}")
            print(f"  Variation: {max(tensor_counts) - min(tensor_counts)}")
            
            if grad_tensor_counts:
                print(f"\nTensors with gradients:")
                print(f"  Min: {min(grad_tensor_counts)}")
                print(f"  Max: {max(grad_tensor_counts)}")
                print(f"  Average: {sum(grad_tensor_counts) / len(grad_tensor_counts):.1f}")
            
            # Check for accumulation
            if len(tensor_counts) >= 10:
                recent_avg = sum(tensor_counts[-10:]) / 10
                early_avg = sum(tensor_counts[:10]) / 10
                
                if recent_avg > early_avg * 1.5:
                    print(f"\n⚠️  WARNING: Tensor count increasing over time!")
                    print(f"     Early average: {early_avg:.1f}")
                    print(f"     Recent average: {recent_avg:.1f}")
                    print(f"     Increase: {((recent_avg / early_avg - 1) * 100):.1f}%")
                    
                    finding = {
                        'type': 'tensor_accumulation',
                        'severity': 'warning',
                        'description': f"Tensor count increased {((recent_avg / early_avg - 1) * 100):.1f}% over time",
                        'early_avg': early_avg,
                        'recent_avg': recent_avg,
                    }
                    self.findings.append(finding)
        
        return {
            'tensor_counts': tensor_counts,
            'grad_tensor_counts': grad_tensor_counts,
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on findings."""
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        recommendations = []
        
        # Analyze findings
        has_excessive_growth = any(f['type'] == 'excessive_backward_growth' for f in self.findings)
        has_leak = any(f['type'] == 'memory_leak' for f in self.findings)
        has_nan_inf = any(f['type'] in ('nan_gradients', 'inf_gradients') for f in self.findings)
        has_tensor_accumulation = any(f['type'] == 'tensor_accumulation' for f in self.findings)
        
        if has_excessive_growth:
            recommendations.append(
                "🔧 REDUCE BATCH SIZE: Backward pass shows excessive memory growth. "
                "Try reducing batch_size by 25-50%."
            )
            recommendations.append(
                "🔧 ENABLE GRADIENT CHECKPOINTING: Trades computation for memory. "
                "Add gradient checkpointing to large model components."
            )
            recommendations.append(
                "🔧 INCREASE GRADIENT ACCUMULATION: Instead of batch_size=32, use "
                "batch_size=16 with gradient_accumulation_steps=2."
            )
        
        if has_leak:
            recommendations.append(
                "🔧 CHECK FOR MEMORY LEAKS: Ensure optimizer.zero_grad(set_to_none=True) "
                "is called and no tensors are being retained unnecessarily."
            )
            recommendations.append(
                "🔧 USE CONTEXT MANAGERS: Ensure forward pass outputs don't retain "
                "unnecessary references to intermediate activations."
            )
        
        if has_nan_inf:
            recommendations.append(
                "🔧 GRADIENT CLIPPING: Enable gradient clipping to prevent gradient explosion. "
                "Set clip_grad_norm to 1.0 or lower."
            )
            recommendations.append(
                "🔧 REDUCE LEARNING RATE: NaN/Inf gradients may indicate learning rate is too high."
            )
            recommendations.append(
                "🔧 CHECK DATA SCALING: Verify input data is properly normalized/scaled."
            )
        
        if has_tensor_accumulation:
            recommendations.append(
                "🔧 DETACH INTERMEDIATE TENSORS: Ensure loss computation doesn't retain "
                "full computational graph. Use .detach() where appropriate."
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "✅ NO CRITICAL ISSUES FOUND: Memory usage appears normal. "
                "If you're still seeing OOM, try reducing model size (d_model, n_heads, e_layers)."
            )
        
        recommendations.extend([
            "",
            "🔧 GENERAL OPTIMIZATIONS:",
            "   • Enable mixed precision training (use_amp=True)",
            "   • Use pin_memory=True for data loaders",
            "   • Clear CUDA cache periodically: torch.cuda.empty_cache()",
            "   • Reduce sequence length if possible",
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{rec}")
        
        return recommendations
    
    def print_summary(self):
        """Print comprehensive summary."""
        print("\n" + "="*80)
        print("DIAGNOSTICS SUMMARY")
        print("="*80)
        
        print(f"\nDiagnostics file: {self.diagnostics_path}")
        print(f"Total snapshots: {len(self.snapshots)}")
        print(f"Findings: {len(self.findings)}")
        
        if self.findings:
            print("\nKey findings:")
            for finding in self.findings:
                severity_icon = "🔴" if finding['severity'] == 'critical' else "🟡"
                print(f"  {severity_icon} [{finding['severity'].upper()}] {finding['description']}")
        
        print("\n" + "="*80)
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        if not self.load_diagnostics():
            return False
        
        self.analyze_memory_growth_patterns()
        self.analyze_backward_pass_specifically()
        self.detect_memory_leaks()
        self.analyze_tensor_accumulation()
        self.generate_recommendations()
        self.print_summary()
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Analyze backward pass memory diagnostics')
    parser.add_argument(
        'diagnostics_file',
        nargs='?',
        default='checkpoints/*/backward_memory_diagnostics.json',
        help='Path to diagnostics JSON file'
    )
    
    args = parser.parse_args()
    
    # Find diagnostics file
    diagnostics_path = Path(args.diagnostics_file)
    
    if '*' in str(diagnostics_path):
        # Find most recent
        import glob
        matches = glob.glob(str(diagnostics_path))
        if not matches:
            print(f"❌ No diagnostics files found matching: {diagnostics_path}")
            print("\nLooking for diagnostics in checkpoints directory...")
            
            checkpoint_dir = Path("checkpoints")
            if checkpoint_dir.exists():
                all_diagnostics = list(checkpoint_dir.glob("*/backward_memory_diagnostics.json"))
                if all_diagnostics:
                    # Get most recent
                    diagnostics_path = max(all_diagnostics, key=lambda p: p.stat().st_mtime)
                    print(f"✅ Found: {diagnostics_path}")
                else:
                    print("❌ No diagnostics files found in checkpoints/")
                    print("\nRun training with backward debugging enabled first:")
                    print("  python scripts/train/train_with_backward_debugging.py")
                    return 1
            else:
                print("❌ Checkpoints directory not found")
                return 1
        else:
            diagnostics_path = Path(max(matches, key=lambda p: Path(p).stat().st_mtime))
            print(f"✅ Using most recent: {diagnostics_path}")
    
    if not diagnostics_path.exists():
        print(f"❌ File not found: {diagnostics_path}")
        return 1
    
    # Run analysis
    analyzer = MemoryDiagnosticsAnalyzer(diagnostics_path)
    success = analyzer.run_full_analysis()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
