import os
import gc
import shutil
import tempfile
import glob
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from utils.logger import logger

plt.switch_backend('agg')


def clear_all_cache(workspace_path=None, verbose=True):
    """
    Clear all types of cache (Python, PyTorch, etc.)
    
    Args:
        workspace_path (str): Path to workspace. If None, uses current directory
        verbose (bool): Whether to print detailed output
    
    Returns:
        dict: Summary of clearing results
    """
    results = {}
    
    if verbose:
        print("=" * 60)
        print("COMPREHENSIVE CACHE CLEARING")
        print("=" * 60)
    
    # Clear Python cache
    python_cleared = clear_python_cache(workspace_path, verbose)
    results['python_cache_cleared'] = python_cleared
    
    if verbose:
        print("\n" + "-" * 40)
    
    # Clear PyTorch cache
    torch_cleared = clear_torch_cache(verbose)
    results['torch_cache_cleared'] = torch_cleared
    
    if verbose:
        print("\n" + "=" * 60)
        print("CACHE CLEARING SUMMARY")
        print("=" * 60)
        print(f"Python cache items cleared: {python_cleared}")
        print(f"PyTorch cache cleared: {'Yes' if torch_cleared else 'No'}")
        print("\nNext steps for VS Code cache:")
        print("1. Close VS Code completely")
        print("2. Delete these directories (if they exist):")
        print("   - %APPDATA%\\Code\\User\\workspaceStorage")
        print("   - %APPDATA%\\Code\\User\\History")
        print("   - %APPDATA%\\Code\\logs")
        print("   - %APPDATA%\\Code\\CachedExtensions")
        print("3. In VS Code, run: Ctrl+Shift+P > 'Developer: Reload Window'")
        print("4. In VS Code, run: Ctrl+Shift+P > 'Python: Clear Cache and Reload Window'")
        print("=" * 60)
    
    return results


def clear_python_cache(workspace_path=None, verbose=True):
    """
    Clear all Python cache files (__pycache__, .pyc, .pyo files) from the workspace
    
    Args:
        workspace_path (str): Path to workspace. If None, uses current directory
        verbose (bool): Whether to print detailed output
    
    Returns:
        int: Number of items cleared
    """
    if workspace_path is None:
        workspace_path = os.getcwd()
    
    cleared_count = 0
    
    if verbose:
        print(f"Clearing Python cache from: {workspace_path}")
    
    # Clear __pycache__ directories
    for root, dirs, files in os.walk(workspace_path):
        for dir_name in dirs[:]:  # Use slice to avoid modification during iteration
            if dir_name == '__pycache__':
                cache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(cache_path)
                    if verbose:
                        print(f"Removed: {cache_path}")
                    cleared_count += 1
                    dirs.remove(dir_name)  # Don't recurse into deleted directory
                except Exception as e:
                    if verbose:
                        print(f"Error removing {cache_path}: {e}")
    
    # Clear .pyc and .pyo files
    for pattern in ['**/*.pyc', '**/*.pyo']:
        for file_path in glob.glob(os.path.join(workspace_path, pattern), recursive=True):
            try:
                os.remove(file_path)
                if verbose:
                    print(f"Removed: {file_path}")
                cleared_count += 1
            except Exception as e:
                if verbose:
                    print(f"Error removing {file_path}: {e}")
    
    # Clear pytest cache
    pytest_cache = os.path.join(workspace_path, '.pytest_cache')
    if os.path.exists(pytest_cache):
        try:
            shutil.rmtree(pytest_cache)
            if verbose:
                print(f"Removed: {pytest_cache}")
            cleared_count += 1
        except Exception as e:
            if verbose:
                print(f"Error removing {pytest_cache}: {e}")
    
    if verbose:
        print(f"Python cache clearing complete! Cleared {cleared_count} items.")
    
    return cleared_count


def clear_torch_cache(verbose=True):
    """
    Clear PyTorch cache
    
    Args:
        verbose (bool): Whether to print output
    
    Returns:
        bool: True if successful
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if verbose:
                print("PyTorch CUDA cache cleared")
        
        # Clear PyTorch hub cache
        torch_cache_dir = torch.hub.get_dir()
        if os.path.exists(torch_cache_dir):
            for item in os.listdir(torch_cache_dir):
                item_path = os.path.join(torch_cache_dir, item)
                if os.path.isdir(item_path):
                    try:
                        shutil.rmtree(item_path)
                        if verbose:
                            print(f"Removed PyTorch hub cache: {item_path}")
                    except Exception as e:
                        if verbose:
                            print(f"Error removing {item_path}: {e}")
        
        return True
    except Exception as e:
        if verbose:
            print(f"Error clearing PyTorch cache: {e}")
        return False


def clear_vscode_cache():
    """
    Instructions and helper for clearing VS Code cache.
    This function provides guidance since VS Code cache clearing requires manual steps.
    """
    logger.info("VS Code cache clearing instructions:")
    
    vscode_cache_paths = []
    
    # Windows paths
    if os.name == 'nt':
        user_profile = os.environ.get('USERPROFILE', '')
        vscode_cache_paths.extend([
            os.path.join(user_profile, 'AppData', 'Roaming', 'Code', 'CachedData'),
            os.path.join(user_profile, 'AppData', 'Roaming', 'Code', 'logs'),
            os.path.join(user_profile, 'AppData', 'Roaming', 'Code', 'User', 'workspaceStorage'),
            os.path.join(user_profile, '.vscode', 'extensions'),
        ])
    
    # macOS paths
    elif os.name == 'posix' and 'Darwin' in os.uname().sysname:
        home = os.path.expanduser('~')
        vscode_cache_paths.extend([
            os.path.join(home, 'Library', 'Application Support', 'Code', 'CachedData'),
            os.path.join(home, 'Library', 'Application Support', 'Code', 'logs'),
            os.path.join(home, 'Library', 'Application Support', 'Code', 'User', 'workspaceStorage'),
            os.path.join(home, '.vscode', 'extensions'),
        ])
    
    # Linux paths
    else:
        home = os.path.expanduser('~')
        vscode_cache_paths.extend([
            os.path.join(home, '.config', 'Code', 'CachedData'),
            os.path.join(home, '.config', 'Code', 'logs'),
            os.path.join(home, '.config', 'Code', 'User', 'workspaceStorage'),
            os.path.join(home, '.vscode', 'extensions'),
        ])
    
    print("\n" + "="*60)
    print("VS CODE CACHE CLEARING INSTRUCTIONS")
    print("="*60)
    print("\n1. CLOSE VS Code completely before clearing cache")
    print("\n2. Cache directories to clear:")
    
    for path in vscode_cache_paths:
        if os.path.exists(path):
            print(f"   ✓ {path}")
        else:
            print(f"   ✗ {path} (not found)")
    
    print("\n3. Manual steps:")
    print("   - Close VS Code completely")
    print("   - Delete the cache directories listed above")
    print("   - Restart VS Code")
    
    print("\n4. VS Code Commands (run in Command Palette):")
    print("   - 'Developer: Reload Window'")
    print("   - 'Developer: Reload Window With Extensions Disabled'")
    print("   - 'Python: Clear Cache and Reload Window'")
    
    print("\n5. Workspace-specific cache:")
    print("   - Close workspace")
    print("   - Delete .vscode folder in project root")
    print("   - Reopen workspace")
    
    print("="*60)
    
    return vscode_cache_paths


def clear_workspace_cache(workspace_path=None):
    """
    Clear workspace-specific cache files.
    
    Args:
        workspace_path: Path to workspace (default: current directory)
    """
    if workspace_path is None:
        workspace_path = os.getcwd()
    
    logger.info(f"Clearing workspace cache in: {workspace_path}")
    
    cache_patterns = [
        '.vscode',
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '.pytest_cache',
        '.coverage',
        'node_modules',
        '.DS_Store',
        'Thumbs.db'
    ]
    
    cleared_items = []
    
    for root, dirs, files in os.walk(workspace_path):
        # Clear cache directories
        for cache_dir in [d for d in dirs if d in cache_patterns]:
            cache_path = os.path.join(root, cache_dir)
            try:
                shutil.rmtree(cache_path)
                cleared_items.append(cache_path)
                logger.info(f"Cleared directory: {cache_path}")
            except Exception as e:
                logger.warning(f"Could not clear {cache_path}: {e}")
        
        # Clear cache files
        for file in files:
            if any(file.endswith(pattern.replace('*', '')) for pattern in cache_patterns if '*' in pattern):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    cleared_items.append(file_path)
                    logger.info(f"Cleared file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not clear {file_path}: {e}")
    
    logger.info(f"Workspace cache clearing completed. Cleared {len(cleared_items)} items.")
    return cleared_items


def adjust_learning_rate(optimizer, epoch, args):
    logger.debug(f"Adjusting learning rate at epoch {epoch}")
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, warmup_epochs=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.warmup_epochs = warmup_epochs

    def __call__(self, val_loss, model, path, epoch=None):
        score = -val_loss

        if epoch is not None and epoch < self.warmup_epochs:
            logger.info(f"Warmup epoch {epoch+1}/{self.warmup_epochs}. Skipping early stopping check.")
            if self.best_score is None or score > self.best_score - self.delta:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path)
            return

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    logger.info(f"Saving visual to {name}")
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
