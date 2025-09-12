# Debugging Troubleshooting Guide

## Common Issues When Breakpoints Don't Work

### 1. Python Interpreter Issues
- **Check VS Code Python Interpreter**: Press `Ctrl+Shift+P` → "Python: Select Interpreter"
- **Ensure correct environment**: Should be `tsl-env` or the virtual environment you're using
- **Verify interpreter path**: Should point to your virtual environment's Python executable

### 2. VS Code Settings Check

#### Required VS Code Extensions:
- Python extension (ms-python.python)
- Pylance (ms-python.vscode-pylance)

#### Settings to verify in VS Code settings.json:
```json
{
    "python.defaultInterpreterPath": "./tsl-env/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "python.debugging.console": "integratedTerminal"
}
```

### 3. Debugging Steps

#### Step 1: Verify Python Path
1. Open VS Code terminal
2. Run: `python -c "import sys; print(sys.executable)"`
3. Should show path to your virtual environment

#### Step 2: Test Simple Breakpoint
1. Open `test_breakpoint.py`
2. Set breakpoint on line 15: `print("🔍 Testing breakpoint functionality...")`
3. Press F5 → Select "Python: Current File"
4. Debugger should stop at breakpoint

#### Step 3: Check Debug Console
- When debugging, check the "Debug Console" tab
- Look for any error messages or warnings

### 4. Alternative Debugging Methods

#### Method 1: Use debugpy directly
```python
import debugpy
debugpy.listen(5678)
print("Waiting for debugger...")
debugpy.wait_for_client()
print("Debugger attached!")
# Your code here
```

#### Method 2: Use pdb (Python debugger)
```python
import pdb
pdb.set_trace()  # This will create a breakpoint
```

### 5. VS Code Launch Configuration Fixes

If breakpoints still don't work, try this minimal configuration:

```json
{
    "name": "Debug: Simple Python",
    "type": "debugpy",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal",
    "justMyCode": false,
    "cwd": "${workspaceFolder}",
    "env": {
        "PYTHONPATH": "${workspaceFolder}"
    }
}
```

### 6. Common Solutions

1. **Restart VS Code**: Sometimes a simple restart fixes debugger issues
2. **Reload Window**: `Ctrl+Shift+P` → "Developer: Reload Window"
3. **Clear VS Code cache**: Close VS Code, delete `.vscode` folder, restart
4. **Reinstall Python extension**: Uninstall and reinstall the Python extension

### 7. Diagnostic Commands

Run these in VS Code terminal to diagnose issues:

```bash
# Check Python version and path
python --version
which python  # On Windows: where python

# Check if debugpy is installed
python -c "import debugpy; print(debugpy.__version__)"

# Test basic debugging
python -m debugpy --listen 5678 --wait-for-client test_breakpoint.py
```

### 8. Last Resort Solutions

1. **Create new workspace**: Sometimes workspace settings get corrupted
2. **Use different debug configuration**: Try "Python: Module" instead of "Python: Current File"
3. **Check Windows Defender**: Sometimes antivirus blocks debugger
4. **Run VS Code as Administrator**: May help with permission issues

## Quick Test Checklist

- [ ] Correct Python interpreter selected
- [ ] Python extension installed and enabled
- [ ] Breakpoint set on executable line (not comment/blank line)
- [ ] Using F5 to start debugging (not Ctrl+F5)
- [ ] "justMyCode": false in launch.json
- [ ] No syntax errors in the file
- [ ] File is saved before debugging

## Still Not Working?

If none of the above solutions work:
1. Check VS Code Developer Tools: `Help` → `Toggle Developer Tools`
2. Look for JavaScript errors in the console
3. Try debugging a simple "Hello World" script first
4. Consider using PyCharm or another IDE temporarily