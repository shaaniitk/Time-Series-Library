Interpreter note:

Tests previously failed on imports (yaml, psutil, patoolib) because system Python 3.13 was used instead of project venv.
Activate environment before running pytest:

PowerShell:
  .\tsl-env\Scripts\Activate.ps1
Then:
  python -m pytest -m extended

Or explicitly:
  .\tsl-env\Scripts\python.exe -m pytest -m extended

Consider setting VSCode Python interpreter to tsl-env/ to avoid mismatch.
