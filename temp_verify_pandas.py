import sys
print(sys.executable)
try:
    import pandas as pd
    print('pandas ' + pd.__version__)
except ImportError:
    print('pandas not installed')