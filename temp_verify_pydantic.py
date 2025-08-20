import sys
print(sys.executable)
try:
    import pydantic
    print('pydantic ' + pydantic.__version__)
except ImportError:
    print('pydantic not installed')