# 1. Start from the 'latest' tag, which we know works.
FROM rocm/pytorch:latest
    
# 2. Set your working directory
WORKDIR /workspace/Time-Series-Library

# 3. Copy your requirements file
COPY requirements.txt ./

# 4. Install your packages
# - We add "--no-build-isolation" to fix the 'ModuleNotFoundError: No module named 'torch''
# - We remove all the complex URLs and let pip find the correct versions
RUN pip install --no-build-isolation \
    torch-geometric \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
 && pip install -r requirements.txt