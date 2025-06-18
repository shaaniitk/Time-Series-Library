import re

# Read the file
with open('data_provider/data_loader.py', 'r') as f:
    content = f.read()

# Replace the problematic syntax
content = content.replace("df_stamp.drop(['date'], 1).values", "df_stamp.drop(['date'], axis=1).values")

# Write back
with open('data_provider/data_loader.py', 'w') as f:
    f.write(content)

print("Fixed all pandas drop() calls")
