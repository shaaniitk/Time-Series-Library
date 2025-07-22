# D:\workspace\Time-Series-Library\test_input.py

import sys

print("Hello from test_input.py!")
sys.stdout.flush() # Ensure this is printed immediately

try:
    user_input = input("Please type something and press Enter: ")
    print(f"You typed: {user_input}")
    sys.stdout.flush()
except EOFError:
    print("EOFError: Input stream closed unexpectedly. This usually means the script is not running in an interactive terminal.")
    sys.stdout.flush()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.stdout.flush()

print("Test script finished.")
sys.stdout.flush()
