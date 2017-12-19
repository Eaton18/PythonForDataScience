
import os
import sys

# from Utilities.ImportModule.aaa import PrintHello

print("Test Module import.")

print(os.getcwd())
print(os.path.dirname(os.getcwd()))
print(os.path.dirname(os.path.dirname(os.getcwd())))
print(__file__)

# workspace_dir = os.path.dirname(os.path.dirname(os.getcwd()))
workspace_dir = os.getcwd()
sys.path.append(workspace_dir)

from aaa import PrintHello

PrintHello()
