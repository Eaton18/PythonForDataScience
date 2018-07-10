
from os import path
d = path.dirname(__file__)
# __file__ is current file, #d = path.dirname('.')
print("Current file:\t", __file__)
print("Current file path:\t", d)

parent_path = path.dirname(d)
print("Current file father's path:\t", parent_path)

parent_parent_path = path.dirname(parent_path)
print("Current file father's father's path:\t", parent_parent_path)

abspath = path.abspath(d) # absolute path
print("Current file absolute path\t", abspath)
