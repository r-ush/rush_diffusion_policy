import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
line = ax.axvline(x=5, color='r')
print("Before:", line.get_xdata())
line.set_xdata([10, 10])
print("After:", line.get_xdata())
