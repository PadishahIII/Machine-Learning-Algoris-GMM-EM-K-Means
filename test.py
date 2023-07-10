import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("./UCIdata.csv", header=0, nrows=2000, encoding="gbk")
#print(data["a"])
a = data["b"]
b = data["c"]

aa = np.array(a)
bb = np.array(b)

c = np.array(data[["a", "b"]])

#print(c)

plt.scatter(aa, bb, marker='*', c='orange', label='class1')
plt.show()
