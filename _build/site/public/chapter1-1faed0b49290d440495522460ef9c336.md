```python

import numpy as np
import matplotlib.pyplot as plt

mupos = 90
muneg = 70
sigma = 20
Pos = 50
Neg = 50

px = np.random.normal(mupos, sigma, Pos)
nx = np.random.normal(muneg, sigma, Neg)

bins = np.arange(muneg - 2 * sigma, mupos + 2 * sigma + 10, 10)

counts, xout = np.histogram(np.concatenate((px, nx)), bins)

plt.style.use('ggplot')

plt.bar(xout[:-1], counts, width=10, align='edge',  edgecolor = "black")
plt.savefig("light-plot.png")
plt.close()

plt.style.use('dark_background')

plt.bar(xout[:-1], counts, width=10, align='edge',  edgecolor = "black")
plt.savefig("dark-plot.png")
plt.close()

counts = counts.reshape(-1, 1) 
p = counts[:, 0] / (counts[:, 0] + counts[:, 0])  

TP = 0
FP = 0
tp = [0]
fp = [0]

for i in range(len(counts)):
    tp.append(TP)
    fp.append(FP)
    TP += counts[i, 0]
    FP += counts[i, 0]

tp.append(TP)
fp.append(FP)

plt.style.use('ggplot')
plt.plot(fp, tp, marker='o')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig("light-plot-rates.png")
plt.close()

plt.style.use('dark_background')
plt.plot(fp, tp, marker='o')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig("dark-plot-rates.png")
plt.close()

counts2 = np.zeros((6, 1))
counts2[0] = counts[0] + counts[1]
counts2[1] = counts[2] + counts[3]
counts2[2] = counts[4] + counts[5]
counts2[3] = counts[6]
counts2[4] = counts[7] + counts[8]
counts2[5] = counts[9] + counts[10] if len(counts) > 10 else counts[9]

bins2 = [35, 55, 75, 90, 110, 130]

plt.style.use('ggplot')
plt.bar(bins2, counts2.flatten(), width=10, align='center')
plt.savefig("fig-3-light.png")
plt.close()

plt.style.use('dark_background')
plt.bar(bins2, counts2.flatten(), width=10, align='center')
plt.savefig("fig-3-dark.png")
plt.close()

```