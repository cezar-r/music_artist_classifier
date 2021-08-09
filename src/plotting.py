import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
plt.style.use("dark_background")
plt.rcParams["figure.figsize"] = (18, 7)


x = np.linspace(5, 115, 23).tolist()
x.insert(0, 0)

y = [1/336, 0.0524388751987976, 0.08529487954408083, 0.11531964989507089, 0.14301608172347527, 0.1652851251718249, 0.18302688809806247, 0.2037279826711509, 0.21904577940892409, 0.2397992111868053, 0.25188976377952754, 0.26058631921824105,
0.2794158424993068, 0.28891977760127086, 0.2944843617920541, 0.2971217381565685, 0.30105362850716233, 0.30648769574944074, 0.31234519619345585, 0.3236576723902971, 0.3257963594994312, 0.3300740740740741,
0.3308721735117674, 0.33958962939398757]

plt.title("Accuracy v. Snippet Length", fontsize=20)
plt.bar(x, y, width = 3, color='lightcoral')
plt.ylabel("Accuracy", fontsize=15)
plt.xlabel("Snippet Length", fontsize=15)
plt.savefig("../images/accuracy_vs_snippet_bar2.png")
plt.show()

# ysmoothed = gaussian_filter1d(y, sigma=2)
# plt.title("Accuracy v. Snippet Length")
# plt.ylabel("Accuracy", fontsize=12)
# plt.xlabel("Snippet Length", fontsize=12)
# plt.plot(x, ysmoothed, color='lightcoral', linewidth=4)


# plt.savefig("../images/accuracy_vs_snippet_line.png")


# 28.7
# 113.9