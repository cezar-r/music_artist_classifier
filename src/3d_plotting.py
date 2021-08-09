import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")

fig = plt.figure()
ax = plt.axes(projection='3d')

zdata = [4, 4, 5, 4, 5, 4, 5, 4, 5, 4]
xdata = [0, 15, 15, 20, 20, 25, 25, 30, 30, 50]
ydata = [26, 34, 40, 45, 45, 49, 50, 52, 53, 60]
ax.set_xlabel("Snippet length")
ax.set_ylabel("ngrams")
ax.set_zlabel("accuracy")

ax.scatter3D(xdata, zdata, ydata, color = 'mediumblue')
plt.show()




'''
nip_len = None, explode = False,  over_ampling = True, ngrams = (1, 4), max_features = None, min_df = None
26% accuracy 6.5/10 

snip_len = 15, over_sampling = True, ngrams = (1, 4), max_features = None, min_df = None
34% accuracy = 6/10

snip_len = 15, over_sample = True, ngrams = (1, 5),max_featues = None, min_df = None
40% accuracy 6.5/10

snip_len = 15, over_sample = True, ngrams = (1, 10), max_feature = None, min_df = None
42% accuracy

snip_len = 20, over_sample = True, ngram = (1, 4)
45% accuracy 7/10 9gb model

snip_len = 20, over_sample = True, ngram = (1, 5)
45% accuracy 6/10 12 gb model

snip_len 25, over_sample = True, ngram = (1, 4)
49% accuracy 7/10

snip_len = 25, over_sample = True, ngram = (1, 5)
50% accuracy 7/10

snip_len = 30, over_sample = True, ngram = (1, 4)
52% acccuracy 7/10

snip_len - 30, over_sample = True, ngram = (1, 5)
53% accuracy 7/10

snip_len = 50, over_sample = True, ngram = (1, 4)
60% accuracy 
'''