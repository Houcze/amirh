import numpy as np
import matplotlib.pyplot as plt
import os


index = os.listdir('./result')

for rec in index:
    ds = np.genfromtxt('./result/{}'.format(rec))
    plt.imshow(ds)
    plt.axis('off')
    plt.savefig('./result-visual/{}.jpg'.format(rec.split('.')[0]))