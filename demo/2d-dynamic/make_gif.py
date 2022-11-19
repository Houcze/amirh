import imageio
import os

def compose_gif():
    index = os.listdir('./result-visual')
    gindex = []
    for p in index:
        gindex.append(imageio.imread('./result-visual/' + p))
    imageio.mimsave('test.gif', gindex, fps=3)

compose_gif()