from tqdm import tqdm
from sh import wgrib2
import os


index = os.listdir('./ds')
for ds in tqdm(index):
    wgrib2('./ds/{}'.format(ds), '-netcdf', './ds/{}.nc'.format(ds.split('.')[0]))