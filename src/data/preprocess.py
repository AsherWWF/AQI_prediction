import os
import h5py
import numpy as np


from data import utils

DATA_PATH = '../data/data_all'
SAVE_PATH = '../data/'

data = utils.load_h5(os.path.join(DATA_PATH, 'data_17.h5'), ['data'])

station_map, station_loc, station_geo = utils.load_h5(os.path.join(DATA_PATH, 'geo.h5'),
													['station_map', 'station', 'geo_feature'])

def in_huabei(loc):
    return 34.109 < loc[0] < 41.691 and 110.938 < loc[1] < 122.321

def prop_missing(data):
    missing = np.isnan(data).sum()
    count = data.size
    return float(missing) / float(count)

index = []
n = station_map.shape[0]
for i in range(n):
    # print(i, data[:,i].shape, prop_missing(data[:,i,0]))
    if in_huabei(station_loc[i]) and prop_missing(data[:,i,0]) < 0.3:
        index.append(i)

# print(data[:,index].shape)
f = h5py.File(os.path.join(SAVE_PATH, 'data_17.h5'))
f.create_dataset('data', data=data[:,index])
f.close()

f = h5py.File(os.path.join(SAVE_PATH, 'geo.h5'))
f.create_dataset('station_map', data=station_map[index])
f.create_dataset('station', data=station_loc[index])
f.create_dataset('geo_feature', data=station_geo[index])
f.close()
