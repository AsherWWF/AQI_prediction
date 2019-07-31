import os
import h5py
import numpy as np
import pandas as pd

from config import DATA_PATH

class Scaler:
	def __init__(self, data):
		self.mean = np.nanmean(data, axis=(0,1))
		self.std = np.nanstd(data, axis=(0,1))
	
	def transform(self, data):
		d = data.shape[2]
		return (data - self.mean[:d]) / self.std[:d]
	
	def inverse_transform(self, data):
		std = self.std[0]
		mean = self.mean[0]
		return data * std + mean

def load_h5(filename, keywords):
	f = h5py.File(filename, 'r')
	data = []
	for name in keywords:
		data.append(np.array(f[name]))
	f.close()
	if len(data) == 1:
		return data[0]
	return data

def get_distance_matrix(loc):
	n = loc.shape[0]
	
	loc_1 = np.tile(np.reshape(loc, (n,1,2)), (1,n,1)) * np.pi / 180.0   #[n, n, 2]
	loc_2 = np.tile(np.reshape(loc, (1,n,2)), (n,1,1)) * np.pi / 180.0   #[n, n, 2]

	loc_diff = loc_1 - loc_2
	
	dist = 2.0 * np.arcsin(
		np.sqrt(np.sin(loc_diff[:,:,0] / 2) ** 2 + np.cos(loc_1[:,:,0]) * np.cos(loc_2[:,:,0]) * np.sin(loc_diff[:,:,1] / 2) ** 2)
	)
	dist = dist * 6378.137 * 10000000 / 10000
	return dist   #[n, n]

def build_graph(station_map, station_loc, city, n_neighbors):
	dist = get_distance_matrix(station_loc)

	n = station_map.shape[0]
	src, dst = [], []
    
	# designed for linking stations in each city
	# ----------------------------------------------------------
    num_nodes_each_city = np.sum(city, axis=0)  #[num_city]
	
	low = 0
	high = num_nodes_each_city[0]
	for city_index in range(num_nodes_each_city.shape[0]):
		list_local_stations = range(low, high)
		for i in list_local_stations:
			src += list_local_stations
			dst += [i] * len(list_local_stations)
		if city_index < num_nodes_each_city.shape[0] -1:
		    low += num_nodes_each_city[city_index]
			high += num_nodes_each_city[city_index + 1]  
	# -----------------------------------------------------------



	# designed for k nearest neighbor
	# -------------------------------------------------------------
	# for i in range(n):
	# 	src += list(np.argsort(dist[:, i])[:n_neighbors + 1])
	# 	dst += [i] * (n_neighbors + 1) 
    # -------------------------------------------------------------

	mask = np.zeros((n, n))
	mask[src, dst] = 1
	dist[mask == 0] = np.inf

	values = dist.flatten()
	values = values[values != np.inf]

	dist_mean = np.mean(values)
	dist_std = np.std(values)
	dist = np.exp(-(dist - dist_mean) / dist_std)

	return dist, src, dst  #[n, n], list, list

def fill_missing(data):
	T, N, D = data.shape
	data = np.reshape(data, (T, N * D))
	df = pd.DataFrame(data)
	df = df.fillna(method='pad')
	df = df.fillna(method='bfill')
	data = df.values
	data = np.reshape(data, (T, N, D))
	data[np.isnan(data)] = 0
	return data