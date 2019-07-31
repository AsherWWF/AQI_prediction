import os
import h5py
import logging
import mxnet as mx
import numpy as np
import pandas as pd
import math

from data import utils
from config import DATA_PATH, TRAIN_PROP, EVAL_PROP

def city_of_station(station_map):
	n = station_map.shape[0]
	city_id = np.unique(station_map // 1000)
	city = np.zeros((n, len(city_id)))

	idx = 0
	for i in range(n):
		while city_id[idx] != station_map[i] // 1000: idx += 1
		city[i, idx] = 1

	return city    # [num_station, num_city]

def geo_transform(station_geo):
	# geo = (station_geo - np.mean(station_geo, axis=0)) / np.std(station_geo, axis=0)
	poi = station_geo[:, :20]
	road = station_geo[:, 20:]
	
	poi_cnt = np.sum(poi, axis=1, keepdims=True)
	poi = poi / (poi_cnt + 1e-8)

	geo = np.concatenate((poi_cnt, road), axis=1)
	geo = (geo - np.mean(geo, axis=0)) / (np.std(geo, axis=0) + 1e-8)

	geo = np.concatenate((poi, geo), axis=1)
	return geo  #[num_station, num_geo_feature (24)]

def get_geo_feature(dataset):
	station_map, station_loc, station_geo = utils.load_h5(os.path.join(DATA_PATH, 'geo.h5'),
													['station_map', 'station', 'geo_feature'])

	station_geo[np.isnan(station_geo)] = 0
	loc = (station_loc - np.mean(station_loc, axis=0)) / np.std(station_loc, axis=0)  #[num_station, 2]
	city = city_of_station(station_map)   # [num_station, num_city]
	geo = geo_transform(station_geo)  #[num_station, num_geo_feature]

	# feature = np.concatenate((loc, city, geo), axis=1)
	feature = np.concatenate((loc, geo), axis=1)  #[num_station, num_geo_feature (26)]
	# feature = loc

	graph = utils.build_graph(station_map, station_loc, city, dataset['n_neighbors'])
	return feature, graph   #[num_station, num_geo_feature (26)],   #[n, n], list, list

def dataloader(dataset):
	data = utils.load_h5(os.path.join(DATA_PATH, 'data_17.h5'), ['data'])
	data = data[-90 * 24:]
	data[data > 500] = np.nan

	n_timestamp = data.shape[0]
	num_train = int(n_timestamp * TRAIN_PROP)
	num_eval = int(n_timestamp * EVAL_PROP)
	num_test = n_timestamp - num_train - num_eval

	return data[:num_train], data[num_train: num_train + num_eval], data[-num_test:]


def dataiter_all_sensors_seq2seq(aqi, scaler, setting, shuffle=True):
	dataset = setting['dataset']
	training = setting['training']

	aqi_fill = utils.fill_missing(aqi)
	aqi_fill = scaler.transform(aqi_fill)   #[T, N, D]

	n_timestamp, num_nodes, _ = aqi_fill.shape

	timespan = (np.arange(n_timestamp) % 24) / 24
	timespan = np.tile(timespan, (1, num_nodes, 1)).T
	aqi_fill = np.concatenate((aqi_fill, timespan), axis=2)    #[T, N, D]  add time of day 

	geo_feature, _ = get_geo_feature(dataset)  #[num_station, num_geo_feature (26)]

	data_fill = np.concatenate((aqi_fill, np.tile(np.expand_dims(geo_feature, axis=0), (n_timestamp, 1, 1))), axis=2)

	input_len = dataset['input_len']
	output_len = dataset['output_len']
	feature, data, mask, label  = [], [], [], []
	for i in range(n_timestamp - input_len - output_len + 1):
		data.append(aqi_fill[i: i + input_len])

		mask.append(1.0 - np.isnan(aqi[i + input_len: i + input_len + output_len,:,0]).astype(float))

		label.append(aqi_fill[i + input_len: i + input_len + output_len])
		
		feature.append(geo_feature)

		if i % 1000 == 0:
			logging.info('Processing %d timestamps', i)
			# if i > 0: break

	data = mx.nd.array(np.stack(data)) # [B, T, N, D(33)]
	label = mx.nd.array(np.stack(label)) # [B, T, N, D]
	mask = mx.nd.array(np.expand_dims(np.stack(mask), axis=3)) # [B, T, N, 1]
	feature = mx.nd.array(np.stack(feature)) # [B, N, D]
	

	logging.info('shape of feature: %s', feature.shape)
	logging.info('shape of data: %s', data.shape)
	logging.info('shape of mask: %s', mask.shape)
	logging.info('shape of label: %s', label.shape)

	from mxnet.gluon.data import ArrayDataset, DataLoader
	return DataLoader(
		ArrayDataset(feature, data, label, mask),
		shuffle		= shuffle,
		batch_size	= training['batch_size'],
		num_workers	= 4,
		last_batch	= 'rollover',
	)

def dataloader_all_sensors_seq2seq(setting):
	train, eval, test = dataloader(setting['dataset'])  #[T, N, D]
	scaler = utils.Scaler(train)

	return dataiter_all_sensors_seq2seq(train, scaler, setting), \
		   dataiter_all_sensors_seq2seq(eval, scaler, setting, shuffle=False), \
		   dataiter_all_sensors_seq2seq(test, scaler, setting, shuffle=False), \
		   scaler