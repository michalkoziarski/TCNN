import logging
import argparse

from model import C2DNetwork, C3DNetwork, MultiStreamNetwork
from data import JesterDataSet, FlattenedJesterDataSet, MultiStreamJesterDataSet
from trainer import Trainer


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('-network', type=str, choices=['C2D', 'C3D', 'MultiStream'], required=True)
parser.add_argument('-axis', type=int, choices=[0, 1, 2])
parser.add_argument('-name', type=str)
parser.add_argument('-learning_rate', type=float, default=0.0001)
parser.add_argument('-batch_size', type=int, default=10)
parser.add_argument('-epochs', type=int, default=20)
parser.add_argument('-shape', type=list, default=[30, 100, 100, 3])
parser.add_argument('-verbose', type=bool, default=True)
parser.add_argument('-data_path', type=str)
parser.add_argument('-early_stopping', type=bool, default=True)
parser.add_argument('-proportion', type=float, default=1.0)
parser.add_argument('-classes', type=int)
parser.add_argument('-preload', type=bool, default=False)

params = vars(parser.parse_args())

if params['name'] is None:
    params['name'] = params['network']

    if params['network'] == 'C2D':
        params['name'] += '_%d' % params['axis']

dataset_params = {key: params[key] for key in ['shape', 'batch_size', 'data_path', 'verbose', 'proportion', 'classes', 'preload']}
trainer_params = {key: params[key] for key in ['epochs', 'learning_rate', 'early_stopping', 'verbose']}

if params['network'] == 'C2D':
    DataSet = FlattenedJesterDataSet
    Network = C2DNetwork

    dataset_params['axis'] = params['axis']
elif params['network'] == 'C3D':
    DataSet = JesterDataSet
    Network = C3DNetwork
else:
    DataSet = MultiStreamJesterDataSet
    Network = MultiStreamNetwork

train_set = DataSet(partition='train', **dataset_params)
validation_set = DataSet(partition='validation', **dataset_params)

if params['network'] == 'C2D':
    input_shape = train_set.flattened_shape
elif params['network'] == 'C3D':
    input_shape = train_set.shape
else:
    input_shape = train_set.input_shape

network = Network(input_shape=input_shape, output_shape=[train_set.N_CLASSES], name=params['name'])
trainer = Trainer(network, train_set, validation_set=validation_set, **trainer_params)
trainer.train()
