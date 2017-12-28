import argparse

from model import C2DNetwork, C3DNetwork
from data import JesterDataSet, FlattenedJesterDataSet
from trainer import Trainer


params = {
    'network': 'C2D',
    'axis': 0,
    'name': None,
    'learning_rate': 0.0001,
    'batch_size': 10,
    'epochs': 20,
    'shape': [30, 100, 100, 3],
    'verbose': True,
    'data_path': None,
    'early_stopping': True,
    'proportion': 1.0,
    'classes': None
}

parser = argparse.ArgumentParser()

for k in params.keys():
    parser.add_argument('-%s' % k)

args = vars(parser.parse_args())

for k, v in params.items():
    if args.get(k) is not None:
        params[k] = eval(args.get(k))

if params['name'] is None:
    params['name'] = params['network']

    if params['network'] == 'C2D':
        params['name'] += '_%d' % params['axis']

dataset_params = {key: params[key] for key in ['shape', 'batch_size', 'data_path', 'verbose', 'proportion', 'classes']}
trainer_params = {key: params[key] for key in ['epochs', 'learning_rate', 'early_stopping', 'verbose']}

if params['network'] == 'C2D':
    DataSet = FlattenedJesterDataSet
    Network = C2DNetwork

    dataset_params['axis'] = params['axis']
else:
    DataSet = JesterDataSet
    Network = C3DNetwork

train_set = DataSet(partition='train', preload=False, **dataset_params)
validation_set = DataSet(partition='validation', preload=True, **dataset_params)
network = Network(input_shape=train_set.flattened_shape, output_shape=[train_set.N_CLASSES], name=params['name'])
trainer = Trainer(network, train_set, validation_set=validation_set, **trainer_params)
trainer.train()
