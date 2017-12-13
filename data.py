import os
import numpy as np
import pandas as pd
import imageio as io


class JesterDataSet:
    N_CLASSES = 27

    def __init__(self, partition='train', classes=None, proportion=1.0, size=(30, 100, 100, 3),
                 data_path=os.path.join(os.path.dirname(__file__), 'data', '20BN-JESTER')):
        """
        Container for the 20BN-JESTER dataset. Note that both the data and the labels have to be downloaded manually
        into the directory specified as an argument. The data can be found at https://www.twentybn.com/datasets/jester.

        Arguments:
            partition: value in {'train', 'validation', 'test'}
            classes: classes that should be loaded. Either None, in which case all of the classes will be loaded,
                an int n, in which case n first classes will be loaded, or a list of ints representing IDs, in which
                case the classes with the specified IDs will be loaded
            proportion: proportion of data that should be loaded for each class
            data_path: directory containing the data
        """
        assert partition in ['train', 'validation', 'test']
        assert classes is None \
               or (type(classes) is int and classes <= self.N_CLASSES) \
               or (min(classes) >= 0 and max(classes) < self.N_CLASSES)
        assert 0 < proportion <= 1.0
        assert os.path.exists(data_path)

        if classes is None:
            classes = range(0, 27)
        elif type(classes) is int:
            classes = range(0, classes)

        df = pd.read_csv(os.path.join(data_path, 'jester-v1-labels.csv'), header=None)

        self.label_names = list(df.loc[classes, 0])
        self.label_dictionary = {}

        for cls in range(len(classes)):
            self.label_dictionary[self.label_names[cls]] = cls

        df = pd.read_csv(os.path.join(data_path, 'jester-v1-%s.csv' % partition), header=None, delimiter=';')

        if partition in ['train', 'validation']:
            grouped_video_ids = {}

            for cls in range(len(classes)):
                grouped_video_ids[cls] = []

            for row in df.iterrows():
                video_id, label_name = row[1]

                if label_name not in self.label_names:
                    continue

                label = self.label_dictionary[label_name]

                grouped_video_ids[label].append(video_id)

            if proportion < 1.0:
                for cls in range(len(classes)):
                    n = int(proportion * len(grouped_video_ids[cls]))

                    assert n > 0

                    grouped_video_ids[cls] = grouped_video_ids[cls][:n]

            self.video_ids = sum(grouped_video_ids.values(), [])
            self.labels = sum([[cls] * len(grouped_video_ids[cls]) for cls in range(len(classes))], [])
        else:
            self.video_ids = []

            for row in df.iterrows():
                video_id = row[1][0]

                self.video_ids.append(video_id)

            self.labels = None

    def shuffle(self):
        pass

    def load_video(self, video_id):
        pass
