import json
from collections import Counter

import pandas as pd
import numpy as np
from vtkplotter import load


class VisualizerSampler:

    def __init__(self, np_pc_tested, np_scores, csv_file_scores, json_propagation_file, ply_obj_file):
        self.np_pc_tested = np_pc_tested
        self.np_scores = np_scores
        self.csv_file_scores = csv_file_scores
        self.json_file_propagation = json_propagation_file
        self.ply_obj_file = ply_obj_file
        self.votes = None
        self.pd_best_scores = None
        self.idx_votes = -1
        self.vtk_samples = []

    def filter_data_scores(self):
        with open(self.json_file_propagation) as f:
            propagation_data = json.load(f)

        max_score = propagation_data['max_limit_score']
        max_missing = propagation_data['max_limit_missing']

        pd_scores = pd.read_csv(self.csv_file_scores)

        filtered_df = pd_scores.loc[(pd_scores.score.notnull()) &  # avoiding null scores (bar normal environment)
                                    (pd_scores.missings <= max_missing) &  # avoiding scores with more than max missing
                                    (pd_scores.score <= max_score),
                                    pd_scores.columns != 'interaction']  # returning all columns but interaction name

        return filtered_df.loc[filtered_df.groupby(['point_x', 'point_y', 'point_z'])['score'].idxmin()]

    def get_sample(self):
        if self.votes is None:
            self.pd_best_scores = self.filter_data_scores()
            self.votes = self.generate_votes()

        while True:
            self.idx_votes += 1
            idx_sample = self.votes[self.idx_votes][0]
            point_sample = self.np_pc_tested[idx_sample]
            angle_sample = self.angle_with_best_score(x=point_sample[0], y=point_sample[1], z=point_sample[2])
            if angle_sample != -1:
                vtk_object = load(self.ply_obj_file)
                vtk_object.rotate(angle_sample, axis=(0, 0, 1), rad=True)
                vtk_object.pos(x=point_sample[0], y=point_sample[1], z=point_sample[2])
                self.vtk_samples.append(vtk_object)
                break
        return vtk_object

    def generate_votes(self):
        sum_mapped_norms = sum(self.np_scores)
        probabilities = [float(score) / sum_mapped_norms for score in self.np_scores]
        n_rolls = 10 * self.np_scores.shape[0]
        rolls = np.random.choice(self.np_scores.shape[0], n_rolls, p=probabilities)
        return Counter(rolls).most_common()

    def angle_with_best_score(self, x, y, z):
        angles = self.pd_best_scores[(self.pd_best_scores['point_x'].round(decimals=5) == round(x, 5)) &
                            (self.pd_best_scores['point_y'].round(decimals=5) == round(y, 5)) &
                            (self.pd_best_scores['point_z'].round(decimals=5) == round(z, 5))].angle

        return angles.array[0] if (angles.shape[0] == 1) else -1