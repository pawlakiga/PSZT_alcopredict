"""
Pawlak Iga, Michalski Marcin
PSZT Projekt 2
Alcohol Consumption Prediction
Data
"""

import os
import pandas
import numpy as np
from numpy import sort

"""
Class representing data used for classification
Fields:     objects - DataFrame
            attributes - names of the columns of the DataFrame that don't hold class values 

Methods:    load_data(file1_path, file2_path) - load data from .csv files at given file paths

"""


class StudentsData:

    def __init__(self):
        self.objects = None
        self.attributes = list()

    def load_data(self, file1_path, file2_path=None):
        data_mat = pandas.read_csv(file1_path)
        data_port = pandas.read_csv(file2_path)
        data = data_mat.append(data_port, ignore_index=True)
        data.drop_duplicates(
            subset=['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                    'Mjob', 'Fjob', 'reason', 'nursery', 'internet'],
            inplace=True, ignore_index=True)
        data.sort_index(inplace=True, kind='mergesort')
        self.objects = data
        self.attributes = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                           'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
                           'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                           'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'health', 'absences', 'G1',
                           'G2', 'G3']
