import os
import pandas
import numpy as np
from numpy import sort


class StudentsData:

    def __init__(self):
        data_mat = pandas.read_csv(
            'C:\\Users\\Iga\\Documents\\GitHub\\PSZT_alcopredict\\student-alcohol-consumption\\student-mat.csv')
        data_port = pandas.read_csv(
            'C:\\Users\\Iga\\Documents\\GitHub\\PSZT_alcopredict\\student-alcohol-consumption\\student-por.csv')
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


name = "F"
data = StudentsData()
print(data.objects.query('sex=='+"\""+name+"\""))

#print(students_data.classes)
#print(students_data.objects.loc[4].loc['sex'])
#print(list(students_data.objects.columns))
#students_data.attributes.remove('school')
#print(students_data.attributes)
#print(students_data.objects.columns.remove('sex'))

#print(len(students_data.class_members(1)))
