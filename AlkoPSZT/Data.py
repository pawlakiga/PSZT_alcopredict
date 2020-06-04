import os
import pandas
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
        self.data = data

    def set_classes(self, column_name):
        if column_name != 'Walc' and column_name != 'Dalc':
            return -1
        else:
            self.classes = sort(self.data[column_name].unique())
            self.class_column_name = column_name
            return 1

    def set_attributes(self):
        self.attributes = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                           'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
                           'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                           'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'health', 'absences', 'G1',
                           'G2', 'G3']

    def get_all_attribute_values(self, column_name):
        return self.data[column_name].unique()

    def get_object_attribute_value(self, index, column_name):
        return self.data.loc[index, column_name]

    def get_object_class(self, index):
        return self.data.loc[index, self.class_column_name]


students_data = StudentsData()
students_data.set_classes('Dalc')
print(students_data.classes)
print(students_data.data.columns)
