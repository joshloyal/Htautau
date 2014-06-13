#from rootpy.tree import Tree, TreeModel, FloatCol, IntCol
#from rootpy.io import root_open

import csv as csv
import numpy as np

def csv_to_d3pd():
    training_file = open( './training.csv', 'rb' )
    csv_file_object = csv.reader( training_file )
    header = csv_file_object.next()

    data = []
    for row in csv_file_object:
        data.append(row)
    data = np.array(data)

csv_to_d3pd()
