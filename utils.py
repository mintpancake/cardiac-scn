import json
import os
import math
import csv
from datetime import datetime
from PIL import Image
import numpy as np

STRUCTS = ['A2C-LV apex', 'A4C-LV apex', 'A4C-TV tip', 'ALAX-LV apex', 'Anterior mitral annulus', 'Anterolateral mitral annulus',
           'Anterolateral papillary muscle', 'Aortic annulus', 'Center of AV', 'IAS', 'IVS', 'IW', 'Interventricular septum',
           'LV', 'Lateral mitral annulus', 'MV anterior leaflet  A2', 'MV anterior leaflet  A3', 'MV anterior leaflet A1',
           'MV posterior leaflet P1', 'MV posterior leaflet P2', 'MV posterior leaflet P3', 'MV tip', 'Medial mitral annulus',
           'PV tip', 'Posterior mitral annulus', 'Posteromedial mitral annulus', 'Posteromedial papillary muscle', 'RV', 'RV apex',
           'SAXA-LV apex', 'SAXB-TV tip', 'Tricuspid annulus.']

VIEWS = ['2 chamber view (A2C)', '4 chamber view (A4C)', 'Apical LV short-axis view (SAXA)', 'Basal short-axis view (SAXB)',
         'Long-axis view (ALAX)', 'MV short-axis view (SAXMV)', 'Mid LV short-axis view (SAXM)']

VIEWS_ABBR = ['A2C', 'A4C', 'SAXA', 'SAXB', 'ALAX', 'SAXMV', 'SAXM']

VIEW_STRUCTS = {
    'A2C': [0, 5, 25],
    'A4C': [1, 2, 14, 21, 22, 31],
    'SAXA': [12, 28, 29],
    'SAXB': [8, 9, 23, 30],
    'ALAX': [3, 4, 7, 24],
    'SAXMV': [15, 16, 17, 18, 19, 20],
    'SAXM': [6, 10, 11, 13, 26, 27]
}


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_config(path):
    with open(path) as json_file:
        config = json.load(json_file)
    print(json.dumps(config, indent=2))
    return config


def update_config(path, config):
    with open(path, 'w') as json_file:
        json.dump(config, json_file, indent=4)


def current_time():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_struct_name(idx):
    return STRUCTS[idx]


def get_struct_idx(name):
    return STRUCTS.index(name)


def get_view_name(idx):
    return VIEWS[idx]


def get_view_abbr(idx):
    return VIEWS_ABBR[idx]


def get_view_index(name, type='abbr'):
    if type == 'name':
        return VIEWS.index(name)
    elif type == 'abbr':
        return VIEWS_ABBR.index(name)


def read_ratio(path):
    ratios = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            ratios[row[0]] = float(row[3])
    return ratios


def draw(data, filename, mode='clip'):
    path = os.path.split(filename)[0]
    ensure_dir(path)

    if mode == 'clip':
        data[data < 0.0] = 0.0

    int_data = (((data - data.min()) / (data.max() - data.min()))
                * 255.9).astype(np.uint8)
    image = Image.fromarray(int_data)
    image.save(filename)


def fit_plane(xyz):
    centroid = xyz.mean(axis=0)
    xyzR = xyz - centroid
    u, sigma, v = np.linalg.svd(xyzR)
    normal = v[2]
    if normal[2] < 0.0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    return (centroid, normal)


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2, directed=False):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/math.pi*180
    if not directed and angle > 90.0:
        angle = 180.0-angle
    return angle


def distance_along_direction(x, y, u):
    xy = y-x
    u = unit_vector(u)
    d = np.dot(xy, u)*u
    return np.linalg.norm(d)


def distance_to_plane(points, centroid, normal):
    n = unit_vector(normal)
    v = points-centroid
    d = np.abs(v@n.T)
    return d
