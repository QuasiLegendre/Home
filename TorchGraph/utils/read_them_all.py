from os import listdir
from os.path import isfile, join
from .ssv2Data import ssv2Data
def ReadThemAll(file_path):
    files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    dataset = {}
    for file in files:
        person = int(file.split('_')[0])
        dataset[person] = {}
    for file in files:
        person, num = file.split('.')[0].split('_')
        person = int(person)
        num = int(num)
        dataset[person][num] = ssv2Data(file_path+file)
    for person in dataset.keys():
        gl = []
        keys = sorted(dataset[person].keys())
        for key in keys:
            gl.append(dataset[person][key])
        dataset[person] = gl
    return dataset