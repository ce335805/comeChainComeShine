import h5py

def readGreen(fileName, dataSet):

    file = h5py.File(fileName, 'r')
    data = file[dataSet][()]
    return data