import h5py

def writeGreen(fileName, dataName, dataToWrite):
    file = h5py.File(fileName, 'w')
    file.create_dataset(dataName, data=dataToWrite)
    file.close()