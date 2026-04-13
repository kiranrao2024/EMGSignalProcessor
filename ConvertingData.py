
# Switch between whether you want to convert EMG or Glove data by changing the file names in the code below. Change variable name if you want to as well.
# data organization:
#   || The EMG/glove data is organized in a hierarchical structure as follows: ||

#└── ProcessedEMG                                                   datatype = dictionary

 #└── s1 ... s27 # 27 subjects                                      datatype = dictionary

 #└── S1_A1_E1 ... # Subject/Attempt/Exercise                       datatype = dictionary

 #└── Restimulus_1 ... Restimulus_N # Gesture classes               datatype = dictionary

 #└── Rerepetition_1 ... Rerepetition_10 # Trial repetitions        datatype = dictionary

 #└── (n_samples × 10) numpy array # Raw EMG data                   datatype = numpy.ndarray

# data file
#   || The data is stored in a file called EMGdata.pkl, which is a file format used that can store the variable exactly as how it was created ||
# use the following code to load the data from the .pkl file:

#with open('DB1processed.pkl', 'rb') as f:
#    data = pickle.load(f)


import numpy as np
import scipy.io
import pickle


def convert_scipy(obj):
    if isinstance(obj, np.ndarray):
        if obj.shape == ():  # 0-dimensional array
            return convert_scipy(obj.item())
        elif obj.dtype.names:  # it's a struct
            return {name: convert_scipy(obj[name]) for name in obj.dtype.names}
        elif obj.shape == (1, 1):
            return convert_scipy(obj[0, 0])
        elif obj.ndim == 1 and obj.shape[0] == 1:
            return convert_scipy(obj[0])
        else:
            return obj  # numpy matrix, keep as-is
    else:
        return obj

def load_mat(filepath):
    try:
        raw = scipy.io.loadmat(filepath, struct_as_record=True, squeeze_me=False)
        data = {k: convert_scipy(v) for k, v in raw.items() if not k.startswith('__')}
        print("Loaded with scipy.io")
        return data
    except Exception as e:
        print(f"scipy failed ({e}), trying h5py...")
        with h5py.File(filepath, 'r') as f:
            data = convert_hdf5(f)
        print("Loaded with h5py")
        return data

Glovedata = load_mat('DB1processedGlove.mat')

with open('DB1processedGlove.pkl', 'wb') as f:
    pickle.dump(Glovedata, f)

data = Glovedata['']