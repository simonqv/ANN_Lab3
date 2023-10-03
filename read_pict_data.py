import numpy as np

"""
Read the pict.dat and structure into 9 32x32 arrays.
"""
def read_pict_data():
    
    # Read pict data and put in array.
    file_path = "pict.dat"

    data_list = []
    with open(file_path, "r") as file:
        # Read each line and append it to the list
        for line in file:
            data_list.append(line.strip()) 

    data = []
    for element in data_list[0].split(','):
        data.append(int(element))

    # Reshape the data into 9 NumPy arrays with shape (1024,)
    num_arrays = 9
    array_shape = (1024,)
    numpy_arrays = []

    for i in range(num_arrays):
        start_idx = i * array_shape[0] 
        end_idx = (i + 1) * array_shape[0]
        numpy_arrays.append(np.array(data[start_idx:end_idx]).reshape(array_shape))
        print(numpy_arrays[i].shape)

    return numpy_arrays
