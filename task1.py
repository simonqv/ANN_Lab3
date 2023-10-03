import numpy as np
from read_pict_data import read_pict_data

def calc_weight(input_patterns):
    # no scaling by 1/N
    num_neurons = input_patterns[0].shape[0]    # 1024
    weight_matrix = np.zeros((num_neurons, num_neurons))

    for pattern_mu in input_patterns:
        weight_matrix += np.outer(pattern_mu, pattern_mu)

    '''
    # no scaling by 1/N
    
    W = np.zeros((len(input_patterns[0]), len(input_patterns[0])))
    for mu in range(len(input_patterns)):
        W += np.outer(input_patterns[mu].reshape((len(input_patterns[0]), 1)).T, input_patterns[mu])
    '''
    return weight_matrix


def update_rule(pattern, weight_matrix):
    max_iters = 2
    updated_pattern = pattern.copy()
    for _ in range(max_iters):
        for i, w_row in enumerate(weight_matrix):
            # Compute the weighted sum of inputs
            weighted_sum = np.dot(w_row, updated_pattern)
            if weighted_sum >= 0:
                updated_pattern[i] = 1
            else:
                updated_pattern[i] = -1
    return updated_pattern


def test():
    x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
    x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
    x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])

    X = np.array([x1, x2, x3])
    W = calc_weight(X)
    recall = update_rule(x3, W)

    return 0


def task3_1():
    x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
    x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
    x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])
    X = np.array([x1, x2, x3])

    W = calc_weight(X)

    # distorted patterns to try and retrieve original X which we trained weights on
    x1d = np.array([1, -1, 1, -1, 1, -1, -1, 1])
    x2d = np.array([1, 1, -1, -1, -1, 1, -1, -1])
    x3d = np.array([1, 1, 1, -1, 1, 1, -1, 1])
    X_distorted = np.array([x1d, x2d, x3d])

    # try to recall all original patterns from the distorted ones
    loops_needed = [0, 0, 0]
    for i, pattern in enumerate(X_distorted):
        loop = 0
        recall  = update_rule(pattern, W)
        
        while True:
            recall  = update_rule(recall.T, W.T)
            if np.array_equal(recall, X[i]):
                loops_needed[i] = loop
                break
            if loop > 3: # log(8) < 3 so should be enough to see stable point
                loops_needed[i] = "didn't converge"
                break
            loop += 1

    print(loops_needed)


    # See how many attractors there are in this network
    # to do this test all 2^8 possible patterns

    # create patterns
    X_all = np.empty((256, 8))
    for i in range(256):
        X_all[i] = np.array(list(format(i, '08b')), dtype=int)*2 -1



def task3_2():
    # patterns_array = [p1, p2, ..., p9]
    patterns_array = read_pict_data()
    print(patterns_array)
    p1 = patterns_array[0] # (1024,)
    p2 = patterns_array[1]
    p3 = patterns_array[2]

    # train on p1, p2, p3
    x_patterns = np.array([p1, p2, p3])

    weight_matrix = calc_weight(x_patterns)
    print("W res", weight_matrix.shape)

    # check that the three patterns are stable.
    for i, x in enumerate(x_patterns):
        recall = update_rule(x, weight_matrix)     

        wrong_elements = 0
        print(recall.shape, x.shape)
        for elem in recall == x:
            if elem == False:
                wrong_elements+=1
        print(f"p{i+1} WRONG ELEMENTS", wrong_elements)

            
        
    
    #print(type(x[30][17]), type(recall.reshape((32,32))[30][17]))
    #print(x[30][17], recall.reshape((32,32))[30][17])

def main():
    # test before starting (in section 2.2)
    #test()

    # task 3.1 Convergence and attractors
    task3_1()

    # task 3.2 Sequential update
    #task3_2()

main()