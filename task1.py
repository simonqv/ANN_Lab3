import numpy as np
from read_pict_data import read_pict_data

def calc_weight(input_patterns, pattern_is_matrix = True):
    # no scaling by 1/N
    if pattern_is_matrix == True:
        W = np.zeros((len(input_patterns[0])**2, len(input_patterns[0])**2))
        print("W shape", W.shape)
    elif pattern_is_matrix == False:
        W = np.zeros((len(input_patterns[0]), len(input_patterns[0])))

    for mu in range(len(input_patterns)):
        W += np.outer(input_patterns[mu].T, input_patterns[mu])
        
    return W


def update_rule(pattern, W):
    print("PATTERN.shape", pattern.shape)
    recall = np.empty(pattern.shape, dtype=int)
    for i, w_row in enumerate(W):
        sum_wx = np.dot(w_row, pattern.T)
        if sum_wx >= 0:
            recall[i] = 1
        else:
            recall[i] = -1
       
    return recall


def test():
    x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
    x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
    x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])

    X = np.array([x1, x2, x3])
    W = calc_weight(X, pattern_is_matrix=False)
    recall = update_rule(x3, W)

    return 0


def task3_1():
    x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
    x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
    x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])
    X = np.array([x1, x2, x3])

    W = calc_weight(X, pattern_is_matrix=False)

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
    #print(patterns_array)
    p1 = patterns_array[0]
    p2 = patterns_array[1]
    p3 = patterns_array[2]

    print("p1 shape", p1.shape)
    # train on p1, p2, p3
    X = np.array([p1, p2, p3])
    W = calc_weight(X)
    print("W res", W.shape)
    
    # check that the three patterns are stable.
    number_of_updates = 5
    for i, x in enumerate(X):
        if i == 0:
            j = 0
            recall = x
            while j < number_of_updates:
                #print("x.shape", recall.flatten().shape, "W.shape", W.shape)
                recall = update_rule(recall.flatten(), W)
                j+=1
            #print(recall.flatten())
            #print(x[i])
            
            wrong_elements = 0
            for elem in recall.reshape((32,32)) == x[i]:
                for e in elem:
                    if e == False:
                        #print(e)
                        wrong_elements+=1
            print("WRONG ELEMENTS", wrong_elements)

            
        
    
    #print(type(x[30][17]), type(recall.reshape((32,32))[30][17]))
    #print(x[30][17], recall.reshape((32,32))[30][17])

def main():
    # test before starting (in section 2.2)
    #test()

    # task 3.1 Convergence and attractors
    #task3_1()

    # task 3.2 Sequential update
    task3_2()

main()