import numpy as np

def calc_weight(input_patterns):
    # no scaling by 1/N
    W = np.zeros((len(input_patterns[0]), len(input_patterns[0])))

    for mu in range(len(input_patterns)):
        W += np.outer(input_patterns[mu].T, input_patterns[mu])
        
    return W


def update_rule(pattern, W):
    recall = np.empty(pattern.shape)
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
            recall  = update_rule(recall, W)
            if np.array_equal(recall, X[i]):
                loops_needed[i] = loop
                break
            if loop > 3: # log(8) < 3 so should be enough to see stable point
                loops_needed[i] = "didn't converge to stored"
                break
            loop += 1

    print("Does the distorted patterns converge to stored ones? In how long?\n", loops_needed)


    # See how many attractors there are in this network
    # to do this test all 2^8 possible patterns

    # create patterns
    X_all = np.empty((256, 8))
    for i in range(256):
        X_all[i] = np.array(list(format(i, '08b')), dtype=int)*2 -1

    # find attractors
    stable_states = np.empty((256,8))
    
    for i, row in enumerate(X_all):
        loop = 0
        recall = update_rule(row, W)
        while(loop < 3):
            recall = update_rule(recall,W)
            loop += 1
        stable_state = recall
        stable_states[i] = stable_state
    stable_states = np.unique(stable_states, axis=0)
      
    print("\nNumber of stable states/attractors", len(stable_states))


    # Make the starting pattern more dissimilar, 
    # more than half wrong
    x1_big_distortion = np.array([-1, -1, 1, 1, 1, 1, -1, -1])
    recall = update_rule(x1_big_distortion, W)
    for i in range(5):
        recall = update_rule(recall, W)
    print("\nThe distorted pattern\t\t", x1_big_distortion)
    print("Recall after big distortion\t", recall[:])
    print("True pattern x1 we tried to find", x1)










def main():
    # test before starting (in section 2.2)
    #test()

    # task 3.1 Convergence and attractors
    task3_1()

main()