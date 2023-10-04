import random
import numpy as np
import matplotlib.pyplot as plt
from read_pict_data import read_pict_data

"""
Takes in a pattern of shape (1024,), and plots it in a 32x32 colormap.
"""
def plot_pattern(x, recall):
    # Create a figure and axis for visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Adjust the figsize as needed
    
    # Display the original pattern using a binary colormap
    x = x.reshape((32,32))
    axes[0].imshow(x, cmap='binary')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title("Input pattern")

    # Display the recall pattern using a binary colormap
    recall = recall.reshape((32,32))
    axes[1].imshow(recall, cmap='binary')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("Recall pattern")

    plt.show()


def plot_async_update(partial_updates):
    fig, axes = plt.subplots(5, 4, figsize=(10, 5))
    ind=0
    for i in range(5):
        for j in range(4):
            pu = partial_updates[ind]
            axes[i%5, j%4].imshow(pu.reshape((32, 32)), cmap='binary')
            axes[i%5, j%4].set_xticks([])
            axes[i%5, j%4].set_yticks([])
            axes[i%5, j%4].set_title(f"Iteration nr: {200*ind}")
            ind+=1
    plt.show()


def calc_weight(input_patterns):
    # no scaling by 1/N
    num_neurons = input_patterns[0].shape[0]    # 1024
    weight_matrix = np.zeros((num_neurons, num_neurons))
    for pattern_mu in input_patterns:
        weight_matrix += np.outer(pattern_mu, pattern_mu)
    return weight_matrix


def update_rule(pattern, weight_matrix):
    updated_pattern = pattern.copy()
    updated_pattern = np.sign(np.dot(weight_matrix, updated_pattern))
    updated_pattern[updated_pattern == 0] = 1
    return updated_pattern.astype(int)


def update_rule_async(pattern, weight_matrix):
    updated_pattern = pattern.copy()
    random_inds = [random.randint(0, 1023) for _ in range(4000)]
    counter = 0
    partial_updates = []
    for ind in random_inds:
        updated_pattern[ind] = np.sign(np.sum(weight_matrix[ind] * updated_pattern))
        updated_pattern[updated_pattern == 0] = 1
        if counter%200 == 0:
            partial_updates.append(updated_pattern.copy())
        counter += 1
    plot_async_update(partial_updates)
    return updated_pattern


def degrade_patterns(pattern1, pattern2_3):
    rand_ints = np.random.randint(0, 1024, 200)
    for i in rand_ints:
        pattern1[i] = 1
    
    pattern11 = np.append(pattern2_3[0][:450], pattern2_3[1][450:]) # was 512. Dependig on majority, one side "wins" if too similar, a random local minima wins
    return pattern1, pattern11


def test():
    x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
    x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
    x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])

    X = np.array([x1, x2, x3])
    W = calc_weight(X)
    recall = update_rule(x3, W)

    return 0


def task3_1():
    # Original patterns
    x1 = np.array([-1, -1,  1, -1,  1, -1, -1,  1])
    x2 = np.array([-1, -1, -1, -1, -1,  1, -1, -1])
    x3 = np.array([-1,  1,  1, -1, -1,  1, -1,  1])
    X = np.array([x1, x2, x3])

    W = calc_weight(X)

    # distorted patterns to try and retrieve original X which we trained weights on
    x1d = np.array([1, -1,  1, -1,  1, -1, -1,  1])
    x2d = np.array([1,  1, -1, -1, -1,  1, -1, -1])
    x3d = np.array([1,  1,  1, -1,  1,  1, -1,  1])
    X_distorted = np.array([x1d, x2d, x3d])

    # try to recall all original patterns from the distorted ones
    loops_needed = [0, 0, 0]
    for i, pattern in enumerate(X_distorted):
        loop = 0
        recall = update_rule(pattern, W)

        while True:
            recall = update_rule(recall.T, W.T)
            if np.array_equal(recall, X[i]):
                loops_needed[i] = loop
                break
            if loop > 3:  # log(8) < 3 so should be enough to see stable point
                loops_needed[i] = "didn't converge to stored"
                break
            loop += 1

    print("Does the distorted patterns converge to stored ones? In how long?\n", loops_needed)

    # See how many attractors there are in this network
    # to do this test all 2^8 possible patterns

    # create patterns
    X_all = np.empty((256, 8))
    for i in range(256):
        X_all[i] = np.array(list(format(i, '08b')), dtype=int) * 2 - 1

    # find attractors
    stable_states = np.empty((256, 8))

    for i, row in enumerate(X_all):
        loop = 0
        recall = update_rule(row, W)
        while loop < 3:
            recall = update_rule(recall, W)
            loop += 1
        stable_state = recall
        stable_states[i] = stable_state
    stable_states = np.unique(stable_states, axis=0)

    print("\nNumber of stable states/attractors", len(stable_states))

    # Make the starting pattern more dissimilar,
    # more than half wrong
    x1_big_distortion = np.array([-1, -1, 1, 1, 1, 1, -1, -1])
    recall = update_rule(x1_big_distortion, W)
    for i in range(3):
        recall = update_rule(recall, W)
    print("\nThe distorted pattern\t\t", x1_big_distortion)
    print("Recall after big distortion\t", recall[:])
    print("True pattern x1 we tried to find", x1)

    """
    for state in stable_states:
        print("", x2,"\n", state)
        print(np.array_equal(state, x2))
    """


def task3_2():
    # patterns_array = [p1, p2, ..., p9]
    patterns_array = read_pict_data()

    p1 = patterns_array[0] # (1024,)
    p2 = patterns_array[1]
    p3 = patterns_array[2]
    p10 = patterns_array[0].copy() # to be degraded p1
    p11 = np.array([patterns_array[1].copy(), patterns_array[2].copy()]) # to be mixture of p2 and p3

    # train on p1, p2, p3
    x_patterns = np.array([p1, p2, p3])

    weight_matrix = calc_weight(x_patterns)

    # point 1: check that the three patterns are stable
    max_iters = int(np.log(1024))
    for i, x in enumerate(x_patterns):
        if i == 0:
            x = patterns_array[6]
        recall = x.copy()
        for _ in range(max_iters):
            recall = update_rule(recall, weight_matrix)   
        
        # plot and count differences in the final recall pattern
        # plot_pattern(x, recall)
        wrong_elements = 0
        for elem in recall == x:
            if elem == False:
                wrong_elements+=1
        print(f"p{i+1} WRONG ELEMENTS", wrong_elements)


    # point 2: test if we can complete a degraded pattern (p10 for p1)
    # allow 7 tries to find attractor since log(1024) ~ 6,9
    p10, p11 = degrade_patterns(p10, p11)
    recall = update_rule(p10, weight_matrix)
    for i in range(6):
        recall = update_rule(recall, weight_matrix)
    if np.array_equal(recall, p1):
        print("Found p1 from p10\n")
    else:
        print("Could NOT find p1 from p10\n")
    # plot_pattern(p10, recall)

    recall = update_rule(p11, weight_matrix)
    # Dependig on majority, one side "wins" if too similar, a random local minima wins
    for i in range(6):
        recall = update_rule(recall, weight_matrix)
    if np.array_equal(recall, p2):
        print("Could find p2 from p11")
    elif np.array_equal(recall, p3):
        print("Could find p3 from p11")
    else:
        print("Found neither p2 nor p3 from p11")
    # plot_pattern(p11, recall)

    
    # point 3: Randomly select units. 

    #for i, x in enumerate(x_patterns):
    recall = p11.copy() # x.copy()    
# for i, x in enumerate(x_patterns):
    # for _ in range(max_iters):
    recall = update_rule_async(recall, weight_matrix)   
        
    # plot and count differences in the final recall pattern
    # plot_pattern(x, recall)
    wrong_elements = 0
    for elem in recall == x:
        if elem == False:
            wrong_elements+=1
    print(f"p{i+1} WRONG ELEMENTS", wrong_elements)


    

def task3_3():
    pass

def task3_4():
    pass

def task3_5():
    pass

def task3_6():
    pass


def main(task):

    # test before starting (in section 2.2)
    # test()
    if task == 1:
        # task 3.1 Convergence and attractors
        task3_1()
    elif task == 2:
        # task 3.2 Sequential update
        task3_2()

    # TODO: everything below :p
    elif task == 3:
        task3_3()
    elif task == 4:
        task3_4()
    elif task == 5:
        task3_5()
    elif task == 6:
        task3_6()


main(2)
