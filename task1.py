import copy
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
    for i, pu in enumerate(partial_updates):
        axes[i%5, i%4].imshow(pu.reshape((32, 32)), cmap='binary')
        axes[i%5, i%4].set_xticks([])
        axes[i%5, i%4].set_yticks([])
        axes[i%5, i%4].set_title(f"Iteration nr: {i}")
    #plt.show()


def plot_energy(energy_list):
    x = np.arange(0, 4000, 200)
    plt.plot(x, energy_list)
    #plt.show()


def calc_weight(input_patterns, normalize=False):
    # no scaling by 1/N
    num_neurons = input_patterns[0].shape[0]    # 1024
    weight_matrix = np.zeros((num_neurons, num_neurons))

    for pattern_mu in input_patterns:
        weight_matrix += np.outer(pattern_mu.T, pattern_mu)

    if normalize == True:
        weight_matrix = weight_matrix/num_neurons
    
    return weight_matrix


def update_rule(pattern, weight_matrix):
    updated_pattern = pattern.copy()
    updated_pattern = np.sign(np.dot(weight_matrix, updated_pattern))
    updated_pattern[updated_pattern == 0] = 1
    return updated_pattern.astype(int)


def update_rule_async(pattern, weight_matrix):

    updated_pattern = pattern.copy()
    random_inds = np.random.randint(0, 1023, 4000)

    counter = 0
    partial_updates = []
    partial_energy = [0]
    no_change = 0
    for ind in random_inds:
        updated_pattern[ind] = np.sign(np.sum(weight_matrix[ind] * updated_pattern))
        updated_pattern[updated_pattern == 0] = 1
        if counter%200 == 0:
            partial_updates.append(updated_pattern.copy())
            energy = calc_energy(weight_matrix, updated_pattern)
            partial_energy.append(energy)
        counter += 1

    #plot_async_update(partial_updates)
    
    return updated_pattern, partial_energy[1:]


def degrade_patterns(pattern1, pattern2_3):
    rand_ints = np.random.randint(0, 1024, 200)
    for i in rand_ints:
        pattern1[i] = 1
    
    pattern11 = np.append(pattern2_3[0][:450], pattern2_3[1][450:]) # was 512. Dependig on majority, one side "wins" if too similar, a random local minima wins
    return pattern1, pattern11


def calc_energy(weights, pattern):
    energy = 0
    for x_i in range(len(pattern)):
        sum_i = 0
        for x_j in range(len(pattern)):
            sum_i += weights[x_i, x_j]* pattern[x_i] * pattern[x_j]
        energy += sum_i
    energy = -energy
    return energy


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
    recall, _ = update_rule_async(recall, weight_matrix)   
        
    # plot and count differences in the final recall pattern
    # plot_pattern(x, recall)
    wrong_elements = 0
    for elem in recall == x:
        if elem == False:
            wrong_elements+=1
    print(f"p{i+1} WRONG ELEMENTS", wrong_elements)


def task3_3():
    patterns_array = read_pict_data()
    p1 = patterns_array[0] # (1024,)
    p2 = patterns_array[1]
    p3 = patterns_array[2]
    p10 = patterns_array[0].copy() # to be degraded p1
    p11 = np.array([patterns_array[1].copy(), patterns_array[2].copy()]) # to be mixture of p2 and p3
    p10, p11 = degrade_patterns(p10, p11)
    
    # train on p1, p2, p3
    x_patterns = np.array([p1, p2, p3])
    weight_mat = calc_weight(x_patterns, normalize=True)
    
    # point 1 and 2 - what is the energy at different attractors
    energy_p1 = calc_energy(weight_mat, p1)
    print("\nEnergy at p1 \t", energy_p1)
    energy_p2 = calc_energy(weight_mat, p2)
    print("Energy at p2 \t", energy_p2)
    energy_p3 = calc_energy(weight_mat, p3)
    print("Energy at p3 \t", energy_p3)
    print()
    energy_p10 = calc_energy(weight_mat, p10)
    print("Energy at p10 \t", energy_p10)
    energy_p11 = calc_energy(weight_mat, p11)
    print("Energy at p11 \t", energy_p11)
    plt.show()
    # point 3 - see how the energy changes:
    _, energy_list = update_rule_async(p10, weight_mat)
    plt.figure(31)
    plt.title("Energy change over iterations 0-4000")#we update energy every 200 itr
    plot_energy(energy_list)
    plt.xlabel("Iteration in Sequential Learning")
    plt.ylabel("Normalized Energy")


    # point 5 - symmetric random 
    random_p = [np.random.normal(0, 1, 1024)]
    random_w_mat_sym = calc_weight(random_p, True)
    new_p_sym, energy_sym = update_rule_async(p1, random_w_mat_sym)
    plt.figure(33)
    plt.title("Energy Change with Random Symmetric Weights")
    plot_energy(energy_sym)
    plt.xlabel("Iteration in Sequential Learning")
    plt.ylabel("Normalized Energy")
 


    # point 4 - random weight matrix
    # iterate random starting state (we try p1)
    #shuffle the symmetric weight matrix
    random_w_mat = copy.deepcopy(random_w_mat_sym)
    np.random.shuffle(random_w_mat)

    new_p, energy_rand = update_rule_async(p1, random_w_mat)
    plt.figure(32)
    plot_energy(energy_rand)
    plt.title("Energy Change with Random Weights")
    plt.xlabel("Iteration in Sequential Learning")
    plt.ylabel("Normalized Energy")

    plt.show()


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

    elif task == 3:
        task3_3()

    # TODO: everything below :p
    elif task == 4:
        task3_4()
    elif task == 5:
        task3_5()
    elif task == 6:
        task3_6()


main(3)
