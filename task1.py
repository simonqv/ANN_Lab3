import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from read_pict_data import read_pict_data
import copy

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

def plot_energy(energy_list):
    x = np.arange(0, 4000, 200)
    plt.plot(x, energy_list)
    #plt.show()

def calc_weight_sparse_patterns(input_patterns, average_activity):
    # no scaling by 1/N
    num_neurons = input_patterns[0].shape[0]    # 1024
    weight_matrix = np.zeros((num_neurons, num_neurons))
    for pattern_mu in input_patterns:
        weight_matrix += np.outer(pattern_mu.T - average_activity, pattern_mu - average_activity)
    return weight_matrix

def calc_weight(input_patterns, normalize=False):
    # no scaling by 1/N
    num_neurons = input_patterns[0].shape[0]    # 1024
    weight_matrix = np.zeros((num_neurons, num_neurons))
    for pattern_mu in input_patterns:

        weight_matrix += np.outer(pattern_mu.T, pattern_mu)

    if normalize == True:
        weight_matrix = weight_matrix/num_neurons
    

    return weight_matrix

def update_rule_sparse_patterns(pattern, weight_matrix, bias):
    updated_pattern = pattern.copy()
    sign = np.sign(np.dot(weight_matrix, updated_pattern) - bias)
    updated_pattern = 0.5 + 0.5 * sign
    return updated_pattern.astype(float)

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


def flip_bits_in_pattern(number_of_flips, pattern):
    new_pattern = pattern.copy()
    rand_ints = np.random.randint(0, len(pattern), number_of_flips)
    for i in rand_ints:
        new_pattern[i] = new_pattern[i]*(-1)
    return new_pattern

def flip_bits_in_pattern_sparse_patterns(number_of_flips, pattern):
    new_pattern = pattern.copy()
    rand_ints = np.random.randint(0, len(pattern), number_of_flips)
    for i in rand_ints:
        new_pattern[i] = new_pattern[i] + 1
    return new_pattern

def random_array(n=1024):
    # Generate a random array of shape (n,) with values -1 or 1
    return np.random.choice([-1, 1], size=n)

def random_array_w_bias(n=100):
    return np.sign(np.random.choice([-1, 1], size=n)+0.5)

def calc_activity_average(patterns):
    network_size = len(patterns[0])
    number_of_patterns = len(patterns)
    activity_average = 0
    for pattern in patterns:
        activity_average += np.sum(pattern)
    activity_average = activity_average / number_of_patterns / network_size
    return activity_average


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
    # distortion resistance

    # patterns_array = [p1, p2, ..., p9]
    patterns_array = read_pict_data()

    p1 = patterns_array[0] # (1024,)
    p2 = patterns_array[1]
    p3 = patterns_array[2]
    
    # train on p1, p2, p3
    x_patterns = np.array([p1, p2, p3])
    weight_matrix = calc_weight(x_patterns)
    number_of_units = len(p1)

    max_iters = int(np.log(1024))

    attractors = []
    for pattern in x_patterns:
        
        for flip_fraction in range(0,110,10):
            number_of_flips = int(number_of_units*flip_fraction/100)
            recall = flip_bits_in_pattern(number_of_flips, pattern)
            for iter in range(max_iters):
                recall = update_rule(recall, weight_matrix)
            attractors.append(recall.copy())
            '''
            plt.figure()
            plt.imshow(recall.reshape((32,32)), cmap='binary')
            plt.title(f"Flipped {number_of_flips} ({flip_fraction}%)")
            print("LEN OF ATTRACTOR LIST", len(attractors))
            '''
        '''
        plt.figure()
        plt.imshow(pattern.reshape((32,32)), cmap='binary')
        plt.title(f"Real pattern")
        plt.show()
        '''

    unique_attractors = np.unique(np.array(attractors), axis=0)
    print("LLELL", len(unique_attractors))
    for fig_nr, attractor in enumerate(unique_attractors): 
        print(len(attractor))
        plt.figure()
        plt.imshow(attractor.reshape((32,32)), cmap='binary')
        plt.title(f"Attractor")
    plt.show()


def task3_5():
    # Add more and more memories to the network to see where the limit is.
    # Start by adding p4 into the weight matrix and check if moderately distorted patters can still be recognized.
    # Then continue by adding others such as p5, p6 and p7 in some order and checking the performance after each addition
    random_flag = True
    if random_flag == False:
        patterns = read_pict_data()

        p10 = copy.deepcopy(patterns[0]) # to be degraded p1
        p11 = np.array([copy.deepcopy(patterns[1]), copy.deepcopy(patterns[2])])  # to be mixture of p2 and p3
        p10, p11 = degrade_patterns(p10, p11)

    elif random_flag == True:
        patterns = []
        for i in range(9):
            patterns.append(random_array())

    for end in range(3, 10):
        print("\n----- new run -----\n")
        x_patterns = np.array([copy.deepcopy(patterns[i]) for i in range(end)])
        # x_patterns = [patterns[2], patterns[1], patterns[0], patterns[7]]
        weight_matrix = calc_weight(x_patterns)

        # point 1: check that the three patterns are stable
        max_iters = int(np.log(1024))
        for i, x in enumerate(x_patterns):
            recall = copy.deepcopy(x)
            for _ in range(max_iters):
                recall = update_rule(recall, weight_matrix)

            wrong_elements = 0
            condition = recall == x
            for elem in condition:
                if not elem:
                    wrong_elements += 1
            print(f"p{i + 1} WRONG ELEMENTS", wrong_elements, "out of ", len(condition), "ratio: ", wrong_elements/len(condition))

    """
        recall = update_rule(p10, weight_matrix)
        for i in range(6):
            recall = update_rule(recall, weight_matrix)
        if np.array_equal(recall, patterns[0]):
            print("Found p1 from p10\n")
        else:
            print("Could NOT find p1 from p10\n")
        plot_pattern(p10, recall)

        recall = update_rule(p11, weight_matrix)
        # Dependig on majority, one side "wins" if too similar, a random local minima wins
        for i in range(6):
            recall = update_rule(recall, weight_matrix)
        if np.array_equal(recall, patterns[1]):
            print("Could find p2 from p11")
        elif np.array_equal(recall, patterns[2]):
            print("Could find p3 from p11")
        else:
            print("Found neither p2 nor p3 from p11")
        # plot_pattern(p11, recall)
    """

def task3_5_random(noise_flag=True, remove_w_diagonal=False, bias=False):
    y_axis = []
    x_axis= []

    # generate 300 patterns of shape (200,)
    number_of_patterns = 300
    network_size = 100
    patterns = []
    for i in range(number_of_patterns):    
        if bias == True:
            patterns.append(random_array_w_bias())
        else:
            patterns.append(random_array(network_size))

    # calculate weights, recall and plot
    for end in range(1, number_of_patterns+1):
        print("\n----- new run -----\n")
        x_patterns = np.array([copy.deepcopy(patterns[i]) for i in range(end)])
        # x_patterns = [patterns[2], patterns[1], patterns[0], patterns[7]]
        weight_matrix = calc_weight(x_patterns)
        if remove_w_diagonal == True:
            # Find the indices of the diagonal elements
            diag_indices = np.diag_indices_from(weight_matrix)
            # Set the values of the diagonal elements to zero
            np.put(weight_matrix, diag_indices, 0)
        count = 0
        # point 1: check that the three patterns are stable
        max_iters = int(np.log(network_size))
        for i, x in enumerate(x_patterns):
            recall = copy.deepcopy(x)
            if noise_flag == True:
                recall = flip_bits_in_pattern(10, copy.deepcopy(x))
            for _ in range(max_iters):
                recall = update_rule(recall, weight_matrix)
            if np.array_equal(recall, x):
                count +=1
        y_axis.append(count/len(x_patterns))
    x_axis = np.array(range(1, number_of_patterns+1))
    y_axis = np.array(y_axis)
    plt.figure()
    plt.plot(x_axis, y_axis)
    plt.title("Memorization Hopfield network 100 nodes \n Random patterns")
    plt.xlabel("Number of patterns")
    plt.ylabel("Number of memorized patterns")
    #plt.ylim([0,1.1])

    if noise_flag == True:
        plt.title("Memorization Hopfield network 100 nodes \n Random patterns \n 10% noise")
    if remove_w_diagonal == True:
        plt.title("Memorization Hopfield network 100 nodes \n Random patterns \n 10% noise \n no self-connections")
    if bias == True:
        plt.title("Memorization Hopfield network 100 nodes \n Random patterns \n added bias")
    #plt.show()


def task3_6():
    sparsity_list = np.array([0.1, 0.05, 0.01])
    for sparsity in sparsity_list:
        bias_list = np.array([0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 1.1, 1.5])#np.arange(0.1, 0.4, 0.1) #np.array([0.3])
        network_size = 100#100
        number_of_patterns = 300
        patterns = []
        number_of_flips = int(sparsity*network_size)
        for i in range(number_of_patterns):
            all_zero_pattern = np.zeros((network_size, ))
            pattern = flip_bits_in_pattern_sparse_patterns(number_of_flips, all_zero_pattern)
            #print("NR 1:s", len(pattern[pattern == 1]))
            patterns.append(pattern)

        storage_capacity = []
        for bias in bias_list:
            #for end in range(3, number_of_patterns+1):
            storage_capacity_w_bias = []
            for end in range(1, number_of_patterns+1):
                max_iters = int(np.log(network_size))
                x_patterns = [copy.deepcopy(patterns[i]) for i in range(end)]
                average_activity = calc_activity_average(x_patterns)
                count = 0
                weight_matrix = calc_weight_sparse_patterns(x_patterns, average_activity)
                for pattern in x_patterns:
                    #print("NEW")
                    #print("NR 1:s", len(pattern[pattern == 1]))
                    recall = copy.deepcopy(pattern)
                    for _ in range(max_iters):
                        recall = update_rule_sparse_patterns(recall, weight_matrix, bias)
                        #print("NR 1:s", len(recall[recall == 1]))
                    diff = recall == pattern
                    if np.array_equal(recall, pattern):
                        count += 1
                #print("COUNT", count/len(x_patterns))
                storage_capacity_w_bias.append(count/len(x_patterns))
            storage_capacity.append(storage_capacity_w_bias)
        

        x_axis = np.array(range(1, number_of_patterns+1))
    
        plt.figure()
        for bias, y in enumerate(storage_capacity):
            print(y)
            plt.plot(np.array(x_axis), np.array(y), label = f"bias {bias_list[bias]}")
        plt.title(f"Memorization Hopfield network 100 nodes \n bias \n sparsity = {sparsity}")
        plt.xlabel("Number of patterns")
        plt.ylabel("Number of memorized patterns")
        plt.legend()
    plt.show()
    


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
        #task3_5()
        task3_5_random(noise_flag=False)
        task3_5_random()
        task3_5_random(remove_w_diagonal=True)
        task3_5_random(noise_flag=False, bias=True)
        plt.show()
    elif task == 6:
        task3_6()



main(6)

