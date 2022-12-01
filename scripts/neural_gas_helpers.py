import numpy as np
from neupy import algorithms
from scipy.spatial import Voronoi
import time 
import random 
from collections import deque

def L2dist(n1, n2):
    return (np.sum((n1-n2)**2))**(0.5)

def data2memories(states, controls, max_memories, gng_epochs, step=0.2, n_start_nodes=2, max_edge_age=50):
    # Concatenate states and controls
    inputs = np.concatenate((states, controls), axis=1)

    # GrowingNeuralGas
    gng = algorithms.GrowingNeuralGas(
            n_inputs=inputs.shape[1],
            n_start_nodes=max_memories,
            shuffle_data=True,
            verbose=True,
            # step=step,
            # neighbour_step=0.005,
            # max_edge_age=max_edge_age,
            max_nodes=max_memories,
            # n_iter_before_neuron_added=1,
            # after_split_error_decay_rate=0.5,
            # error_decay_rate=0.995,
            # min_distance_for_update=0.01,
        )
    gng.train(inputs, epochs=gng_epochs)    

    # # obtain nodes and edges
    # nodes = np.array([n.weight.flatten() for n in gng.graph.nodes]) # (max is <max_nodes> but can be small also such as 210, 6) # 6 because train_input is (21000, 6)
    # edges = list(gng.graph.edges.keys())
    # print(f'num of nodes = {len(nodes)} and num of edges = {len(edges)}')
    # return nodes

    # # find closest samples to each node in train_input
    # memories = []
    # for n in nodes:
    #     samples_sorted_by_dist = sorted([(s, i, L2dist(n, s)) for i, s in enumerate(inputs)], key=lambda triple:triple[2])
    #     _, idx, _ = samples_sorted_by_dist[0]
    #     memories.append( inputs[idx] ) # (states[idx], controls[idx], next_states[idx])

    return gng 

def create_voronoi(gng):
    """
    midpoint of line between n1 and n2: (n1+n2)/2
    """
    midpoints = {}
    """
    Bisecting hyperplane: v^T x = b where 
    normal:               v = (n1-n2)/||n1-n2||
    offset:               b = (||n1||^2 - ||n2||^2)/||n1-n2|| (https://math.stackexchange.com/a/1834895)
    """
    normals = {}
    offsets = {}
    for n1, connected_nodes in gng.graph.edges_per_node.items():
        # print(f'{n1.weight=}') # (1, 24) for Drones
        midpoints[n1] = []
        normals[n1] = [] 
        offsets[n1] = []
        for n2 in connected_nodes:
            # print(f'{n2.weight=}')
            m = (n1.weight.flatten() + n2.weight.flatten())/2.0
            midpoints[n1].append(m)
            v = (n1.weight.flatten() - n2.weight.flatten())/L2dist(n1.weight.flatten() - n2.weight.flatten(), 0)
            b = (L2dist(n1.weight.flatten(), 0)**2 - L2dist(n2.weight.flatten(), 0)**2)/(2 * L2dist(n1.weight.flatten() - n2.weight.flatten(), 0))
            normals[n1].append(v)
            offsets[n1].append(b)

    return (midpoints, normals, offsets)

def m_step_neighbours(node, gng, m=4):
    # BFS with exit at order=m
    visited = {}
    queue = deque()
    queue.append((node, 0))
    while queue:
        n, curr_order = queue.popleft()
        visited[n] = True
        if curr_order == m:
            break

        for neighbour in gng.graph.edges_per_node[n]:
            if neighbour not in visited:
                visited[neighbour] = True
                queue.append((neighbour, curr_order+1))

    return visited.keys()

def voronoi_2_voronoi_bounds(gng, midpoints, states, controls, traj_starts, family_of_dynamics, num_control_inputs, num_samples=25):
    voronoi_bounds = {'lo': {}, 'up': {}}
    least_midpoint_distance = {}
    greatest_midpoint_distance = {}
    t0 = time.time()
    num_nodes = len(gng.graph.nodes)

    points_in_voronoi_cell = {}
    for n1 in midpoints.keys():
        points_in_voronoi_cell[n1] = []

    for pt_idx, (x, u, is_traj_starting) in enumerate(zip(states, controls, traj_starts)):
        input = np.concatenate((x, u))

        if pt_idx == 0 or np.sum(is_traj_starting): # sum because each is_traj_starting is either [0] or [1]
            # print(f'{pt_idx=}, {is_traj_starting=}')
            nodes_to_search = midpoints.keys()
        else:
            nodes_to_search = m_step_neighbours(closest_node, gng, m=10 * int(num_nodes/1000))

        nodes_sorted_by_dist = sorted([(node, L2dist(input, node.weight.flatten())) for node in nodes_to_search], key= lambda tup:tup[1])    
        closest_node = nodes_sorted_by_dist[0][0]

        points_in_voronoi_cell[closest_node].append(input)

    t1 = time.time()
    print(f'Found points in each voronoi cell  ({t1-t0} sec)')

    for n1 in midpoints.keys():
        all_sampled_points = n1.weight # initialize
        
        if len(points_in_voronoi_cell[n1]) > 0:
            all_sampled_points = np.concatenate((all_sampled_points, np.array(points_in_voronoi_cell[n1])), axis=0)

        if len(midpoints[n1]) > 0: # if there is atleast 1 midpoints, append to all_sampled_poonts
            all_sampled_points = np.concatenate((all_sampled_points, np.array(midpoints[n1])), axis=0)

            sorted_dist_to_midpoints_from_n1 = sorted([(midpt, L2dist(midpt, n1.weight.flatten())) for midpt in midpoints[n1]], key= lambda tup:tup[1])
            least_midpoint_distance[n1] = sorted_dist_to_midpoints_from_n1[0][1]
            greatest_midpoint_distance[n1] = sorted_dist_to_midpoints_from_n1[-1][1]
            # print(f'{least_midpoint_distance[n1]=}, {greatest_midpoint_distance[n1]=}')
            
        if len(midpoints[n1]) > 1: # if there is atleast 2 midpoints, sample num_samples points on pareto frontier with center at n1 that is between them
            more_points = []
            num_more_points = num_samples - len(all_sampled_points)
            if num_more_points > 0:
                for _ in range(num_more_points):
                    two_midpoints = random.sample(midpoints[n1], 2)
                    lamda1 = random.random()
                    lamda2 = random.random()
                    point_between_two_midpoints = n1.weight.flatten() + lamda1 * (two_midpoints[0] - n1.weight.flatten()) + (lamda2) * (two_midpoints[1] - n1.weight.flatten())
                    more_points.append(point_between_two_midpoints)
                all_sampled_points = np.concatenate((all_sampled_points, np.array(more_points)), axis=0)

        voronoi_bounds['up'][n1] = -np.inf
        voronoi_bounds['lo'][n1] = np.inf
        for dyn_idx, dyn in enumerate(family_of_dynamics):
            for pt_idx, pt in enumerate(all_sampled_points):
                state = pt[:-num_control_inputs]
                control = pt[-num_control_inputs:]
                next_state_for_pt = dyn([state, control])
                voronoi_bounds['lo'][n1] = np.minimum(voronoi_bounds['lo'][n1], next_state_for_pt) # element-wise maximum
                voronoi_bounds['up'][n1] = np.maximum(voronoi_bounds['up'][n1], next_state_for_pt) # element-wise maximum

        # print(f"just {voronoi_bounds['up'][n1]=}, {voronoi_bounds['lo'][n1]=}")
        # print(f"subtracting {(voronoi_bounds['up'][n1]-voronoi_bounds['lo'][n1])=}")
        # print(f"{L2dist(voronoi_bounds['lo'][n1], voronoi_bounds['up'][n1])=}")

    t2 = time.time()
    print(f'Obtained bounds for all voronoi cells ({t2-t1} sec)')

    return voronoi_bounds, least_midpoint_distance, greatest_midpoint_distance


def voronoi_bounds_2_bounds(gng, midpoints, voronoi_bounds, states, controls, next_states, traj_starts, delta=0.05, DEBUG=True):
    t0 = time.time()
    num_nodes = len(gng.graph.nodes)

    # Max and Min for INFO
    # print(f'Max Min INFO {np.amax(next_states, axis=0)=}, {np.amin(next_states, axis=0)=}, {(np.amax(next_states, axis=0)-np.amin(next_states, axis=0))=}')
        
    ## WIDEST BOUNDS POSSIBLE !!!!!!!!!!
    next_states_lower_bounds = np.amin(next_states, axis=0)
    next_states_upper_bounds = np.amax(next_states, axis=0)
    delta_array = delta * np.abs(next_states_upper_bounds - next_states_lower_bounds)

    for n1 in midpoints.keys():
        voronoi_bounds['lo'][n1] -= delta_array
        voronoi_bounds['up'][n1] += delta_array

    upper_bounds = []
    lower_bounds = []
    if DEBUG:
        all_diffs_lower = []
        all_diffs_upper = []
    for pt_idx, (x, u, next_x, is_traj_starting) in enumerate(zip(states, controls, next_states, traj_starts)):
        input = np.concatenate((x, u))
        
        if pt_idx == 0 or np.sum(is_traj_starting): # sum because each is_traj_starting is either [0] or [1]
            nodes_to_search = midpoints.keys()
        else:
            nodes_to_search = m_step_neighbours(closest_node, gng, m=10 * int(num_nodes/1000))
        
        nodes_sorted_by_dist = sorted([(node, L2dist(input, node.weight.flatten())) for node in nodes_to_search], key= lambda tup:tup[1])
        closest_node = nodes_sorted_by_dist[0][0]

        if DEBUG:
            checks = [voronoi_bounds['lo'][closest_node][i] <= next_x[i] <= voronoi_bounds['up'][closest_node][i] for i in range(len(next_x))]
            if not sum(checks) == len(next_x):
                print(pt_idx, checks, '\n comparison of next_states: \n lo: ', voronoi_bounds['lo'][closest_node], '\n GT: ', next_x, '\n up: ', voronoi_bounds['up'][closest_node], '\n note this is after +-delta_array of ', delta_array, '\n\n')
            diffs_lower = [abs(voronoi_bounds['lo'][closest_node][i] - next_x[i]) for i in range(len(next_x))]
            diffs_upper = [abs(voronoi_bounds['up'][closest_node][i] - next_x[i]) for i in range(len(next_x))]
            all_diffs_lower.append(diffs_lower)
            all_diffs_upper.append(diffs_upper)

        lower_bounds.append( voronoi_bounds['lo'][closest_node] )
        upper_bounds.append( voronoi_bounds['up'][closest_node] )

    if DEBUG:
        print(f'{np.amax(np.array(all_diffs_lower), axis=0)=}')
        print(f'{np.amax(np.array(all_diffs_upper), axis=0)=}')

    print(f'Obtained bounds for all datapoints ({time.time()-t0} sec)')

    return {'lo': np.array(lower_bounds), 'up': np.array(upper_bounds)}



        
        

