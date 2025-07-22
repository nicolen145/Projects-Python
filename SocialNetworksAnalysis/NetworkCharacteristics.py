#!/usr/bin/env python
# coding: utf-8

# The code was written in Jupyter Notebook


# Imports:
import networkx as nx
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import pickle
import matplotlib.pyplot as plt
import powerlaw
import random



# # Part 1 - Generating and analysing random networks


def random_networks_generator(n, p, num_networks=1, directed=False, seed=209296169):
    """
    Generates random networks using the G(n, p) model.

    Parameters:
        n (int): Number of nodes in each network (>0)
        p (float): Probability of edge creation (0 < p <= 1)
        num_networks (int): Number of networks to generate
        directed (bool): Whether the network is directed
        seed (int): Seed for random generation

    Returns:
        List of generated NetworkX graphs.
    """
    random.seed(seed)
    generated_graphs = []

    for i in range(num_networks):
        g = nx.gnp_random_graph(n, p, seed=seed + i, directed=directed)
        generated_graphs.append(g)

    return generated_graphs


def network_stats(g):
    """
    Calculates basic statistics for a given NetworkX graph.

    Parameters:
        g: (networkx.Graph): A NetworkX graph (can be directed or undirected)

    Returns:
        dict: Dictionary containing the following statistics:
        1. Average degrees' distribution.
        2. Standard deviation degrees distribution.
        3. Degrees distribution minimum value.
        4. Degrees distribution maximum value.
        5. Average shortest path length between pairs of users.
        6. The network’s diameter.
    """
    degrees = []
    for n, d in g.degree():
        degrees.append(d)

    net_stats = {
        'degrees_avg': float(np.mean(degrees)),
        'degrees_std': float(np.std(degrees)),
        'degrees_min': float(np.min(degrees)),
        'degrees_max': float(np.max(degrees)),
        'spl': None,
        'diameter': None
    }

    # Ensure the graph is connected before calculating SPL and diameter
    if nx.is_connected(g.to_undirected()):
        net_stats['spl'] = float(nx.average_shortest_path_length(g))
        net_stats['diameter'] = float(nx.diameter(g))

    return net_stats


def networks_avg_stats(network_list):
    """
    Calculates average statistics over a list of NetworkX graphs.

    Parameters:
        network_list (list): List of NetworkX graphs.

    Returns:
        dict: Dictionary of averaged statistics.
    """
    all_stats = {
        'degrees_avg': [],
        'degrees_std': [],
        'degrees_min': [],
        'degrees_max': [],
        'spl': [],
        'diameter': []
    }

    for g in network_list:
        net_stats = network_stats(g)
        for key in all_stats:
            if net_stats[key] is not None:
                all_stats[key].append(net_stats[key])

    # Compute average for each stat
    avg_stats = {key: float(np.mean(values)) if values else None for key, values in all_stats.items()}
    return avg_stats


# network_labels = ['a', 'b', 'c', 'd']
# networks = [
#     random_networks_generator(100, 0.1, 20, False),
#     random_networks_generator(100, 0.6, 20, False),
#     random_networks_generator(1000, 0.1, 10, False),
#     random_networks_generator(1000, 0.6, 10, False)
# ]

# for label, network_list in zip(network_labels, networks):
#     net_stats = networks_avg_stats(network_list)
#     print(f"\nStats for network type '{label}':")
#     for k, v in net_stats.items():
#         print(f"  {k}: {v:.4f}")

# -------------------------------------------------------------------------------------------
# # Part 2 - Random networks - hypothesis testing

def rand_net_hypothesis_testing(network, theoretical_p, alpha=0.05):
    """
    Performs hypothesis testing on a random network's 'p' parameter.

    Parameters:
        network (networkx.Graph): The random network to test.
        theoretical_p (float): The assumed value of p.
        alpha (float): Significance level.

    Returns:
        tuple: (p-value, 'accept' or 'reject')
    """

    # Get number of nodes and edges
    n = network.number_of_nodes()
    m = network.number_of_edges()

    # Estimate observed p from the network
    max_edges = n * (n - 1) / 2
    observed_p = m / max_edges

    # Compute standard error of p under H0
    std_error = np.sqrt(theoretical_p * (1 - theoretical_p) / max_edges)

    # Compute z-score
    z = (observed_p - theoretical_p) / std_error

    # Compute two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z)))

    # Decision
    decision = 'reject' if p_value < alpha else 'accept'

    return p_value, decision


def most_probable_p(graph):
    possible_ps = [0.01, 0.1, 0.3, 0.6]
    n = graph.number_of_nodes()
    degrees = []
    for node in graph.nodes():
        degrees.append(graph.degree(node))

    avg_deg = np.mean(degrees)

    best_p = -1
    best_pval = 0  # higher is better (we want to accept H0)

    for p in possible_ps:
        expected_mu = (n - 1) * p
        expected_sigma = np.sqrt((n - 1) * p * (1 - p))

        # Perform z-test on the observed average degree
        z = (avg_deg - expected_mu) / expected_sigma
        pval = 2 * (1 - norm.cdf(abs(z)))  # two-tailed

        if pval > best_pval and pval >= 0.05:
            best_pval = pval
            best_p = p

    return best_p


# Load the pickle file of random networks
with open('rand_nets.p', 'rb') as f:
    rand_networks = pickle.load(f)

# # Go through the 10 random networks
# for i, net in enumerate(rand_networks):
#     best_p = most_probable_p(net)
#     print(f"Network {i}: most probable p = {best_p}")

#     # Stop if a valid p was found
#     if best_p in [0.01, 0.1, 0.3, 0.6]:
#         print(f"\nFound a good match: network index {i}, p = {best_p}")
#         print(f"\nNumber of nodes: {net.number_of_nodes()}")
#         chosen_network = net
#         chosen_p = best_p
#         break



# # Run hypothesis test with the original p
# p_val_orig, decision_orig = rand_net_hypothesis_testing(chosen_network, chosen_p)
# print(f"Original p = {chosen_p}")
# print(f"p-value: {p_val_orig:.4f}, decision: {decision_orig}")

# # Run with +10%
# p_up = round(chosen_p * 1.1, 4)
# p_val_up, decision_up = rand_net_hypothesis_testing(chosen_network, p_up)
# print(f"\n+10% p = {p_up}")
# print(f"p-value: {p_val_up:.4f}, decision: {decision_up}")

# # Run with -10%
# p_down = round(chosen_p * 0.9, 4)
# p_val_down, decision_down = rand_net_hypothesis_testing(chosen_network, p_down)
# print(f"\n-10% p = {p_down}")
# print(f"p-value: {p_val_down:.4f}, decision: {decision_down}")



# #Find a small network (fewer than 150 nodes)
# small_net = None

# for net in rand_networks:
#     if net.number_of_nodes() < 150:
#         small_net = net
#         break

# if small_net:
#     print(f"Small network found with {small_net.number_of_nodes()} nodes")

#     best_p = most_probable_p(small_net)
#     print(f"Most probable p: {best_p}")

#     if best_p in [0.01, 0.1, 0.3, 0.6]:
#         # Run hypothesis tests
#         p_val_orig, decision_orig = rand_net_hypothesis_testing(small_net, best_p)
#         p_val_up, decision_up = rand_net_hypothesis_testing(small_net, round(best_p * 1.1, 4))
#         p_val_down, decision_down = rand_net_hypothesis_testing(small_net, round(best_p * 0.9, 4))

#         print(f"\nOriginal p = {best_p}")
#         print(f"p-value: {p_val_orig:.4f}, decision: {decision_orig}")
#         print(f"\n+10% p = {round(best_p * 1.1, 4)}")
#         print(f"p-value: {p_val_up:.4f}, decision: {decision_up}")
#         print(f"\n-10% p = {round(best_p * 0.9, 4)}")
#         print(f"p-value: {p_val_down:.4f}, decision: {decision_down}")
#     else:
#         print("No suitable p was found for this network.")
# else:
#     print("No small network found.")



def plot_qq_for_network(network, p_value_used):
    degrees = list(dict(network.degree()).values())
    n = network.number_of_nodes()

    # Expected degree distribution ~ Binomial(n-1, p) ≈ Normal(μ, σ)
    mu = (n - 1) * p_value_used
    sigma = np.sqrt((n - 1) * p_value_used * (1 - p_value_used))

    # QQ-Plot against normal distribution
    stats.probplot(degrees, dist="norm", sparams=(mu, sigma), plot=plt)
    plt.title(f"QQ-Plot of Degree Distribution\nAgainst N({mu:.1f}, {sigma:.1f}²)")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Observed Degrees")
    plt.grid(True)
    plt.show()


# plot_qq_for_network(chosen_network, chosen_p)

# # -------------------------------------------------------------------------------------------
# # Part 3 - Find an optimal 𝛾 parameter to a scale-free network

with open('scalefree_nets.p', 'rb') as f:
    scalefree_networks = pickle.load(f)

# print(f"Loaded {len(scalefree_networks)} scale-free networks.")


def find_opt_gamma(network, treat_as_social_network=True):
    """
    Estimates the optimal gamma (α in powerlaw) for a given network using the powerlaw package.

    Parameters:
        network: A scale-free network.
        treat_as_social_network (bool): Whether to treat the degree distribution as discrete (default: True).

    Returns:
        float: Estimated gamma (alpha) value.
    """
    degrees = []
    for node in network.nodes():
        degrees.append(network.degree(node))

    # Fit the degree distribution
    fit = powerlaw.Fit(degrees, discrete=treat_as_social_network, verbose=False)

    return fit.alpha



# gammas = []

# for i, net in enumerate(scalefree_networks):
#     gamma = find_opt_gamma(net)
#     gammas.append(gamma)
#     print(f"Network {i}: gamma = {gamma:.4f}")




# stats_sf = network_stats(scalefree_networks[2])
# print(f"Number of nodes: {scalefree_networks[2].number_of_nodes()}")
# print(f"Number of Eds:ge: {scalefree_networks[2].number_of_edges()}")
# for k, v in stats_sf.items():
#     print(f"{k}: {v:.4f}")

# -------------------------------------------------------------------------------------------
# # Part 4 - Distinguish between random networks and scale free networks


with open('mixed_nets.p', 'rb') as f:
    mixed_networks = pickle.load(f)

# print(f"Loaded {len(mixed_networks)} networks.")


def netwrok_classifier(network):
    """
    Classify a network as random (1) or scale-free (2)
    using both most_probable_p and gamma range check.
    """
    # Try most probable p first
    p = most_probable_p(network)
    if p in [0.01, 0.1, 0.3, 0.6]:
        return 1  # random

    # Otherwise check if gamma is in expected scale-free range
    gamma = find_opt_gamma(network)
    if 2.0 <= gamma <= 3.0:
        return 2  # scale-free
    else:
        return 1  # gamma > 3 -> classify as random


#Use the classifier

# for i, net in enumerate(mixed_networks):
#     result = netwrok_classifier(net)
#     label = "Random" if result == 1 else "Scale-Free"
#     print(f"Network {i}: classified as {label}")

