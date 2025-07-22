import networkx as nx

def centrality_measures(network, node, iterations=100):
    """
    Calculates key centrality measures for a given node in a network.

    Parameters:
    - network: networkx.Graph — the graph to analyze.
    - node: int — the node for which to compute the centralities.
    - iterations: int (default=100) — number of iterations for PageRank and HITS.

    Returns:
    A dictionary with:
    - 'deg_cen': Degree centrality
    - 'clos_cen': Closeness centrality
    - 'nbtw_cen': Normalized betweenness centrality
    - 'npr': Normalized PageRank score (damping factor 0.92)
    - 'nauth': Normalized authority score (from HITS)
    """
    degree_centrality = nx.degree_centrality(network).get(node, 0)
    closeness_centrality = nx.closeness_centrality(network).get(node, 0)
    betweenness_centrality = nx.betweenness_centrality(network, normalized=True).get(node, 0)
    page_rank = nx.pagerank(network, alpha=0.92, max_iter=iterations).get(node, 0)
    try:
        hits = nx.hits(network, max_iter=iterations)[1]  # authority score
        nauth = hits.get(node, 0)
    except nx.PowerIterationFailedConvergence:
        nauth = 0
    return {
        'deg_cen': degree_centrality,
        'clos_cen': closeness_centrality,
        'nbtw_cen': betweenness_centrality,
        'npr': page_rank,
        'nauth': nauth
    }

# print the centrality measures for nodes 1,50,100
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     G1 = nx.read_gml("friendships.gml.txt")
#
#     nodes_to_check = [1, 50, 100]
#
#     print("=== Centrality Measures ===")
#     for node in nodes_to_check:
#         print(f"Node {node}:")
#         result = centrality_measures(G1, node)
#         for k, v in result.items():
#             print(f"  {k}: {v:.4f}")
#         print()


def single_step_voucher(network):
    """
    Finds the best node to give the voucher to,
    when it can only be shared with direct friends (1 step away).

    The best node is the one with the most neighbors (highest degree).
    """
    best_node = None
    max_friends = -1

    for node in network.nodes:
        num_friends = len(list(network.neighbors(node)))
        if num_friends > max_friends:
            best_node = node
            max_friends = num_friends

    return best_node


# print the best for single-step voucher
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     G1 = nx.read_gml("friendships.gml.txt")
#     print(f"Best for single-step voucher: {single_step_voucher(G1)}")



def multiple_steps_voucher(network):
    """
    Finds the best node to give the voucher to,
    when it can be passed through the network with no step limit.

    The best node is the one that is, on average, closest to everyone else (closeness centrality).
    """
    best_node = None
    highest_closeness = -1

    closeness_scores = nx.closeness_centrality(network)

    for node, score in closeness_scores.items():
        if score > highest_closeness:
            best_node = node
            highest_closeness = score

    return best_node


# print the best for multiple-steps voucher
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     G1 = nx.read_gml("friendships.gml.txt")
#     print(f"Best for multiple-steps voucher: {multiple_steps_voucher(G1)}")

def multiple_steps_diminished_voucher(network):
    """
    Finds the best person to give the voucher to,
    when the voucher loses 10% of its value with each step,
    and it stops working after 4 steps.

    The goal is to get the most total value (benefit) from the voucher.
    """

    def total_benefit(source):
        # Find all nodes you can reach from 'source' in 4 steps or less
        lengths = nx.single_source_shortest_path_length(network, source, cutoff=4)
        # The benefit goes down 10% with every extra step
        benefit = 0
        for dist in lengths.values():
            benefit += 0.9 ** dist
        return benefit

    # Choose the person (node) that gives the highest total benefit
    return max(network.nodes, key=total_benefit)


# print the best for diminished-value voucher
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     G1 = nx.read_gml("friendships.gml.txt")
#     print(f"Best for diminished-value voucher: {multiple_steps_diminished_voucher(G1)}")

def find_most_valuable(network):
    betweenness = nx.betweenness_centrality(network, normalized=True)
    return max(betweenness, key=betweenness.get)

# #print the most valuable (likely to be targeted)
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     G1 = nx.read_gml("friendships.gml.txt")
#     print(f"Most valuable (likely to be targeted): {find_most_valuable(G1)}")
