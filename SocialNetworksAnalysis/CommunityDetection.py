# Libraries:
import networkx as nx
import community as community_louvain
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community.kclique import k_clique_communities
from itertools import islice
import os
import json
from datetime import datetime
import pandas as pd
import zipfile
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyvis.network import Network
import matplotlib.colors as mcolors



################################################################################################
# Part 1 – Community Detection on Les Misérables Network


# def edge_selector_optimizer(G): #defult edge selector
#     betweenness = nx.edge_betweenness_centrality(G, weight='weight')
#     return max(betweenness, key=betweenness.get)

def edge_selector_optimizer(G):
    """
    Custom edge selector: weighted edge betweenness * (degree(u) + degree(v))
    """
    betweenness = nx.edge_betweenness_centrality(G, weight='weight')
    best_score = -1
    best_edge = None

    for (u, v), b in betweenness.items():
        degree_score = G.degree(u) + G.degree(v)
        score = b * degree_score  # small custom tweak
        if score > best_score:
            best_score = score
            best_edge = (u, v)

    return best_edge



def community_detector(network, algorithm_name, most_valualble_edge=None):
    """
    Runs Louvain, Girvan-Newman, or Clique Percolation and returns:
    {
        'num_partitions': int,
        'modularity': float,
        'partition': List of Lists
    }
    """
    if algorithm_name == 'louvain':
        partition_dict = community_louvain.best_partition(network, weight='weight')
        communities = {}
        for node, comm_id in partition_dict.items():
            communities.setdefault(comm_id, []).append(node)
        partition = list(communities.values())
        modularity = community_louvain.modularity(partition_dict, network, weight='weight')

    elif algorithm_name == 'girvin_newman':
        comp_gen = girvan_newman(network, most_valuable_edge=most_valualble_edge)
        best_modularity = -1
        best_partition = None

        for communities in islice(comp_gen, 30):  # limit to first 30 iterations
            partition = [list(c) for c in communities]
            if len(partition) > 50:  #stop if too fragmented
                break
            mod = nx.algorithms.community.modularity(network, partition, weight='weight')
            if mod > best_modularity:
                best_modularity = mod
                best_partition = partition

        partition = best_partition
        modularity = best_modularity

    elif algorithm_name == 'clique_percolation':
        best_modularity = -1
        best_partition = None
        best_k = None

        for k in range(3, 7):
            try:
                cliques = list(k_clique_communities(network.to_undirected(), k))
                if not cliques:
                    continue
                partition = [list(c) for c in cliques]

                m = network.size(weight='weight')
                degrees = dict(network.degree(weight='weight'))

                node_community_count = {}
                for community in partition:
                    for node in community:
                        node_community_count[node] = node_community_count.get(node, 0) + 1

                Q = 0.0
                for community in partition:
                    for i in community:
                        for j in community:
                            if i == j:
                                continue
                            A_ij = network[i][j]['weight'] if network.has_edge(i, j) else 0.0
                            expected = degrees[i] * degrees[j] / (2 * m)
                            t_i = node_community_count[i]
                            t_j = node_community_count[j]
                            Q += (A_ij - expected) / (t_i * t_j)
                modularity = Q / (2 * m)

                if modularity > best_modularity:
                    best_modularity = modularity
                    best_partition = partition
                    best_k = k
            except Exception as e:
                print(f"Error for k={k}: {e}")
                continue

        if best_partition is None:
            raise ValueError("No valid clique partition found")

        partition = best_partition
        modularity = best_modularity

    else:
        raise ValueError(f"Unknown algorithm name: {algorithm_name}")

    return {
        'num_partitions': len(partition),
        'modularity': modularity,
        'partition': partition
    }


# if __name__ == "__main__":
#     G = nx.les_miserables_graph()
#
#     print("----- Louvain -----")
#     louvain_result = community_detector(G, 'louvain')
#     print(f"Number of Partitions: {louvain_result['num_partitions']}")
#     print(f"Modularity: {louvain_result['modularity']}")
#     print(f"Partition: {louvain_result['partition']}\n")
#
#     print("----- Girvan-Newman -----")
#     gn_result = community_detector(G, 'girvin_newman', most_valualble_edge=edge_selector_optimizer)
#     print(f"Number of Partitions: {gn_result['num_partitions']}")
#     print(f"Modularity: {gn_result['modularity']}")
#     print(f"Partition: {gn_result['partition']}\n")
#
#     print("----- Clique Percolation -----")
#     cp_result = community_detector(G, 'clique_percolation')
#     print(f"Number of Partitions: {cp_result['num_partitions']}")
#     print(f"Overlapping Modularity (approx): {cp_result['modularity']}")
#     print(f"Partition: {cp_result['partition']}")


################################################################################################
# Part 2 – Community Detection on Twitter Political Network

def construct_heb_edges(files_path, start_date='2019-03-15', end_date='2019-04-15', non_parliamentarians_nodes=0):

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Unzip tweets into a subfolder
    extract_folder = os.path.join(files_path, "extracted")
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

        # Find the ZIP file (assumes there's only one zip in the folder)
        zip_files = [f for f in os.listdir(files_path) if f.endswith(".zip")]
        if zip_files:
            zip_path = os.path.join(files_path, zip_files[0])
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
        else:
            raise FileNotFoundError("No zip file found in folder to extract tweet data.")

    # Load central political players
    pol_path = os.path.join(files_path, "central_political_players.csv")
    central_df = pd.read_csv(pol_path)
    central_users = set(central_df.iloc[:, 0].astype(str))

    edge_dict = defaultdict(int)
    retweeter_counts = defaultdict(int)
    user_activity = defaultdict(set)

    # Read txt tweet files from extracted folder
    tweet_files = [f for f in os.listdir(extract_folder)
                   if f.startswith("Hebrew_tweets.json.") and f.endswith(".txt")]

    for fname in tweet_files:
        try:
            date_str = fname.split("json.")[1][:10]
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
        except:
            continue

        if not (start_dt <= file_date <= end_dt):
            continue

        file_path = os.path.join(extract_folder, fname)

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    tweet = json.loads(line)
                    if 'retweeted_status' not in tweet:
                        continue

                    retweeter = str(tweet['user']['id'])
                    original = str(tweet['retweeted_status']['user']['id'])

                    edge_dict[(retweeter, original)] += 1
                    user_activity[retweeter].add(original)

                    if retweeter not in central_users and original in central_users:
                        retweeter_counts[retweeter] += 1
                except:
                    continue

    # Include extra non-parliamentarians
    extra_users = set()
    if non_parliamentarians_nodes > 0:
        sorted_users = sorted(retweeter_counts.items(), key=lambda x: x[1], reverse=True)
        extra_users.update(uid for uid, _ in sorted_users[:non_parliamentarians_nodes])

    allowed_users = central_users.union(extra_users)

    filtered_dict = {
        (src, tgt): count
        for (src, tgt), count in edge_dict.items()
        if src in allowed_users and tgt in allowed_users
    }

    return filtered_dict


def construct_heb_network(edge_dict):
    """
    Builds a directed, weighted NetworkX graph from the edge dictionary.
    """
    G = nx.DiGraph()

    for (src, tgt), weight in edge_dict.items():
        G.add_edge(src, tgt, weight=weight)

    return G


# if __name__ == "__main__":
#
#     path = "D:\לימודים\ניתוח רשתות חברתיות\עבודה 2"
#
#     # Case 1: Only central political players
#     print("=== Central Players Only ===")
#     edges_only_central = construct_heb_edges(path, non_parliamentarians_nodes=0)
#     net_only_central = construct_heb_network(edges_only_central)
#     if net_only_central.number_of_edges() == 0:
#         print("Graph is empty — no edges to analyze.")
#     else:
#         net_only_central_undirected = net_only_central.to_undirected()
#         result1 = community_detector(net_only_central_undirected, 'girvin_newman', most_valualble_edge=edge_selector_optimizer)
#         print("Communities:", result1['num_partitions'])
#         print("Modularity:", round(result1['modularity'], 4))
#
#     # Case 2: With extra active users
#     print("\n=== With Extra Users ===")
#     edges_with_extra = construct_heb_edges(path, non_parliamentarians_nodes=20)
#     net_with_extra = construct_heb_network(edges_with_extra)
#     if net_only_central.number_of_edges() == 0:
#         print("Graph is empty — no edges to analyze.")
#     else:
#         net_with_extra_undirected = net_with_extra.to_undirected()
#         result2 = community_detector(net_with_extra_undirected, 'girvin_newman', most_valualble_edge=edge_selector_optimizer)
#         print("Communities:", result2['num_partitions'])
#         print("Modularity:", round(result2['modularity'], 4))
#
#
# def visualize_pyvis(graph, partition, filename="graph.html", id_to_name=None):
#     net = Network(height="800px", width="100%", directed=True, notebook=False)
#
#     # Community coloring
#     colors = list(mcolors.TABLEAU_COLORS.values())
#
#     for i, community in enumerate(partition):
#         color = colors[i % len(colors)]  # cycle through color list
#         for node in community:
#             label = id_to_name.get(node, node) if id_to_name else str(node)
#             size = graph.degree(node) * 2 + 10  # scale node size
#             net.add_node(node, label=label, color=color, size=size)
#
#     for u, v, data in graph.edges(data=True):
#         weight = data['weight'] if 'weight' in data else 1
#         net.add_edge(u, v, value=weight)
#
#     net.toggle_physics(True)
#     net.show_buttons(filter_=['physics'])
#     net.show(filename)
#
# # Optional: Load names from CSV
# df = pd.read_csv(os.path.join(path, "central_political_players.csv"))
# id_to_name = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1]))
#
# # Scenario A
# visualize_pyvis(
#     net_only_central_undirected,
#     result1['partition'],
#     filename="central_only_graph.html",
#     id_to_name=id_to_name
# )
#
# # Scenario B
# visualize_pyvis(
#     net_with_extra_undirected,
#     result2['partition'],
#     filename="central_plus_extra_graph.html",
#     id_to_name=id_to_name
# )
