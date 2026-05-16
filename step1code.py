# ==========================================
# STEP 1: FINGERPRINT EACH ATTACK TYPE
# FINAL VERSION
# ==========================================

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import community as community_louvain


# ==========================================
# CONFIGURATION
# ==========================================

RESULTS_DIR = "results/step1"

COLUMNS = [
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "PROTOCOL",
    "FLOW_DURATION",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_COUNT",
    "FLAG",
    "LABEL"
]

extra_cols = 35 - len(COLUMNS)

COLUMNS += [
    f"EXTRA_{i}"
    for i in range(extra_cols)
]

ATTACK_COLORS = {
    "ddos": "red",
    "dos": "orange",
    "scanning": "blue",
    "password": "purple",
    "backdoor": "brown",
    "injection": "pink",
    "xss": "cyan",
    "mitm": "green",
    "ransomware": "black",
    "benign": "gray"
}


# ==========================================
# CREATE FOLDERS
# ==========================================

def create_folders():

    os.makedirs(
        f"{RESULTS_DIR}/graphs",
        exist_ok=True
    )

    os.makedirs(
        f"{RESULTS_DIR}/histograms",
        exist_ok=True
    )


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():

    print("Loading dataset...")

    df = pd.read_csv(
        "data/NF-ToN-IoT.csv",
        header=None,
        names=COLUMNS
    )

    print("Dataset loaded!")

    return df


# ==========================================
# BUILD GRAPH
# ==========================================

def build_graph(df_attack):

    G = nx.DiGraph()

    G.add_edges_from(
        zip(
            df_attack["IPV4_SRC_ADDR"],
            df_attack["IPV4_DST_ADDR"]
        )
    )

    return G


# ==========================================
# DEGREE FEATURES
# ==========================================

def compute_degree_features(G):

    in_deg = dict(G.in_degree())

    out_deg = dict(G.out_degree())

    avg_in = (
        np.mean(list(in_deg.values()))
        if len(in_deg) > 0 else 0
    )

    avg_out = (
        np.mean(list(out_deg.values()))
        if len(out_deg) > 0 else 0
    )

    return (
        in_deg,
        out_deg,
        avg_in,
        avg_out
    )


# ==========================================
# COMMUNITY DETECTION
# ==========================================

def detect_communities(G):

    if G.number_of_nodes() == 0:

        return 0, [], {}

    partition = community_louvain.best_partition(
        G.to_undirected()
    )

    num_comm = len(
        set(partition.values())
    )

    community_sizes = {}

    for c in partition.values():

        community_sizes[c] = (
            community_sizes.get(c, 0) + 1
        )

    return (
        num_comm,
        list(community_sizes.values()),
        partition
    )


# ==========================================
# PAGERANK ROLES
# ==========================================

def compute_pagerank_roles(G):

    if G.number_of_nodes() == 0:

        return {}, {
            "low": 0,
            "medium": 0,
            "high": 0
        }

    pr = nx.pagerank(G)

    values = np.array(
        list(pr.values())
    )

    low_t = np.percentile(values, 33)

    high_t = np.percentile(values, 66)

    node_roles = {}

    role_counts = {
        "low": 0,
        "medium": 0,
        "high": 0
    }

    for node, score in pr.items():

        if score <= low_t:

            role = "low"

        elif score <= high_t:

            role = "medium"

        else:

            role = "high"

        node_roles[node] = role

        role_counts[role] += 1

    return node_roles, role_counts


# ==========================================
# SAVE GRAPH IMAGE
# ==========================================

def save_graph_image(G, attack):

    plt.figure(figsize=(8, 8))

    pos = nx.spring_layout(
        G,
        seed=42
    )

    color = ATTACK_COLORS.get(
        attack.lower(),
        "gray"
    )

    nx.draw(
        G,
        pos,
        node_color=color,
        edge_color=color,
        node_size=40,
        width=0.5,
        alpha=0.7,
        with_labels=False,
        arrows=False
    )

    plt.title(
        f"{attack.upper()} Graph"
    )

    plt.savefig(
        f"{RESULTS_DIR}/graphs/{attack}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================
# SAVE HISTOGRAMS
# ==========================================

def save_histograms(
    in_deg,
    out_deg,
    attack
):

    color = ATTACK_COLORS.get(
        attack.lower(),
        "gray"
    )

    # IN-DEGREE

    plt.figure(figsize=(6, 4))

    plt.hist(
        list(in_deg.values()),
        bins=20,
        color=color
    )

    plt.title(
        f"{attack.upper()} In-Degree"
    )

    plt.xlabel("Degree")

    plt.ylabel("Frequency")

    plt.savefig(
        f"{RESULTS_DIR}/histograms/in_{attack}.png",
        dpi=300
    )

    plt.close()

    # OUT-DEGREE

    plt.figure(figsize=(6, 4))

    plt.hist(
        list(out_deg.values()),
        bins=20,
        color=color
    )

    plt.title(
        f"{attack.upper()} Out-Degree"
    )

    plt.xlabel("Degree")

    plt.ylabel("Frequency")

    plt.savefig(
        f"{RESULTS_DIR}/histograms/out_{attack}.png",
        dpi=300
    )

    plt.close()


# ==========================================
# PROCESS ATTACK
# ==========================================

def process_attack(df, attack):

    print(f"Processing {attack}...")

    df_attack = df[
        df["LABEL"] == attack
    ]

    G = build_graph(df_attack)

    nodes = G.number_of_nodes()

    edges = G.number_of_edges()

    (
        in_deg,
        out_deg,
        avg_in,
        avg_out
    ) = compute_degree_features(G)

    (
        num_comm,
        comm_sizes,
        partition
    ) = detect_communities(G)

    (
        node_roles,
        role_counts
    ) = compute_pagerank_roles(G)

    flows = len(df_attack)

    save_graph_image(G, attack)

    save_histograms(
        in_deg,
        out_deg,
        attack
    )

    row = {

        "Attack": attack,

        "Nodes": nodes,

        "Edges": edges,

        "Avg_In_Degree": round(avg_in, 4),

        "Avg_Out_Degree": round(avg_out, 4),

        "Communities": num_comm,

        "Flows": flows
    }

    fingerprint = {

        "nodes": nodes,

        "edges": edges,

        "avg_in_degree": avg_in,

        "avg_out_degree": avg_out,

        "communities": num_comm,

        "community_sizes": comm_sizes,

        "pagerank_roles": role_counts,

        "partition": partition,

        "node_roles": node_roles
    }

    return row, fingerprint


# ==========================================
# SAVE RESULTS
# ==========================================

def save_results(
    results_table,
    fingerprints
):

    pd.DataFrame(
        results_table
    ).to_csv(

        f"{RESULTS_DIR}/results.csv",
        index=False
    )

    with open(
        f"{RESULTS_DIR}/fingerprints.json",
        "w"
    ) as f:

        json.dump(
            fingerprints,
            f,
            indent=4
        )


# ==========================================
# MAIN
# ==========================================

def main():

    create_folders()

    df = load_dataset()

    attacks = df["LABEL"].unique()

    results_table = []

    fingerprints = {}

    for attack in attacks:

        row, fingerprint = process_attack(
            df,
            attack
        )

        results_table.append(row)

        fingerprints[attack] = fingerprint

    save_results(
        results_table,
        fingerprints
    )

    print("\nSTEP 1 COMPLETED")


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    main()