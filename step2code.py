# ==========================================
# STEP 2: FINAL DC-SBM GENERATION
# TOPOLOGY-PRESERVING VERSION
# ==========================================

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random
import community as community_louvain
from scipy.stats import ks_2samp


# ==========================================
# CONFIGURATION
# ==========================================

SCALES = {
    "same": 1.0,
    "larger": 1.5,
    "smaller": 0.7
}

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

COLUMNS += [
    f"EXTRA_{i}"
    for i in range(40)
]


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():

    df = pd.read_csv(
        "data/NF-ToN-IoT.csv",
        header=None,
        names=COLUMNS
    )

    return df


# ==========================================
# LOAD FINGERPRINTS
# ==========================================

def load_fingerprints():

    with open(
        "results/step1/fingerprints.json",
        "r"
    ) as f:

        fingerprints = json.load(f)

    return fingerprints


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
# CREATE FOLDERS
# ==========================================

def create_folders(scale):

    base = f"results/step2/{scale}"

    os.makedirs(
        f"{base}/graphs",
        exist_ok=True
    )

    os.makedirs(
        f"{base}/comparison",
        exist_ok=True
    )

    # Step 3 will read edge lists from here
    os.makedirs(
        f"{base}/edges",
        exist_ok=True
    )

    return base


# ==========================================
# SCALE GRAPH
# ==========================================

def compute_scaled_sizes(
    original_nodes,
    original_edges,
    scale_factor
):

    new_nodes = max(
        2,
        int(original_nodes * scale_factor)
    )

    new_edges = int(
        new_nodes *
        original_edges /
        original_nodes
    )

    return new_nodes, new_edges


# ==========================================
# COMMUNITIES
# ==========================================

def detect_communities(G):

    partition = community_louvain.best_partition(
        G.to_undirected()
    )

    communities = {}

    for node, c in partition.items():

        communities.setdefault(
            c,
            []
        ).append(node)

    return communities, partition


# ==========================================
# COMMUNITY SIZES
# ==========================================

def compute_new_sizes(
    communities,
    new_nodes
):

    sizes = [
        len(v)
        for v in communities.values()
    ]

    total = sum(sizes)

    proportions = [
        s / total
        for s in sizes
    ]

    new_sizes = [
        max(2, int(p * new_nodes))
        for p in proportions
    ]

    return new_sizes


# ==========================================
# OMEGA MATRIX
# ==========================================

def compute_omega(
    G,
    partition,
    communities
):

    block_ids = sorted(
        communities.keys()
    )

    idx = {
        b: i
        for i, b in enumerate(block_ids)
    }

    k = len(block_ids)

    omega = np.zeros((k, k))

    for u, v in G.edges():

        r = idx[partition[u]]

        s = idx[partition[v]]

        omega[r][s] += 1

    return omega, block_ids


# ==========================================
# DEGREE WEIGHTS
# ==========================================

def extract_theta(
    G,
    communities,
    partition,
    block_ids
):

    out_theta = {}

    in_theta = {}

    idx = {
        b: i
        for i, b in enumerate(block_ids)
    }

    for b, nodes in communities.items():

        i = idx[b]

        out_deg = np.array([
            G.out_degree(n)
            for n in nodes
        ], dtype=float)

        in_deg = np.array([
            G.in_degree(n)
            for n in nodes
        ], dtype=float)

        out_deg = np.maximum(
            out_deg,
            0.1
        )

        in_deg = np.maximum(
            in_deg,
            0.1
        )

        out_theta[i] = (
            out_deg / out_deg.sum()
        )

        in_theta[i] = (
            in_deg / in_deg.sum()
        )

    return out_theta, in_theta


# ==========================================
# SAMPLE THETA
# ==========================================

def sample_theta(
    theta,
    size
):

    idx = np.random.choice(
        len(theta),
        size=size,
        replace=True
    )

    sampled = theta[idx]

    return sampled / sampled.sum()


# ==========================================
# GENERATE DC-SBM
# ==========================================

def generate_dcsbm(
    new_sizes,
    omega,
    out_theta,
    in_theta,
    target_edges
):

    G_new = nx.DiGraph()

    k = len(new_sizes)

    offsets = []

    current = 0

    for size in new_sizes:

        offsets.append(current)

        G_new.add_nodes_from(
            range(current, current + size)
        )

        current += size

    omega = omega / omega.sum()

    for r in range(k):

        for s in range(k):

            edges_rs = int(
                omega[r][s] * target_edges
            )

            if edges_rs == 0:
                continue

            src_nodes = np.arange(
                new_sizes[r]
            ) + offsets[r]

            dst_nodes = np.arange(
                new_sizes[s]
            ) + offsets[s]

            src_probs = sample_theta(
                out_theta[r],
                new_sizes[r]
            )

            dst_probs = sample_theta(
                in_theta[s],
                new_sizes[s]
            )

            edge_set = set()

            max_trials = edges_rs * 20

            trials = 0

            while (
                len(edge_set) < edges_rs
                and trials < max_trials
            ):

                u = np.random.choice(
                    src_nodes,
                    p=src_probs
                )

                v = np.random.choice(
                    dst_nodes,
                    p=dst_probs
                )

                if u != v:

                    edge_set.add((u, v))

                trials += 1

            G_new.add_edges_from(edge_set)

    return G_new


# ==========================================
# FIX ISOLATED NODES
# ==========================================

def fix_isolated_nodes(G):

    isolated = [

        n for n in G.nodes()

        if G.degree(n) == 0
    ]

    connected = [

        n for n in G.nodes()

        if G.degree(n) > 0
    ]

    if len(connected) == 0:

        return G

    for node in isolated:

        target = random.choice(
            connected
        )

        G.add_edge(node, target)

    return G


# ==========================================
# CONNECT COMPONENTS
# ==========================================

def connect_components(G):

    UG = G.to_undirected()

    comps = list(
        nx.connected_components(UG)
    )

    if len(comps) <= 1:

        return G

    for i in range(len(comps) - 1):

        a = random.choice(
            list(comps[i])
        )

        b = random.choice(
            list(comps[i + 1])
        )

        G.add_edge(a, b)

    return G


# ==========================================
# RELABEL IPS
# ==========================================

def relabel_nodes(G):

    fake_ips = [

        f"10.0.{i//255}.{i%255}"

        for i in range(
            len(G.nodes())
        )
    ]

    mapping = dict(
        zip(
            G.nodes(),
            fake_ips
        )
    )

    G = nx.relabel_nodes(
        G,
        mapping
    )

    return G


# ==========================================
# METRICS
# ==========================================

def compute_metrics(G):

    avg_degree = (
        G.number_of_edges() /
        G.number_of_nodes()
    )

    partition = community_louvain.best_partition(
        G.to_undirected()
    )

    communities = len(
        set(partition.values())
    )

    modularity = community_louvain.modularity(
        partition,
        G.to_undirected()
    )

    return (
        avg_degree,
        communities,
        modularity
    )


# ==========================================
# KS TEST
# ==========================================

def compute_ks(G_orig, G_new):

    orig_in = [
        d for _, d
        in G_orig.in_degree()
    ]

    new_in = [
        d for _, d
        in G_new.in_degree()
    ]

    orig_out = [
        d for _, d
        in G_orig.out_degree()
    ]

    new_out = [
        d for _, d
        in G_new.out_degree()
    ]

    ks_in, _ = ks_2samp(
        orig_in,
        new_in
    )

    ks_out, _ = ks_2samp(
        orig_out,
        new_out
    )

    return ks_in, ks_out


# ==========================================
# SAVE GRAPH IMAGE
# ==========================================

def save_graph_image(
    G,
    attack,
    scale,
    base
):

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
        f"{attack.upper()} - {scale.upper()}"
    )

    plt.savefig(
        f"{base}/graphs/{attack}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================
# SAVE GRAPH EDGES
# Step 3 needs the edge list of every
# synthetic graph so it can generate one
# flow row per edge. We save src and dst
# IPs as a CSV. The graph object is not
# stored in memory between steps, so
# persisting to disk is the only bridge.
# ==========================================

def save_graph_edges(
    G,
    attack,
    base
):

    edges = [
        {
            "src": u,
            "dst": v
        }
        for u, v in G.edges()
    ]

    pd.DataFrame(edges).to_csv(
        f"{base}/edges/{attack}.csv",
        index=False
    )


# ==========================================
# PROCESS ATTACK
# ==========================================

def process_attack(
    df,
    fingerprints,
    attack,
    scale_name,
    scale_factor,
    base
):

    print(f"Processing {attack}...")

    df_attack = df[
        df["LABEL"] == attack
    ]

    G = build_graph(df_attack)

    original_nodes = G.number_of_nodes()

    original_edges = G.number_of_edges()

    new_nodes, new_edges = (
        compute_scaled_sizes(
            original_nodes,
            original_edges,
            scale_factor
        )
    )

    communities, partition = (
        detect_communities(G)
    )

    new_sizes = compute_new_sizes(
        communities,
        new_nodes
    )

    omega, block_ids = compute_omega(
        G,
        partition,
        communities
    )

    out_theta, in_theta = (
        extract_theta(
            G,
            communities,
            partition,
            block_ids
        )
    )

    G_new = generate_dcsbm(
        new_sizes,
        omega,
        out_theta,
        in_theta,
        new_edges
    )

    G_new = fix_isolated_nodes(G_new)

    G_new = connect_components(G_new)

    G_new = relabel_nodes(G_new)

    (
        avg_degree,
        new_comm,
        modularity
    ) = compute_metrics(G_new)

    ks_in, ks_out = compute_ks(
        G,
        G_new
    )

    original_comm = (
        fingerprints[attack]["communities"]
    )

    community_ratio = round(
        new_comm / original_comm,
        2
    )

    original_avg_degree = (
        original_edges /
        original_nodes
    )

    degree_ratio = round(
        avg_degree /
        original_avg_degree,
        2
    )

    save_graph_image(
        G_new,
        attack,
        scale_name,
        base
    )

    # Save edges so step 3 can read them
    save_graph_edges(
        G_new,
        attack,
        base
    )

    row = {

        "Attack": attack,

        "Original_Nodes": original_nodes,
        "New_Nodes": new_nodes,

        "Original_Edges": original_edges,
        "New_Edges": G_new.number_of_edges(),

        "Original_Avg_Degree":
            round(original_avg_degree, 4),

        "New_Avg_Degree":
            round(avg_degree, 4),

        "Degree_Ratio":
            degree_ratio,

        "Original_Communities":
            original_comm,

        "New_Communities":
            new_comm,

        "Community_Ratio":
            community_ratio,

        "Modularity":
            round(modularity, 4),

        "KS_InDegree":
            round(ks_in, 4),

        "KS_OutDegree":
            round(ks_out, 4),

        "Scale":
            scale_name,

        "Model":
            "DC-SBM-final"
    }

    return row


# ==========================================
# PROCESS SCALE
# ==========================================

def process_scale(
    df,
    fingerprints,
    scale_name,
    scale_factor
):

    print(
        f"\n===== {scale_name.upper()} ====="
    )

    base = create_folders(scale_name)

    rows = []

    for attack in fingerprints.keys():

        row = process_attack(
            df,
            fingerprints,
            attack,
            scale_name,
            scale_factor,
            base
        )

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        f"{base}/comparison.csv",
        index=False
    )


# ==========================================
# MAIN
# ==========================================

def main():

    df = load_dataset()

    fingerprints = load_fingerprints()

    for scale_name, scale_factor in SCALES.items():

        process_scale(
            df,
            fingerprints,
            scale_name,
            scale_factor
        )

    print("\nSTEP 2 COMPLETED")


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    main()