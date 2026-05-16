# ==========================================
# STEP 3: GRAPH-CONDITIONED CTGAN
# FLOW GENERATION — GPU VERSION
# ==========================================
# Recommended environment:
#   pip install sdv torch
# CTGAN in SDV automatically uses CUDA
# if a GPU is available. No code change
# needed — PyTorch detects it at runtime.
#
# Expected training time per variant
# on a modern GPU (RTX 3060+): ~15 min
# Total for all 4 variants: ~1 hour
# ==========================================

import pandas as pd
import networkx as nx
import numpy as np
import os
import json
import warnings
from scipy.stats import ks_2samp
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


# ==========================================
# CONFIGURATION
# ==========================================

FINGERPRINTS_PATH = "results/step1/fingerprints.json"

DATA_PATH = "data/NF-ToN-IoT.csv"

STEP2_DIR = "results/step2"

STEP3_DIR = "results/step3"

SCALES = ["same", "larger", "smaller"]

# On GPU we can afford more rows per class.
# 20000 gives CTGAN a richer view of each
# distribution while staying well within
# GPU memory for all attack types.
MAX_ROWS_PER_ATTACK = 20000

# 300 epochs is the standard for CTGAN
# research papers. On GPU this is feasible.
EPOCHS = 300

# 500 is the CTGAN default and works well
# on GPU. Larger batches give more stable
# gradient estimates per step.
BATCH_SIZE = 500

# Generator and discriminator hidden layer
# sizes. (256, 256) gives the model more
# capacity to learn complex distributions.
GENERATOR_DIM = (256, 256)

DISCRIMINATOR_DIM = (256, 256)

# How many times CTGAN retries generating
# a row when a conditioning combination is
# rare. 1000 maximizes row recovery.
MAX_TRIES_PER_BATCH = 1000

# Flow feature columns CTGAN must generate.
# Conditioning columns and IPs are inputs.
TARGET_COLUMNS = [
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION",
    "PROTOCOL",
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "TCP_FLAGS"
]

# Ablation variants — conditioning columns
# added alongside LABEL for each variant.
# A: baseline (label only)
# B: label + degree information
# C: label + PageRank role information
# D: label + degree + roles (full model)
VARIANTS = {
    "A": [],
    "B": ["src_out_degree", "dst_in_degree"],
    "C": ["src_role", "dst_role"],
    "D": ["src_out_degree", "dst_in_degree",
          "src_role", "dst_role"]
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

COLUMNS += [f"EXTRA_{i}" for i in range(40)]


# ==========================================
# CREATE FOLDERS
# ==========================================

def create_folders():

    os.makedirs(f"{STEP3_DIR}/models", exist_ok=True)

    os.makedirs(f"{STEP3_DIR}/enriched", exist_ok=True)

    os.makedirs(f"{STEP3_DIR}/ks_tables", exist_ok=True)

    for scale in SCALES:

        os.makedirs(
            f"{STEP3_DIR}/generated/{scale}",
            exist_ok=True
        )


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():

    print("Loading dataset...")

    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS
    )

    print(f"Loaded {len(df)} rows.")

    return df


# ==========================================
# LOAD FINGERPRINTS
# ==========================================

def load_fingerprints():

    with open(FINGERPRINTS_PATH, "r") as f:

        fingerprints = json.load(f)

    return fingerprints


# ==========================================
# BUILD ORIGINAL SUBGRAPH
# Rebuilt from raw CSV for each attack.
# Needed to get exact per-node degrees
# for the enrichment lookup.
# ==========================================

def build_subgraph(df, attack):

    df_attack = df[df["LABEL"] == attack]

    G = nx.DiGraph()

    G.add_edges_from(
        zip(
            df_attack["IPV4_SRC_ADDR"],
            df_attack["IPV4_DST_ADDR"]
        )
    )

    return G


# ==========================================
# NODE LOOKUP HELPERS
# Return safe defaults for missing nodes.
# ==========================================

def get_node_degree(G, ip, degree_type):

    if ip not in G:
        return 0

    if degree_type == "out":
        return G.out_degree(ip)

    return G.in_degree(ip)


def get_node_role(node_roles, ip):

    return node_roles.get(ip, "low")


# ==========================================
# ENRICH FLOW TABLE FOR ONE ATTACK
# Adds four structural columns to each row:
#   src_out_degree — out-degree of the
#     source IP in the original subgraph
#   dst_in_degree  — in-degree of the
#     destination IP in the original subgraph
#   src_role — PageRank role of source IP
#   dst_role — PageRank role of dest IP
# These encode the structural position of
# each communicating IP in its subgraph,
# giving CTGAN a signal beyond attack label.
# ==========================================

def enrich_attack_flows(df_attack, G, node_roles):

    df_enriched = df_attack.copy()

    df_enriched["src_out_degree"] = (
        df_enriched["IPV4_SRC_ADDR"].apply(
            lambda ip: get_node_degree(G, ip, "out")
        )
    )

    df_enriched["dst_in_degree"] = (
        df_enriched["IPV4_DST_ADDR"].apply(
            lambda ip: get_node_degree(G, ip, "in")
        )
    )

    df_enriched["src_role"] = (
        df_enriched["IPV4_SRC_ADDR"].apply(
            lambda ip: get_node_role(node_roles, ip)
        )
    )

    df_enriched["dst_role"] = (
        df_enriched["IPV4_DST_ADDR"].apply(
            lambda ip: get_node_role(node_roles, ip)
        )
    )

    return df_enriched


# ==========================================
# ENRICH FULL DATASET
# Each attack type is enriched using its
# own subgraph and node_roles from step 1.
# Never mix attack types — roles are
# computed per-subgraph and have no
# cross-attack meaning.
# Saves to disk and reloads on re-runs
# to avoid recomputing (deterministic).
# ==========================================

def enrich_dataset(df, fingerprints):

    enriched_path = (
        f"{STEP3_DIR}/enriched/enriched_flows.csv"
    )

    if os.path.exists(enriched_path):

        print(
            f"\nEnriched table found on disk, "
            f"loading from {enriched_path}..."
        )

        df_full = pd.read_csv(enriched_path)

        print(f"  Loaded {len(df_full)} rows.")

        return df_full

    print("\nEnriching flow table...")

    enriched_parts = []

    for attack in fingerprints.keys():

        print(f"  Enriching {attack}...")

        df_attack = df[df["LABEL"] == attack]

        if len(df_attack) == 0:
            continue

        G = build_subgraph(df, attack)

        node_roles = fingerprints[attack]["node_roles"]

        df_enriched = enrich_attack_flows(
            df_attack, G, node_roles
        )

        enriched_parts.append(df_enriched)

    df_full = pd.concat(
        enriched_parts, ignore_index=True
    )

    keep_cols = (
        ["LABEL"]
        + TARGET_COLUMNS
        + ["src_out_degree", "dst_in_degree",
           "src_role", "dst_role"]
    )

    df_full = df_full[keep_cols]

    df_full.to_csv(enriched_path, index=False)

    print(f"  Enriched table: {len(df_full)} rows saved.")

    return df_full


# ==========================================
# SAMPLE TRAINING SUBSET
# We cap each attack class at
# MAX_ROWS_PER_ATTACK using stratified
# random sampling with a fixed seed.
# On GPU, 20000 rows per class is feasible
# and gives CTGAN a rich training signal.
# Classes smaller than the cap are kept
# in full (ransomware: 142 rows, etc.).
# The full enriched table stays on disk
# for KS evaluation — only training is
# capped.
# ==========================================

def sample_training_subset(df_enriched):

    print(
        f"\nSampling training subset "
        f"(max {MAX_ROWS_PER_ATTACK} per attack)..."
    )

    parts = []

    for attack, group in df_enriched.groupby("LABEL"):

        n_available = len(group)

        n_sample = min(n_available, MAX_ROWS_PER_ATTACK)

        sampled = group.sample(
            n=n_sample,
            random_state=42
        )

        parts.append(sampled)

        print(
            f"  {attack}: "
            f"{n_available} available "
            f"-> {n_sample} sampled"
        )

    df_sampled = pd.concat(parts, ignore_index=True)

    print(f"  Total training rows: {len(df_sampled)}")

    return df_sampled


# ==========================================
# BUILD TRAINING TABLE FOR ONE VARIANT
# Selects only the columns relevant to the
# variant. LABEL is always present so CTGAN
# conditions on attack type in all variants.
# ==========================================

def build_variant_table(df_sampled, variant_name):

    structural_cols = VARIANTS[variant_name]

    cols = (
        ["LABEL"]
        + TARGET_COLUMNS
        + structural_cols
    )

    return df_sampled[cols].copy()


# ==========================================
# TRAIN ONE CTGAN VARIANT
# GPU parameters:
#   epochs=300       — full training
#   batch_size=500   — stable gradients
#   generator_dim    — larger network
#   discriminator_dim — larger network
# Saves to disk so re-runs skip training.
# ==========================================

def train_variant(df_sampled, variant_name):

    print(f"\n  Training variant {variant_name}...")

    df_variant = build_variant_table(
        df_sampled, variant_name
    )

    print(
        f"  Training rows: {len(df_variant)}, "
        f"columns: {list(df_variant.columns)}"
    )

    metadata = SingleTableMetadata()

    metadata.detect_from_dataframe(df_variant)

    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        generator_dim=GENERATOR_DIM,
        discriminator_dim=DISCRIMINATOR_DIM,
        verbose=True
    )

    synthesizer.fit(df_variant)

    model_path = (
        f"{STEP3_DIR}/models/ctgan_{variant_name}.pkl"
    )

    synthesizer.save(model_path)

    print(f"  Saved to {model_path}")

    return synthesizer


# ==========================================
# LOAD OR TRAIN VARIANT
# Loads from disk if model already exists.
# This means a run that crashes mid-way
# can resume from the last saved variant
# without retraining completed ones.
# ==========================================

def load_or_train_variant(df_sampled, variant_name):

    model_path = (
        f"{STEP3_DIR}/models/ctgan_{variant_name}.pkl"
    )

    if os.path.exists(model_path):

        print(
            f"  Variant {variant_name}: "
            f"loading from {model_path}"
        )

        synthesizer = CTGANSynthesizer.load(model_path)

    else:

        synthesizer = train_variant(
            df_sampled, variant_name
        )

    return synthesizer


# ==========================================
# TRAIN ALL VARIANTS
# ==========================================

def train_all_variants(df_sampled):

    print("\n===== TRAINING CTGAN VARIANTS =====")

    synthesizers = {}

    for variant_name in VARIANTS.keys():

        synthesizers[variant_name] = (
            load_or_train_variant(
                df_sampled, variant_name
            )
        )

    return synthesizers


# ==========================================
# ASSIGN PAGERANK ROLES TO SYNTHETIC GRAPH
# Synthetic nodes are new fake IPs not
# present in fingerprints.json, so we
# cannot look them up in the original
# node_roles. Instead we compute PageRank
# on the synthetic graph itself and
# discretize exactly as step 1 did
# (percentile 33/66 boundaries).
# This is methodologically correct: a
# node's role is a property of its graph
# position, not of its IP address string.
# ==========================================

def assign_pagerank_roles(G):

    pr = nx.pagerank(G)

    values = np.array(list(pr.values()))

    low_thresh = np.percentile(values, 33)

    high_thresh = np.percentile(values, 66)

    node_roles = {}

    for node, score in pr.items():

        if score <= low_thresh:
            node_roles[node] = "low"

        elif score <= high_thresh:
            node_roles[node] = "medium"

        else:
            node_roles[node] = "high"

    return node_roles


# ==========================================
# LOAD SYNTHETIC GRAPH FROM EDGE CSV
# Reloads the edge list saved by step 2
# into a DiGraph for degree and PageRank
# computation on the synthetic graph.
# ==========================================

def load_synthetic_graph(scale, attack):

    edge_path = (
        f"{STEP2_DIR}/{scale}/edges/{attack}.csv"
    )

    df_edges = pd.read_csv(edge_path)

    G = nx.DiGraph()

    G.add_edges_from(
        zip(df_edges["src"], df_edges["dst"])
    )

    return G, df_edges


# ==========================================
# BUILD CONDITIONING ROW FOR ONE EDGE
# Produces the conditioning dict for one
# (src, dst) edge based on its structural
# properties in the synthetic graph.
# Only includes columns that the given
# variant uses — passing extra columns
# to CTGAN would confuse it.
# ==========================================

def build_conditioning_row(
    src, dst, G, node_roles, attack, variant_name
):

    structural_cols = VARIANTS[variant_name]

    row = {"LABEL": attack}

    if "src_out_degree" in structural_cols:
        row["src_out_degree"] = G.out_degree(src)

    if "dst_in_degree" in structural_cols:
        row["dst_in_degree"] = G.in_degree(dst)

    if "src_role" in structural_cols:
        row["src_role"] = node_roles.get(src, "low")

    if "dst_role" in structural_cols:
        row["dst_role"] = node_roles.get(dst, "low")

    return row


# ==========================================
# GENERATE FLOWS FOR ONE ATTACK / SCALE
#
# For every edge in the synthetic graph we
# generate one flow row from CTGAN using
# the edge's structural conditioning.
# Then src/dst IPs are attached so the
# final dataset preserves the topology.
#
# Length alignment:
# CTGAN may return fewer rows than
# requested when conditioning combinations
# are rare in training data. We align
# df_edges to however many rows CTGAN
# returned, taking the first N edges.
# MAX_TRIES_PER_BATCH=1000 maximises
# the number of rows recovered.
# ==========================================

def generate_flows_for_attack(
    synthesizer, scale, attack, variant_name
):

    G, df_edges = load_synthetic_graph(scale, attack)

    node_roles = assign_pagerank_roles(G)

    conditioning_rows = []

    for _, edge_row in df_edges.iterrows():

        src = edge_row["src"]
        dst = edge_row["dst"]

        cond_row = build_conditioning_row(
            src, dst, G, node_roles,
            attack, variant_name
        )

        conditioning_rows.append(cond_row)

    if len(conditioning_rows) == 0:
        return pd.DataFrame()

    df_conditions = pd.DataFrame(conditioning_rows)

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        df_generated = (
            synthesizer.sample_remaining_columns(
                df_conditions,
                max_tries_per_batch=MAX_TRIES_PER_BATCH
            )
        )

    n_generated = len(df_generated)

    n_requested = len(df_edges)

    if n_generated == 0:

        print(
            f"      Warning: CTGAN generated 0 rows "
            f"for {attack} ({scale}). Skipping."
        )

        return pd.DataFrame()

    if n_generated < n_requested:

        print(
            f"      Note: requested {n_requested}, "
            f"generated {n_generated} for "
            f"{attack} ({scale})."
        )

    # Align edge list to actual generated count
    df_edges_aligned = (
        df_edges.iloc[:n_generated].reset_index(drop=True)
    )

    df_generated = df_generated.reset_index(drop=True)

    df_generated["IPV4_SRC_ADDR"] = (
        df_edges_aligned["src"]
    )

    df_generated["IPV4_DST_ADDR"] = (
        df_edges_aligned["dst"]
    )

    df_generated["LABEL"] = attack

    df_generated["Scale"] = scale

    return df_generated


# ==========================================
# GENERATE FLOWS FOR ALL ATTACKS / SCALES
# Uses variant D (full conditioning) for
# the final dataset across all 3 scales.
# ==========================================

def generate_all_flows(synthesizers, fingerprints):

    print("\n===== GENERATING FLOWS (VARIANT D) =====")

    for scale in SCALES:

        print(f"\n  Scale: {scale}")

        all_rows = []

        for attack in fingerprints.keys():

            print(f"    {attack}...")

            df_generated = generate_flows_for_attack(
                synthesizers["D"],
                scale,
                attack,
                "D"
            )

            if len(df_generated) > 0:
                all_rows.append(df_generated)

        if len(all_rows) == 0:
            continue

        df_scale = pd.concat(
            all_rows, ignore_index=True
        )

        out_path = (
            f"{STEP3_DIR}/generated/{scale}/flows.csv"
        )

        df_scale.to_csv(out_path, index=False)

        print(
            f"  Saved {len(df_scale)} rows "
            f"to {out_path}"
        )


# ==========================================
# COMPUTE KS STATISTIC
# Compares real vs synthetic distributions
# for one feature column.
# Lower KS = better distribution match.
# ==========================================

def compute_ks_stat(real_col, synth_col):

    real_vals = pd.to_numeric(
        real_col, errors="coerce"
    ).dropna()

    synth_vals = pd.to_numeric(
        synth_col, errors="coerce"
    ).dropna()

    if len(real_vals) == 0 or len(synth_vals) == 0:
        return np.nan

    stat, _ = ks_2samp(real_vals, synth_vals)

    return round(stat, 4)


# ==========================================
# GENERATE FLOWS FOR ABLATION (ONE VARIANT)
# Always "same" scale so the comparison
# between variants has no scaling confound.
# Any KS difference between A/B/C/D is
# purely due to the conditioning variables.
# ==========================================

def generate_flows_for_ablation(
    synthesizer, fingerprints, variant_name
):

    all_rows = []

    for attack in fingerprints.keys():

        df_generated = generate_flows_for_attack(
            synthesizer, "same", attack, variant_name
        )

        if len(df_generated) > 0:
            all_rows.append(df_generated)

    if len(all_rows) == 0:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


# ==========================================
# BUILD KS TABLE FOR ONE VARIANT
# For each attack and each target column,
# compute KS between real and synthetic.
# Real distributions come from the full
# enriched table (all 1.3M rows), not the
# training sample — this gives a fair
# evaluation against the true distribution.
# ==========================================

def build_ks_table_for_variant(
    df_real_enriched, df_synthetic, variant_name
):

    rows = []

    attacks = df_real_enriched["LABEL"].unique()

    for attack in attacks:

        df_real_attack = df_real_enriched[
            df_real_enriched["LABEL"] == attack
        ]

        df_synth_attack = df_synthetic[
            df_synthetic["LABEL"] == attack
        ]

        row = {
            "Attack": attack,
            "Variant": variant_name
        }

        for col in TARGET_COLUMNS:

            if (col not in df_real_attack.columns
                    or col not in df_synth_attack.columns):

                row[col] = np.nan
                continue

            row[col] = compute_ks_stat(
                df_real_attack[col],
                df_synth_attack[col]
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ==========================================
# RUN ABLATION STUDY
# Generates flows from all 4 variants on
# "same" scale and computes KS for each.
# This produces the ablation table for the
# paper: rows=attacks, columns=features,
# four variants side by side.
# ==========================================

def run_ablation_study(
    synthesizers, df_enriched, fingerprints
):

    print("\n===== ABLATION STUDY =====")

    ks_tables = []

    for variant_name, synthesizer in synthesizers.items():

        print(f"\n  Variant {variant_name}...")

        df_synth = generate_flows_for_ablation(
            synthesizer, fingerprints, variant_name
        )

        ks_table = build_ks_table_for_variant(
            df_enriched, df_synth, variant_name
        )

        ks_tables.append(ks_table)

    df_ks_all = pd.concat(ks_tables, ignore_index=True)

    ks_path = f"{STEP3_DIR}/ks_tables/ablation_ks.csv"

    df_ks_all.to_csv(ks_path, index=False)

    print(f"\n  KS table saved to {ks_path}")

    return df_ks_all


# ==========================================
# PRINT KS SUMMARY
# Prints mean KS across all features per
# attack per variant. You want to see D
# scoring lower (better) than A across
# most attack types — that is the proof
# that graph conditioning helps.
# ==========================================

def print_ks_summary(df_ks_all):

    print(
        "\n===== KS SUMMARY "
        "(mean across features) ====="
    )

    numeric_cols = [
        c for c in df_ks_all.columns
        if c not in ["Attack", "Variant"]
    ]

    df_copy = df_ks_all.copy()

    df_copy["Mean_KS"] = (
        df_copy[numeric_cols].mean(axis=1)
    )

    pivot = df_copy.pivot_table(
        index="Attack",
        columns="Variant",
        values="Mean_KS"
    )

    print(pivot.round(4).to_string())

    print(
        "\nInterpretation: lower KS = better. "
        "D should score lower than A to prove "
        "graph conditioning improves flow fidelity."
    )


# ==========================================
# MAIN
# ==========================================

def main():

    create_folders()

    df = load_dataset()

    fingerprints = load_fingerprints()

    # Step 3.1 — enrich real flows with
    # structural features from step 1.
    # Skips if enriched CSV already exists.
    df_enriched = enrich_dataset(df, fingerprints)

    # Cap training rows per attack class.
    # Full enriched table kept for KS eval.
    df_sampled = sample_training_subset(df_enriched)

    # Step 3.2 — train all 4 CTGAN variants.
    # Loads from disk if already trained.
    synthesizers = train_all_variants(df_sampled)

    # Step 3.3 — generate flows for all
    # attacks and all 3 scales using
    # variant D (full conditioning).
    generate_all_flows(synthesizers, fingerprints)

    # Step 3.4 — ablation KS comparison
    # across all 4 variants on same scale.
    df_ks_all = run_ablation_study(
        synthesizers, df_enriched, fingerprints
    )

    print_ks_summary(df_ks_all)

    print("\nSTEP 3 COMPLETED")


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    main()