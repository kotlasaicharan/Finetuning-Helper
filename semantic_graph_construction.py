!pip install spacy nltk networkx
!python -m spacy download en_core_web_sm
!python -m nltk.downloader wordnet
# !python -m nltk.downloader omw-1.4
!python -m nltk.downloader stopwords

import spacy
import nltk
import networkx as nx
import matplotlib.pyplot as plt

from typing import Optional, List, Tuple, Dict, Set
from collections import defaultdict

from nltk.corpus import wordnet as wn
from nltk.corpus.reader import WordNetError
from nltk.wsd import lesk

# --------------------------- SETUP ---------------------------
nlp = spacy.load("en_core_web_sm")

# Optionally, define a set of extremely generic hypernyms we might skip
GENERIC_HYPERNYMS = {
    "entity.n.01", "object.n.01", "physical_entity.n.01",
    "whole.n.02", "artifact.n.01", "abstraction.n.06"
}

def spacy_pos_to_wordnet_pos(spacy_pos: str) -> Optional[str]:
    """Map spaCy POS to WordNet POS (N, V, A, R)."""
    if spacy_pos.startswith("N"):
        return wn.NOUN
    elif spacy_pos.startswith("V"):
        return wn.VERB
    elif spacy_pos.startswith("J"):
        return wn.ADJ
    elif spacy_pos.startswith("R"):
        return wn.ADV
    return None

def extended_lesk_wsd(token, sent_tokens) -> Optional[wn.synset]:
    """
    Use Lesk with POS; fallback to Lesk without POS if that fails.
    """
    wn_pos = spacy_pos_to_wordnet_pos(token.pos_)
    context_words = [t.text for t in sent_tokens]

    syn = lesk(context_words, token.text, pos=wn_pos)
    if syn:
        return syn

    syn_alt = lesk(context_words, token.text)
    return syn_alt

# --------------------------- Build Semantic Graph (Directed) ---------------------------
def build_semantic_graph(text: str) -> nx.DiGraph:
    """
    1) Parse text with spaCy
    2) For each token, run extended Lesk
    3) Create a node for each (token.lemma_, synset)
    4) Add edges for dependencies + hypernyms (skipping extremely generic hypernyms).
    5) Return a DiGraph (directed).
    """
    doc = nlp(text)
    G = nx.DiGraph()

    token_node_map = {}

    for sent in doc.sents:
        # skip single-char tokens if you want.
        content_tokens = [t for t in sent if t.is_alpha and not t.is_stop and len(t.text) > 1]

        for token in content_tokens:
            syn_obj = extended_lesk_wsd(token, content_tokens)
            syn_name = syn_obj.name() if syn_obj else "None"
            lemma_str = token.lemma_.lower()

            node_id = (lemma_str, syn_name)
            if not G.has_node(node_id):
                G.add_node(
                    node_id,
                    label=node_id,
                    lemma=lemma_str,
                    synset=syn_name,
                    pos=token.pos_
                )
            token_node_map[token] = node_id

        # Add dependency edges as directed edges
        for token in content_tokens:
            if token.head != token and token.head in content_tokens:
                parent_id = token_node_map[token.head]
                child_id = token_node_map[token]
                if parent_id != child_id:
                    G.add_edge(parent_id, child_id, relation=token.dep_)

        # Add hypernym edges (directed: child -> hypernym)
        for token in content_tokens:
            node_id = token_node_map[token]
            syn_name = node_id[1]
            if syn_name and syn_name != "None":
                try:
                    syn_obj = wn.synset(syn_name)
                except WordNetError:
                    continue
                for hyper in syn_obj.hypernyms():
                    if hyper.name() in GENERIC_HYPERNYMS:
                        continue
                    hyper_id = (hyper.name(), "hypernym")
                    if not G.has_node(hyper_id):
                        G.add_node(
                            hyper_id,
                            label=hyper.name(),
                            synset=hyper.name()
                        )
                    G.add_edge(node_id, hyper_id, relation="hypernym")

    return G

# --------------------------- Print Full Graph ---------------------------
def print_full_graph_details(G: nx.DiGraph):
    """
    Print a user-friendly summary of the full graph:
      - Number of nodes, number of edges
      - A concise list of each node with its lemma, synset, and POS
      - A concise list of each edge with its 'relation'
    """
    nodes_data = list(G.nodes(data=True))
    edges_data = list(G.edges(data=True))

    print("\n===== FULL GRAPH DETAILS =====")
    print(f"Total Nodes: {len(nodes_data)}")
    print(f"Total Edges: {len(edges_data)}\n")

    # ---- Print Nodes ----
    print("---- NODES ----")
    for i, (node_id, attrs) in enumerate(nodes_data, start=1):
        lemma = attrs.get("lemma", "None")
        synset = attrs.get("synset", "None")
        pos = attrs.get("pos", "None")
        print(f"{i}. {node_id} | lemma={lemma}, synset={synset}, pos={pos}")

    # ---- Print Edges ----
    print("\n---- EDGES ----")
    for j, (u, v, edge_data) in enumerate(edges_data, start=1):
        relation = edge_data.get("relation", "")
        print(f"{j}. {u} -> {v} [relation={relation}]")

# --------------------------- Visualize Entire Graph as Directed with Arrows ---------------------------
def visualize_entire_graph(G: nx.DiGraph, title="Full Semantic Graph (Directed)"):
    """
    Visualize the directed graph with enhanced arrow visibility
    """
    plt.figure(figsize=(15, 12))
    plt.title(title)

    # Get positions from undirected copy with increased spacing
    G_und = G.to_undirected()
    pos = nx.spring_layout(G_und, seed=42, k=3, iterations=100)  # Increased k for more spacing

    # Separate edges by relation
    dep_edges = []
    hyper_edges = []
    edge_labels = {}

    for (u,v) in G.edges():
        rel = G[u][v].get("relation","")
        edge_labels[(u,v)] = rel
        if rel == "hypernym":
            hyper_edges.append((u,v))
        else:
            dep_edges.append((u,v))

    # Node labels
    node_labels = {n: G.nodes[n].get("label", str(n)) for n in G.nodes()}

    # Draw nodes with reduced size
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size= 11)

    # Draw dependency edges with enhanced arrows
    nx.draw_networkx_edges(
        G, pos,
        edgelist=dep_edges,
        edge_color='gray',
        arrows=True,
        arrowsize=25,  # Increased arrow size
        arrowstyle='->',
        width=1.5,     # Increased edge width
        # connectionstyle='arc3,rad=0.2'
    )

    # Draw hypernym edges with enhanced arrows
    nx.draw_networkx_edges(
        G, pos,
        edgelist=hyper_edges,
        edge_color='blue',
        arrows=True,
        arrowsize=25,  # Increased arrow size
        arrowstyle='->',
        width=1.5,     # Increased edge width
        # connectionstyle='arc3,rad=-0.2'
    )

    # Draw edge labels with adjusted position
    edge_labels_pos = {(u, v): pos for (u, v) in edge_labels.keys()}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_color='red',
        font_size=7,
        label_pos=0.6  # Adjust label position along edge
    )

    plt.axis('off')
    plt.tight_layout()
    plt.show()

# --------------------------- 3) Community Detection ---------------------------
def get_all_hypernyms(synset: wn.synset):
    """Recursively gather hypernyms, skipping GENERIC_HYPERNYMS if desired."""
    visited = set()
    def recurse(s):
        for h in s.hypernyms():
            if h.name() in GENERIC_HYPERNYMS:
                continue
            if h not in visited:
                visited.add(h)
                recurse(h)
    recurse(synset)
    return visited

def find_communities_and_themes(G: nx.DiGraph):
    """
    Use Greedy Modularity for bigger communities.
    We'll keep single-node communities if they appear.
    """
    # For community detection, we treat the graph as undirected
    G_und = G.to_undirected()
    from networkx.algorithms.community import greedy_modularity_communities
    raw_comms = list(greedy_modularity_communities(G_und))

    results = {}
    for idx, comm in enumerate(raw_comms):
        token_nodes = []
        syn_objs = []

        for node_id in comm:
            if node_id[1] != "hypernym":
                data = G.nodes[node_id]
                token_nodes.append((node_id, data))

                syn_name = data.get("synset", None)
                if syn_name and syn_name != "None":
                    try:
                        s_obj = wn.synset(syn_name)
                        syn_objs.append(s_obj)
                    except:
                        pass

        # Summarize hypernyms for the community
        hyper_count = defaultdict(int)
        for s_obj in syn_objs:
            for hsyn in get_all_hypernyms(s_obj):
                hyper_count[hsyn] += 1
        sorted_hyper = sorted(hyper_count.items(), key=lambda x: x[1], reverse=True)
        top_hypernyms = [(h.name(), c) for (h, c) in sorted_hyper]

        results[idx] = {
            "token_nodes": token_nodes,
            "top_hypernyms": top_hypernyms
        }

    return results, raw_comms

# --------------------------- Print Communities ---------------------------
def print_community_details(G: nx.DiGraph, communities, community_data):
    """
    For each community:
      - print the token nodes
      - show dependency edges, hypernym edges
    """
    for cid, info in community_data.items():
        comm_nodes = communities[cid]
        print(f"\n=== Community {cid} ===")
        print("Token Nodes:")
        for (node_id, data) in info["token_nodes"]:
            print("   ", node_id)

        # Build subgraph for printing edges
        sub_g = G.subgraph(comm_nodes).copy()

        dep_edges = []
        hyper_edges = []

        for u,v in sub_g.edges():
            rel = sub_g[u][v].get("relation","")
            if rel == "hypernym":
                hyper_edges.append((u,v))
            else:
                dep_edges.append((u,v))

        print("Dependency Edges:")
        for (u,v) in dep_edges:
            rel = sub_g[u][v].get('relation','')
            u_lbl = sub_g.nodes[u].get("label", str(u))
            v_lbl = sub_g.nodes[v].get("label", str(v))
            print(f"  {u_lbl} -> {v_lbl} ({rel})")

        print("Hypernym Edges:")
        for (u,v) in hyper_edges:
            rel = sub_g[u][v].get('relation','')
            u_lbl = sub_g.nodes[u].get("label", str(u))
            v_lbl = sub_g.nodes[v].get("label", str(v))
            print(f"  {u_lbl} -> {v_lbl} ({rel})")

def get_community_details(G: nx.DiGraph, communities, community_data) -> str:
    """
    For each community:
      - Gather details about token nodes, dependency edges, and hypernym edges.
      - Return the details as a formatted string.
    """
    result = []

    for cid, info in community_data.items():
        comm_nodes = communities[cid]
        community_str = [f"\n=== Community {cid} ==="]
        community_str.append("Token Nodes:")

        for (node_id, data) in info["token_nodes"]:
            community_str.append(f"   {node_id}")

        # Build subgraph for extracting edges
        sub_g = G.subgraph(comm_nodes).copy()
        dep_edges = []
        hyper_edges = []

        for u, v in sub_g.edges():
            rel = sub_g[u][v].get("relation", "")
            u_lbl = sub_g.nodes[u].get("label", str(u))
            v_lbl = sub_g.nodes[v].get("label", str(v))
            if rel == "hypernym":
                hyper_edges.append((u_lbl, v_lbl, rel))
            else:
                dep_edges.append((u_lbl, v_lbl, rel))

        # Add dependency edges
        community_str.append("Dependency Edges:")
        for (u_lbl, v_lbl, rel) in dep_edges:
            community_str.append(f"  {u_lbl} -> {v_lbl} ({rel})")

        # Add hypernym edges
        community_str.append("Hypernym Edges:")
        for (u_lbl, v_lbl, rel) in hyper_edges:
            community_str.append(f"  {u_lbl} -> {v_lbl} ({rel})")

        # Append community details to the result
        result.append("\n".join(community_str))

    # Combine all community details into a single string
    return "\n".join(result)

# --------------------------- Visualize Community (Directed, separate edges) ---------------------------


def visualize_community(G: nx.DiGraph, comm_nodes: Set, title=""):
    """
    Visualize a single community with enhanced arrow visibility
    """
    sub_g = G.subgraph(comm_nodes).copy()
    plt.figure(figsize=(10,8))
    plt.title(title)

    # Get positions with increased spacing
    sub_und = sub_g.to_undirected()
    pos = nx.spring_layout(sub_und, seed=42, k=3, iterations=50)

    # Separate edges
    dep_edges = []
    hyper_edges = []
    edge_labels = {}

    for (u,v) in sub_g.edges():
        rel = sub_g[u][v].get("relation","")
        edge_labels[(u,v)] = rel
        if rel == "hypernym":
            hyper_edges.append((u,v))
        else:
            dep_edges.append((u,v))

    # Node labels
    node_labels = {n: sub_g.nodes[n].get("label", str(n)) for n in sub_g.nodes}

    # Draw nodes with reduced size
    nx.draw_networkx_nodes(sub_g, pos, node_color='lightblue', node_size= 500)
    nx.draw_networkx_labels(sub_g, pos, labels=node_labels, font_size= 11)

    # Draw dependency edges with enhanced arrows
    nx.draw_networkx_edges(
        sub_g, pos,
        edgelist=dep_edges,
        edge_color='gray',
        arrows=True,
        arrowsize=25,
        arrowstyle='->',
        width=1.5,
        # connectionstyle='arc3,rad=0.2'
    )

    # Draw hypernym edges with enhanced arrows
    nx.draw_networkx_edges(
        sub_g, pos,
        edgelist=hyper_edges,
        edge_color='blue',
        arrows=True,
        arrowsize=25,
        arrowstyle='->',
        width=1.5,
        # connectionstyle='arc3,rad=-0.2'
    )

    # Draw edge labels with adjusted position
    edge_labels_pos = {(u, v): pos for (u, v) in edge_labels.keys()}
    nx.draw_networkx_edge_labels(
        sub_g, pos,
        edge_labels=edge_labels,
        font_color='red',
        font_size=7,
        label_pos=0.6
    )

    plt.axis('off')
    # plt.tight_layout()
    plt.show()

# --------------------------- MAIN ---------------------------
if __name__ == "__main__":
    poem_text = """
Golden rays kiss the silent earth,
A hush of dew on leaves unfurls.
The river hums a gentle tune,
Cradling dreams beneath the moon.
    """

    semantic_graph = build_semantic_graph(poem_text)

    print_full_graph_details(semantic_graph)

    visualize_entire_graph(semantic_graph, title="Full Semantic Graph")

    community_data, communities = find_communities_and_themes(semantic_graph)

    print("\nNumber of communities found:", len(communities))

    print_community_details(semantic_graph, communities, community_data)
    string_ot = get_community_details(semantic_graph,communities, community_data )
    # 6) Visualize each community as Directed
    for i, comm_nodes in enumerate(communities):
        visualize_community(semantic_graph, comm_nodes, title=f"Community {i}")
