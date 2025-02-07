import json
import csv
import spacy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches
import argparse
from spacy.matcher import Matcher
from tqdm import tqdm

# Set display options for pandas
pd.set_option('display.max_colwidth', 200)

# Argument parser to get file paths from the user
parser = argparse.ArgumentParser(description="Extract knowledge graph from calendar JSON.")
parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file")
parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV file")
parser.add_argument("--output_svg", type=str, required=True, help="Path to save the knowledge graph in SVG format")
parser.add_argument("--output_png", type=str, required=True, help="Path to save the knowledge graph in PNG format")
args = parser.parse_args()

# Function to load JSON data
def load_json(filepath):
    try:
        with open(filepath, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File {filepath} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON in {filepath}.")
        return {}

# Function to extract triples from the calendar data
def extract_triples(data):
    triples = []
    for month, sections in data.get("AlexCalendar2024", {}).items():  # Extract months
        for section, events in sections.items():  # Extract event categories
            for event in events:
                # Ensure event is a dictionary before accessing keys
                if isinstance(event, dict):  
                    source = section  # Category as the source
                    triples.append((source, "hasEvent", event.get('event', '')))
                    triples.append((source, "atTime", event.get('time', '')))
                    triples.append((source, "onDate", event.get('date', '')))
                else:
                    print(f"Skipping invalid event data: {event}")  # Debugging
    return triples

# Function to save triples to a CSV file
def save_triples_to_csv(triples, filename):
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['source', 'edge', 'target'])
            writer.writerows(triples)
        print(f"Triples saved to {filename}")
    except Exception as e:
        print(f"Error saving triples to CSV: {e}")

# Function to create relations using SpaCy and a Matcher
def create_relation(text, nlp):
    data = nlp(text)
    matcher = Matcher(nlp.vocab)

    # Define the pattern
    pattern = [{'DEP': 'ROOT'},
               {'DEP': 'prep', 'OP': "?"},
               {'DEP': 'agent', 'OP': "?"},
               {'POS': 'ADJ', 'OP': "?"}]
    matcher.add("matching_1", [pattern])
    matches = matcher(data)
    if matches:
        span = data[matches[-1][1]:matches[-1][2]]
        return span.text
    return ""

# Function to build a knowledge graph and save to a CSV
def build_knowledge_graph(subjects, objects, relations, output_csv):
    data_list = [{'source': s, 'target': t, 'edge': r} for s, t, r in zip(subjects, objects, relations)]
    df_graph = pd.DataFrame(data_list)
    df_graph.to_csv(output_csv, index=False)
    print(f"Knowledge graph saved to {output_csv}")
    return df_graph

# Function to visualize the knowledge graph
def visualize_graph(df_graph, output_svg, output_png):
    # Create a MultiDiGraph
    G = nx.from_pandas_edgelist(df_graph, "source", "target", edge_attr="edge", create_using=nx.MultiDiGraph())

    # Convert MultiDiGraph to DiGraph for edge label compatibility
    simplified_G = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if simplified_G.has_edge(u, v):
            simplified_G[u][v]["edge"] += f", {data['edge']}"
        else:
            simplified_G.add_edge(u, v, edge=data["edge"])

    # Define layout for better aesthetics
    pos = nx.spring_layout(simplified_G, seed=42, k=0.5)  # Adjust k to control spacing

    # Define node colors
    node_colors = []
    for node in simplified_G.nodes:
        if "2024" in node:  # Date nodes
            node_colors.append("skyblue")
        elif ":" in node:  # Time nodes
            node_colors.append("orange")
        else:  # Event nodes
            node_colors.append("lightgreen")

    # Draw the graph
    plt.figure(figsize=(6, 6))  # Reduced figure size
    nx.draw(
        simplified_G,
        pos,
        with_labels=True,
        node_color=node_colors,
        edge_cmap=plt.cm.Blues,
        node_size=3000,  # Reduced node size
        font_size=7,  # Smaller font size for labels
        font_weight="bold",
        edge_color="black",
    )

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(simplified_G, "edge")
    nx.draw_networkx_edge_labels(
        simplified_G,
        pos,
        edge_labels=edge_labels,
        font_color="darkred",
        font_size=7,
        font_weight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=1),
    )

    # Simplify legend
    legend_elements = [
        mpatches.Patch(color="skyblue", label="Date Nodes"),
        mpatches.Patch(color="lightgreen", label="Event Nodes"),
        mpatches.Patch(color="orange", label="Time Nodes"),
    ]
    plt.legend(handles=legend_elements, loc="best", fontsize=10, title="Node Types", title_fontsize=12)

    # Save the graph as an image
    plt.savefig(output_svg, dpi=300, format="svg")
    plt.savefig(output_png, dpi=300, format="png")
    plt.show()

    print(f"Graph saved as {output_svg} and {output_png}")

def main():
    # Load JSON data
    data = load_json(args.input)

    # Extract triples
    all_triples = extract_triples(data)

    if not all_triples:  # Check if no valid triples were extracted
        print("❌ No valid triples found. Check the input JSON format.")
        return

    print("Extracted Triples:", all_triples)

    # Load SpaCy model
    nlp = spacy.load('en_core_web_sm')

    # Generate relation texts
    texts = [f"{t[0]} {t[1]} {t[2]}" for t in all_triples]
    relations = [create_relation(text, nlp) for text in tqdm(texts)]

    # Separate Source and Target from the triples for graph construction
    sources = [t[0] for t in all_triples]
    targets = [t[2] for t in all_triples]

    # Build and save the knowledge graph
    df_graph = build_knowledge_graph(sources, targets, relations, args.output_csv)

    if df_graph.empty:
        print("❌ Empty graph. No valid relationships found.")
        return

    # Visualize the knowledge graph
    visualize_graph(df_graph, args.output_svg, args.output_png)

# Run the main function
if __name__ == "__main__":
    main()