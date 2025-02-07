
import json
import csv
import spacy
import pandas as pd
from spacy.matcher import Matcher
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches
# Set display options for pandas
pd.set_option('display.max_colwidth', 200)

# Function to load JSON data
def load_json(filepath):
    try:
        with open(filepath, 'r') as file:
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
    # for month, sections in data.get("AlexCalendar2024", {}).items():
    for section, events in data.items():
        for event in events:
            # Creating triples as Subject, Predicate, Object
            Subject = section
            triples.append((Subject, "hasEvent", event.get('event', '')))
            triples.append((Subject, "atTime", event.get('time', '')))
            triples.append((Subject, "onDate", event.get('date', '')))
    return triples

# Function to save triples to a CSV file
def save_triples_to_csv(triples, filename):
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Subject', 'Predicate', 'Object'])
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



def visualize_graph(df_graph, output_image, output_image_1):
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
        # width=1.5,  # Reduced edge thickness
        edge_color="black",
    )

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(simplified_G, "edge")
    nx.draw_networkx_edge_labels(
    simplified_G,
    pos,
    edge_labels=edge_labels,
    font_color="darkred",
    font_size=7,  # Increase font size slightly
    font_weight="bold",
    bbox=dict(facecolor="white", edgecolor="none", alpha=1),  # Simplified bounding box
)

    # Simplify legend
    legend_elements = [
        mpatches.Patch(color="skyblue", label="Date Nodes"),
        mpatches.Patch(color="lightgreen", label="Event Nodes"),
        mpatches.Patch(color="orange", label="Time Nodes"),
    ]
    plt.legend(handles=legend_elements, loc="best", fontsize=10, title="Node Types", title_fontsize=12)

    # Save the graph as an image
    plt.savefig(output_image_1, dpi=300,  format="png")
    plt.savefig(output_image, dpi=300,  format="svg")
    plt.show()    
# Main function
def main():
    # Load JSON file
    json_file = "<path_to_caldendar_json_file>"

    # Load the JSON data
    with open(json_file, "r") as file:
        data = json.load(file)

    # Extract all months dynamically
    months = data.get("AlexCalendar2024", {})

    # Collect data for each month
    all_triples = []
    for month, month_data in months.items():
        print(f"Processing month: {month}")
        # Extract triples for the current month
        month_triples = extract_triples(month_data)
        all_triples.extend(month_triples)

    # If there are more than 10 triples, randomly sample 8
    # if len(all_triples) > 10:
    #     sampled_triples = random.sample(all_triples, 4)
    # else:
        sampled_triples = all_triples

    print("Sampled triples:", sampled_triples)

    # Load SpaCy model
    nlp = spacy.load('en_core_web_sm')

    # Generate relation texts
    texts = [f"{t[0]} {t[1]} {t[2]}" for t in sampled_triples]
    relations = [create_relation(text, nlp) for text in tqdm(texts)]

    # Separate Subject and Object from the triples for graph construction
    subjects = [t[0] for t in sampled_triples]
    objects = [t[2] for t in sampled_triples]

    # Build and save the knowledge graph
    df_graph = build_knowledge_graph(subjects, objects, relations, '<path_to_output in csv format>')

    # Visualize the knowledge graph
    visualize_graph(df_graph, '<path to image in svg format>', '<path to image in png format>')
# Run the main function
if __name__ == "__main__":
    main()

