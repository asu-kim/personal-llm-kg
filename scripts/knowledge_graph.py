# # # ### https://spacy.io/api/sentencizer 

import spacy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse

# Set up argument parser to get file paths from the user
parser = argparse.ArgumentParser(description="Extract knowledge graph from text.")
parser.add_argument("--input", type=str, required=True, help="Path to the input text file")
parser.add_argument("--output", type=str, required=True, help="Path to save the output CSV file")
parser.add_argument("--image", type=str, required=True, help="Path to save the knowledge graph image")
args = parser.parse_args()

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Read the input file
with open(args.input, "r") as file:
    text = file.read()

def extract_triplet(sent):
    source = ""
    if ":" in sent:
        source = sent.split(":", 1)[0].strip()
        sent = sent.split(":", 1)[1].strip()  

    doc = nlp(sent)
    edge = ""
    target = ""

    # Identify the main verb (relation)
    for token in doc:
        if token.dep_ in {"ROOT", "VERB"}:  
            edge = token.text
            if token.head.text != token.text:
                edge = f"{edge} {token.head.text}"
            break

    # Extract target (was "object")
    for chunk in doc.noun_chunks:
        if chunk.root.dep_ in {"dobj", "pobj", "attr", "oprd", "nsubjpass"}:  
            target = chunk.text
            break

    # Fall back to named entities for target if not found in noun chunks
    if not target:
        for ent in doc.ents:
            if ent.label_ in {"DATE", "TIME", "GPE", "ORG", "PERSON"}:  
                target = ent.text
                break

    # Skip meaningless triples
    if not edge.strip() and not target.strip():
        return None

    return [source.strip(), edge.strip(), target.strip()]

# Process the text into individual lines
lines = text.split("\n")

# Extract entity triplets
triplets = [extract_triplet(line) for line in lines if line.strip()]
triplets = [t for t in triplets if t]  

# Create DataFrame
kg_df = pd.DataFrame(triplets, columns=["source", "edge", "target"])

# Save to CSV
kg_df.to_csv(args.output, index=False)

print(f"Triples saved to {args.output}")
print(kg_df)

# Visualize the Knowledge Graph
G = nx.MultiDiGraph()

# Add edges to the graph
for _, row in kg_df.iterrows():
    if row["source"] and row["edge"] and row["target"]:
        G.add_edge(row["source"], row["target"], key=row["edge"], label=row["edge"])

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)

# Draw nodes and edges
nx.draw(
    G,
    pos=pos,
    with_labels=True,
    node_color="skyblue",
    edge_cmap=plt.cm.Blues,
    node_size=3000,
    font_size=10,
)

# Draw edge labels (relation names)
edge_labels = {(u, v): data["label"] for u, v, k, data in G.edges(data=True, keys=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=8)

# Save the graph as an image
plt.savefig(args.image, format="PNG", dpi=300)
plt.show()

print(f"Knowledge graph visualization saved as {args.image}")