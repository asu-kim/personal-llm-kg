# # # ### https://spacy.io/api/sentencizer 

import spacy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample text
file_path = "<path_to_data>"
with open(file_path, "r") as file:
    text = file.read()

def extract_triplet(sent):
    subject = ""
    if ":" in sent:
        subject = sent.split(":", 1)[0].strip()
        sent = sent.split(":", 1)[1].strip()  

    doc = nlp(sent)
    relation = ""
    obj = ""

   
    for token in doc:
        # Main verb in the sentence
        if token.dep_ in {"ROOT", "VERB"}:  
            relation = token.text
            # descriptive context for relation
            if token.head.text != token.text:
                relation = f"{relation} {token.head.text}"
            break

    # Extract object 
    for chunk in doc.noun_chunks:
        # Direct or prepositional objects
        if chunk.root.dep_ in {"dobj", "pobj", "attr", "oprd", "nsubjpass"}:  
            obj = chunk.text
            break

    # Fall back to named entities for objects if not found in noun chunks
    if not obj:
        for ent in doc.ents:
            if ent.label_ in {"DATE", "TIME", "GPE", "ORG", "PERSON"}:  
                obj = ent.text
                break

    # Skip meaningless triples
    if not relation.strip() and not obj.strip():
        return None

    return [subject.strip(), relation.strip(), obj.strip()]

# Process the text into individual lines (split by newlines)
lines = text.split("\n")

# Extract entity triplets
triplets = [extract_triplet(line) for line in lines if line.strip()]
triplets = [t for t in triplets if t]  # Remove None values

# Create DataFrame
kg_df = pd.DataFrame(triplets, columns=["subject", "relation", "object"])

# Save to CSV
output_csv = "<path_to_output in csv format>"
kg_df.to_csv(output_csv, index=False)

print(f"Triples saved to {output_csv}")
print(kg_df)

# Visualize the Knowledge Graph
G = nx.MultiDiGraph()

# Add edges to the graph
for _, row in kg_df.iterrows():
    if row["subject"] and row["relation"] and row["object"]:
        G.add_edge(row["subject"], row["object"], key=row["relation"], label=row["relation"])

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
graph_image = "<path_to_image in .png format>"
plt.savefig(graph_image, format="PNG", dpi=300)
plt.show()

print(f"Knowledge graph visualization saved as {graph_image}")