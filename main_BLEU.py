
# import statements
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import CSVLoader
from torch import cuda, bfloat16
import transformers
from langchain.schema import Document
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt 
import json
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import argparse
# Argument parser setup
parser = argparse.ArgumentParser(description="Process input, output, and image paths.")
parser.add_argument("--csv_file", type=str, required=True, help="Path to the input CSV file")
parser.add_argument("--json_file", type=str, help="Path to the input JSON file (optional)")
parser.add_argument("--txt_file", type=str, help="Path to the input TXT file (optional)")
parser.add_argument("--qa_file", type=str, required=True, help="Path to the QA file from the dataset")
parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
parser.add_argument("--plot_png_bleu", type=str, required=True, help="Path to save the plot (PNG format)")
parser.add_argument("--plot_svg_bleu", type=str, required=True, help="Path to save the plot (SVG format)")
parser.add_argument("--hf_auth", type=str, required=True, help="Hugging Face authentication token")

# Parse arguments
args = parser.parse_args()

# Assign variables dynamically
csv_file = args.csv_file
json_file = args.json_file
txt_file = args.txt_file
qa_file = args.qa_file
output_file = args.output_file
plot_png_bleu = args.plot_png_bleu
plot_svg_bleu = args.plot_svg_bleu
hf_auth = args.hf_auth
# Initialize Embedding Model id
embed_model_id = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
# Device configuration
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# Initialize Embedding Model 
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)
docs = [
    "this is one document",
    "and another document"
]
embeddings = embed_model.embed_documents(docs)

# Load Llama Model
model_id = 'meta-llama/Llama-2-7b-chat-hf'
# Device configuration
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  
    task='text-generation',
    temperature=1e-5,  
    max_new_tokens=50,  
    repetition_penalty=1.1  
)

# Load CSV file of the knowledge graph generated
csv_file = "<path_to_csv_file>"

# Combine every 3 rows
df = pd.read_csv(csv_file)


# Adjust the column names and logic based on your specific CSV structure
combined_rows = []
for i in range(0, len(df), 3):
    # Check if there are enough rows to form a complete set
    if i + 2 < len(df):
        combined_entry = f"{df.iloc[i]['edge']} {df.iloc[i]['source']} {df.iloc[i]['target']}; " \
                         f"{df.iloc[i+1]['edge']} {df.iloc[i+1]['source']} {df.iloc[i+1]['target']}; " \
                         f"{df.iloc[i+2]['edge']} {df.iloc[i+2]['source']} {df.iloc[i+2]['target']}"
        combined_rows.append(combined_entry)

# Convert combined rows to a DataFrame
combined_df = pd.DataFrame({'text': combined_rows})

# Save the combined data back to a CSV
combined_csv_file = "combined_knowledge_graph.csv"
combined_df.to_csv(combined_csv_file, index=False)

# Load and split combined data
loader = CSVLoader(combined_csv_file)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)

file_path = args.json_file if args.json_file else args.txt_file

if not file_path:
    raise ValueError("No input file provided. Please specify a JSON or TXT file.")

if file_path.endswith(".json"):
    # Process JSON Data if working on the calendar dataset
    with open(file_path, "r") as file:
        data = json.load(file)

    # Extract months dynamically
    months = data["AlexCalendar2024"].keys()
    categories = [
        "MeetingWithFriends",
        "OfficeMeetings",
        "FamilyEvents",
        "FestivalHolidays",
        "Classes",
        "DailyRoutineEvents"
    ]

    # Generate jq_schema dynamically
    jq_schema = ", ".join(
        f".AlexCalendar2024.{month}.{category}[]" for month in months for category in categories
    )

    # Load documents from JSON
    loader = JSONLoader(file_path=file_path, jq_schema=jq_schema, text_content=False)

elif file_path.endswith(".txt"):
    # Process TXT files if working on the conversation dataset
    loader = TextLoader(file_path=file_path, encoding="utf-8")  # Ensure correct encoding

else:
    raise ValueError("Unsupported file type. Please provide a .json or .txt file.")

# Load and split documents
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_documents = text_splitter.split_documents(documents)

# Create vector stores
vectorstore = FAISS.from_documents(split_docs, embed_model)
vectorstore_1 = FAISS.from_documents(split_documents, embed_model)

# Define LLM and Prompt Templates
llm = HuggingFacePipeline(pipeline=generate_text)

prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=(
    "Retrieve the answer from the knowledge graph {context} and generate a concise response to the {query}."

    )
)
prompt_template_1 = PromptTemplate(
    input_variables=["context","query" ],
    template=(
    "Retrieve the information from the {context} generate only a concise response to the {query}. " 
    
    )
)


# RetrievalQA setup for our approach
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type='stuff',
    chain_type_kwargs={"prompt": prompt_template}
)

# RetrievalQA setup for baseline
retriever_1 = vectorstore_1.as_retriever(search_type="similarity", search_kwargs={"k": 5})
rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever_1,
    chain_type='stuff',
    chain_type_kwargs={"prompt": prompt_template_1}
)

# Response Generation Functions
def generate_response(query):
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = prompt_template.format(context=context, query=query)
    return llm(prompt)

def generate_response_rag(query):
    relevant_docs = retriever_1.get_relevant_documents(query)   
    context_1 = "\n".join([doc.page_content for doc in relevant_docs])
    prompt_1 = prompt_template_1.format(context=context_1, query=query)
    return llm(prompt_1)

# Load Questions and Answers from File

def load_qa_file(file_path):
    """
    Reads questions and answers from a file and returns a list of dictionaries.
    Assumes each QA pair is in two consecutive lines with 'Question:' and 'Answer:' prefixes.
    """
    questions_and_answers = []
    with open(file_path, "r") as file:
        lines = file.readlines()
    print(f"Total lines in file: {len(lines)}")
    print("First 10 lines (for verification):")
    print("\n".join(lines[:10]))
        
    for i in range(0, len(lines), 2):  # Process two lines at a time
        if lines[i].startswith("Question:") and lines[i + 1].startswith("Answer:"):
            question = lines[i].replace("Question:", "").strip()
            answer = lines[i + 1].replace("Answer:", "").strip()
            questions_and_answers.append({"query": question, "golden_answer": answer})
    
    return questions_and_answers

#Compute BLEU Score
def compute_bleu_score(generated_response, golden_answer):
    """
    Compute BLEU score between the generated response and the golden answer.
    """
    reference = [golden_answer.split()]
    candidate = generated_response.split()
    return sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))

# Evaluate Responses and Plot BLEU Scores

def evaluate_responses_combined(qa_pairs, output_file):
    """
    Evaluate responses using standard and RAG pipelines, write results to a file, and plot BLEU score comparisons.

    Parameters:
    - qa_pairs: List of dictionaries with 'query' and 'golden_answer'.
    - output_file: Path to the file where results will be saved.
    """
    # Collecting BLEU scores for plotting
    bleu_scores = {"Personalized": [], "RAG": []}

    with open(output_file, "w") as file:
        file.write("Evaluation Log:\n" + "="*50 + "\n\n")
        
        # Iterating over all the QA pairs
        for i, qa in enumerate(qa_pairs, 1):
            query = qa["query"]
            golden_answer = qa["golden_answer"]
            generated_response = generate_response(query)
            bleu_score = compute_bleu_score(generated_response, golden_answer)
            file.write(f"Question {i}: {query}\nPersonalized Assistant: {generated_response}\nGolden Answer: {golden_answer}\n")
            file.write(f"  BLEU Score: {bleu_score:.4f}\n")
            file.write("\n" + "-" * 50 + "\n\n")
            bleu_scores["Personalized"].append(bleu_score)

            generated_response_rag = generate_response_rag(query)
            bleu_score_rag = compute_bleu_score(generated_response_rag, golden_answer)
            file.write(f"Question {i}: {query}\nAssistant (RAG): {generated_response_rag}\nGolden Answer: {golden_answer}\n")
            file.write(f"  BLEU Score: {bleu_score_rag:.4f}\n")
            file.write("\n" + "-" * 50 + "\n\n")
            bleu_scores["RAG"].append(bleu_score_rag)
    
    # Plot BLEU scores comparison
    plot_bleu_scores(bleu_scores, args.plot_png_bleu, args.plot_svg_bleu)
    
    
def plot_bleu_scores(bleu_scores, save_path, save_path_1):
    """
    Plot bar graphs comparing BLEU scores and save the plot as an image.

    Parameters:
    - bleu_scores: Dictionary with BLEU scores for "Personalized" and "RAG".
    - save_path: Path to save the generated plot image.
    - save_path_1: Path to save the plot in PNG format.
    """
    metrics = ["Our Approach", "Baseline"]
    scores = [
        sum(bleu_scores["Personalized"]) / len(bleu_scores["Personalized"]),
        sum(bleu_scores["RAG"]) / len(bleu_scores["RAG"]),
    ]
    
    plt.figure(figsize=(5, 5))
    bars = plt.bar(metrics, scores, color=['blue', 'green'])
    plt.ylabel("BLEU Scores")
    plt.xlabel("Methods")
    
    # Adjust the y-axis limits to leave space for text
    plt.ylim(0, max(scores) + 0.05)
    
    # Add text annotations for the bars
    for bar, score, color in zip(bars, scores, ['blue', 'green']):
        plt.text(
            x=bar.get_x() + bar.get_width() / 2,
            y=bar.get_height() + 0.01,  
            s=f"{score:.2f}",
            ha='center',
            fontsize=10,
            color=color
        )
    
    # Save the plot
    plt.savefig(save_path, format="svg", dpi=300)
    plt.savefig(save_path_1, format="png", dpi=300)
    plt.close()

    print(f"Plot saved as {save_path} and {save_path_1}")
# Path to the QA file
file_path_1 = args.qa_file
output_file = args.output_file

# Load QA pairs from the file

questions_and_answers = load_qa_file(file_path_1)

# Run Evaluation
evaluate_responses_combined(questions_and_answers, output_file)

