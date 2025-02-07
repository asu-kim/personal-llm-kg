from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import CSVLoader
from rouge_score import rouge_scorer
from torch import cuda, bfloat16
import transformers
from langchain.schema import Document
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt 
import json
import time
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import os

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
model_id = 'meta-llama/Llama-2-13b-chat-hf'
hf_auth = '<your_auth_key_here>'
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

# # Load CSV file of the knowledge graph generated
csv_file = "<path_to_csv_file>"

# Combine every 3 rows for conversational data
# df = pd.read_csv(csv_file)
# combined_rows = []
# for i in range(0, len(df), 3):
#     # Check if there are enough rows to form a complete set
#     if i + 2 < len(df):
#         combined_entry = f"{df.iloc[i]['object']} {df.iloc[i]['subject']} {df.iloc[i]['object']}; " \
#                          f"{df.iloc[i+1]['object']} {df.iloc[i+1]['subject']} {df.iloc[i+1]['object']}; " \
#                          f"{df.iloc[i+2]['object']} {df.iloc[i+2]['subject']} {df.iloc[i+2]['object']}"
#         combined_rows.append(combined_entry)

# # Convert combined rows to a DataFrame
# combined_df = pd.DataFrame({'text': combined_rows})
# Load CSV
df = pd.read_csv(csv_file)

# List to store combined rows
combined_rows = []

# Iterate in steps of 3
for i in range(0, len(df), 3):
    # Check if there are enough rows to form a complete set
    if i + 2 < len(df):
        combined_entry = f"{df.iloc[i]['source']} {df.iloc[i]['target']} {df.iloc[i]['edge']}; " \
                         f"{df.iloc[i+1]['source']} {df.iloc[i+1]['target']} {df.iloc[i+1]['edge']}; " \
                         f"{df.iloc[i+2]['source']} {df.iloc[i+2]['target']} {df.iloc[i+2]['edge']}"
        combined_rows.append(combined_entry)

# Convert combined rows to a DataFrame
combined_df = pd.DataFrame({'text': combined_rows})
# Save the combined data back to a CSV
combined_csv_file = "combined_knowledge_graph.csv"
combined_df.to_csv(combined_csv_file, index=False)

#Load and split combined data
loader = CSVLoader(combined_csv_file)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)


#Process JSON Data if you are working on the calendar dataset
# Load JSON file and dynamically generate jq_schema for all months
json_file = "<path_to_calendar_json>"
# Load the JSON data to inspect its structure and dynamically create the jq_schema
with open(json_file, "r") as file:
    data = json.load(file)
# Extract all months dynamically
months = data["AlexCalendar2024"].keys()
# Dynamically build jq_schema for all months and all categories
categories = [
    "MeetingWithFriends",
    "OfficeMeetings",
    "FamilyEvents",
    "FestivalHolidays",
    "Classes",
    "DailyRoutineEvents"
]
jq_schema = ", ".join(
    f".AlexCalendar2024.{month}.{category}[]" for month in months for category in categories
)
# Load documents from JSON using the dynamically generated jq_schema
loader = JSONLoader(file_path=json_file, jq_schema=jq_schema, text_content=False)
documents = loader.load()
# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_documents = text_splitter.split_documents(documents)



#for txt files if you are working on the conversation dataset
# txt_file = "<path_to_txt_data_file>"
# loader = TextLoader(file_path=txt_file, encoding="utf-8")  # Ensure correct encoding
# documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# split_documents = text_splitter.split_documents(documents)



print(f"Number of documents from CSV: {len(split_docs)}")
print(f"Number of documents from JSON: {len(split_documents)}")


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

# RetrievalQA setup our approach
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type='stuff',
    chain_type_kwargs={"prompt": prompt_template}
)

# RetrievalQA setup baseline
retriever_1 = vectorstore_1.as_retriever(search_type="similarity", search_kwargs={"k": 5})
rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever_1,
    chain_type='stuff',
    chain_type_kwargs={"prompt": prompt_template_1}
)

# Function to generate response
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

# Function to compute ROUGE score
def compute_rouge_score(generated_response, golden_answer):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(golden_answer, generated_response)

def load_questions_answers_from_file(file_path):
    """
    Reads questions and answers from a file and returns a list of dictionaries.
    Assumes each QA pair is in two consecutive lines with 'Question:' and 'Answer:' prefixes.
    """
    questions_and_answers = []
    with open(file_path, "r") as file:
        lines = file.readlines()

        
    for i in range(0, len(lines), 2):  # Process two lines at a time
        if lines[i].startswith("Question:") and lines[i + 1].startswith("Answer:"):
            question = lines[i].replace("Question:", "").strip()
            answer = lines[i + 1].replace("Answer:", "").strip()
            questions_and_answers.append({"query": question, "golden_answer": answer})
    
    return questions_and_answers
def plot_rouge_scores(rouge_1, rouge_2, rouge_l, save_path="<output_png_path .png format>", save_path_1="<output_svg_path .svg format>"):
    """
    Plot bar graphs comparing ROUGE-1, ROUGE-2, and ROUGE-L scores and save the plot as an image.

    Parameters:
    - rouge_1: Dictionary with ROUGE-1 scores for "Personalized" and "RAG".
    - rouge_2: Dictionary with ROUGE-2 scores for "Personalized" and "RAG".
    - rouge_l: Dictionary with ROUGE-L scores for "Personalized" and "RAG".
    - save_path: Path to save the generated plot image.
    """
    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    personalized_scores = [
        sum(rouge_1["Personalized"]) / len(rouge_1["Personalized"]),
        sum(rouge_2["Personalized"]) / len(rouge_2["Personalized"]),
        sum(rouge_l["Personalized"]) / len(rouge_l["Personalized"])
    ]
    rag_scores = [
        sum(rouge_1["RAG"]) / len(rouge_1["RAG"]),
        sum(rouge_2["RAG"]) / len(rouge_2["RAG"]),
        sum(rouge_l["RAG"]) / len(rouge_l["RAG"])
    ]
   

    x = range(len(metrics))
    plt.figure(figsize=(6, 5))
    plt.bar(x, personalized_scores, width=0.2, label='Our Approach', align='center')
    plt.bar([p + 0.2 for p in x], rag_scores, width=0.2, label='Baseline', align='center')
    plt.xticks([p + 0.2 for p in x], metrics)
    plt.ylabel("Score")
   
    plt.legend()
    # Annotate the highest values
    for i, score in enumerate(personalized_scores):
        plt.text(
            x=i, 
            y=score + 0.01, 
            s=f"{score:.2f}", 
            ha='center', 
            fontsize=7, 
            color='blue'
        )

    for i, score in enumerate(rag_scores):
        plt.text(
            x=i + 0.2, 
            y=score + 0.01, 
            s=f"{score:.2f}", 
            ha='center', 
            fontsize=7, 
            color='orange'
        )
    # Save the plot as an image
    plt.savefig(save_path, format='png', dpi=300)
    plt.savefig(save_path_1, format='svg', dpi=300)
    plt.close()

    print(f"Plot saved as {save_path}")

def plot_execution_times(avg_time_personalized, avg_time_rag, save_path="<output_png_path .png format>", save_path_1="<output_svg_path .svg format>"):
    """
    Plot bar graphs comparing execution times and save the plot as an image.

    Parameters:
    - avg_time_personalized: Average execution time for the personalized assistant.
    - avg_time_rag: Average execution time for the RAG assistant.
    - save_path: Path to save the generated plot image.
    """
    labels = ['Our Approach', 'Baseline']
    times = [avg_time_personalized, avg_time_rag]

    plt.figure(figsize=(5, 5))
    plt.bar(labels, times, color=['blue', 'orange'])
    plt.ylabel("Execution Time (seconds)")
    # plt.title("Average Execution Time Comparison")
    plt.savefig(save_path, format='png', dpi=300)
    plt.savefig(save_path_1, format='svg', dpi=300)
    plt.close()

    print(f"Execution time plot saved as {save_path}")

def plot_memory_usage(avg_memory_personalized, avg_memory_rag, save_path="<output_png_path .png format>"):
    """
    Plot bar graphs comparing memory usage and save the plot as an image.
    """
    labels = ['Our Approach', 'Baseline']
    memory = [avg_memory_personalized, avg_memory_rag]

    plt.figure(figsize=(5, 5))
    plt.bar(labels, memory, color=['blue', 'orange'])
    plt.ylabel("Memory Usage (MB)")
    # plt.title("Average Memory Usage Comparison")
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()

    print(f"Memory usage plot saved as {save_path}")

def evaluate_responses_combined(qa_pairs, output_file):
    """
    Evaluate responses using standard and RAG pipelines, write results to a file, and plot comparison bar graphs.
    Includes memory usage tracking.
    """
    # Collecting scores and memory usage for plotting
    rouge_1_scores = {"Personalized": [], "RAG": []}
    rouge_2_scores = {"Personalized": [], "RAG": []}
    rouge_l_scores = {"Personalized": [], "RAG": []}
    execution_times = {"Personalized": [], "RAG": []}
    memory_usages = {"Personalized": [], "RAG": []}

    process = psutil.Process(os.getpid())  

    with open(output_file, "w") as file:
        file.write("Evaluation Log:\n" + "=" * 50 + "\n\n")
        
        # Iterating over all the QA pairs
        for i, qa in enumerate(qa_pairs, 1):
            query = qa["query"]
            golden_answer = qa["golden_answer"]

            # response evaluation
          
            mem_before_personalized = process.memory_info().rss / 10**6
            start_time_personalized = time.time() 
            generated_response = generate_response(query)
            end_time_personalized = time.time()
            mem_after_personalized = process.memory_info().rss / 10**6  
             
            
            personalized_execution_time = end_time_personalized - start_time_personalized
            personalized_memory_usage = mem_after_personalized - mem_before_personalized
            execution_times["Personalized"].append(personalized_execution_time)
            memory_usages["Personalized"].append(personalized_memory_usage)
            rouge_scores = compute_rouge_score(generated_response, golden_answer)
            
            # Log results 
            file.write(f"Question {i}: Our Approach\n\n")
            file.write(f"Query: {query}\n\n")
            file.write(f"Generated Response: {generated_response}\n\n")
            file.write(f"Golden Answer: {golden_answer}\n\n")
            file.write(f"Execution Time: {personalized_execution_time:.2f} seconds\n")
            file.write(f"Memory Usage: {personalized_memory_usage:.2f} MB\n\n")
            
            for rouge_type, score in rouge_scores.items():
                file.write(f"  {rouge_type}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1={score.fmeasure:.4f}\n")
            file.write("\n" + "-" * 50 + "\n\n")
            
            # Collect ROUGE scores 
            rouge_1_scores["Personalized"].append(rouge_scores["rouge1"].fmeasure)
            rouge_2_scores["Personalized"].append(rouge_scores["rouge2"].fmeasure)
            rouge_l_scores["Personalized"].append(rouge_scores["rougeL"].fmeasure)

            # RAG response evaluation
            start_time_rag = time.time()  
            mem_before_rag = process.memory_info().rss / 10**6  
            generated_response_rag = generate_response_rag(query)
            mem_after_rag = process.memory_info().rss / 10**6  
            end_time_rag = time.time()  
            
            rag_execution_time = end_time_rag - start_time_rag
            rag_memory_usage = mem_after_rag - mem_before_rag
            execution_times["RAG"].append(rag_execution_time)
            memory_usages["RAG"].append(rag_memory_usage)
            rouge_scores_rag = compute_rouge_score(generated_response_rag, golden_answer)
            
            # Log results 
            file.write(f"Question {i}: Baseline\n\n")
            file.write(f"Query: {query}\n\n")
            file.write(f"Generated Response: {generated_response_rag}\n\n")
            file.write(f"Golden Answer: {golden_answer}\n\n")
            file.write(f"Execution Time: {rag_execution_time:.2f} seconds\n")
            file.write(f"Memory Usage: {rag_memory_usage:.2f} MB\n\n")
            
            for rouge_type, score in rouge_scores_rag.items():
                file.write(f"  {rouge_type}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1={score.fmeasure:.4f}\n")
            file.write("\n" + "-" * 50 + "\n\n")
            
            # Collect ROUGE scores 
            rouge_1_scores["RAG"].append(rouge_scores_rag["rouge1"].fmeasure)
            rouge_2_scores["RAG"].append(rouge_scores_rag["rouge2"].fmeasure)
            rouge_l_scores["RAG"].append(rouge_scores_rag["rougeL"].fmeasure)

    # Calculate average execution times and memory usage
    avg_time_personalized = sum(execution_times["Personalized"]) / len(execution_times["Personalized"])
    avg_time_rag = sum(execution_times["RAG"]) / len(execution_times["RAG"])
    # avg_memory_personalized = sum(memory_usages["Personalized"]) / len(memory_usages["Personalized"])
    # avg_memory_rag = sum(memory_usages["RAG"]) / len(memory_usages["RAG"])

    # Plot ROUGE scores comparison
    plot_rouge_scores(rouge_1_scores, rouge_2_scores, rouge_l_scores)

    # Plot Execution Time Comparison
    plot_execution_times(avg_time_personalized, avg_time_rag)

    # Plot Memory Usage Comparison
    #plot_memory_usage(avg_memory_personalized, avg_memory_rag)

# # Path to the QA file
file_path = "<path_to_qa_file from the dataset>"
output_file = "<path_to_output_file eg output.txt>"

# Load QA pairs from the file
questions_and_answers = load_questions_answers_from_file(file_path)

# Run Evaluation
evaluate_responses_combined(questions_and_answers, output_file)


