# personal-llm-kg
Personalization of LLM using Knowledge Graphs. 
This is a repository containing scripts for processing the **Conversation Calendar** dataset [conversation-calendar](https://huggingface.co/datasets/asu-kim/conversation-calendar/tree/main/Data/calendar) into a **Knowledge Graph (KG)** and then evaluating **LLM (Llama) models** using **ROUGE scores** and **BLEU scores**.
For more details, please refer to our research paper, to be presented and published at the **Web Conference 2025** in late April - early May ([WWW'25](https://doi.org/10.1145/3701716.3715473)), which describes the workflow, the method used to generate our dataset, and the evaluation results of this project.

# Pre-requisite 

Ensure that Python is installed, as the main scripts are written in Python.

## 1. Python libraries
To run this project, the following dependencies are required. The model used in this repository has been quantized, so specific versions of `bitsandbytes`, `torch`, and `torchvision` are needed. While updated versions of other dependencies can be used, the specific versions mentioned below work best for this project. You can also create Python or Conda environments, and the available options are listed below.

TODO(Deeksha): Elaborate which are strict requirements and which are recommended and why specific versions are required.
  
```
pip install accelerate==1.1.0 
pip install transformers==4.46.3
pip install tokenizers==0.20.3
pip install bitsandbytes==0.41.0
pip install einops==0.6.1
pip install xformers==0.0.20
pip install langchain==0.0.240
pip install faiss-gpu==1.7.1.post3
pip install sentence_transformers==2.2.2
pip install torch==2.0.1
pip install torchvision==0.15.2
pip install attrs
pip install psutil
```
If you are using a conda environment:
```
conda install -c conda-forge psutil
conda install -c conda-forge attrs
```
For spacy model: 
```
pip install spacy
pip install requests
pip install idna
pip install click
pip install jinja2
pip install pandas
pip install pytz
pip install six
pip install matplotlib
pip install pillow
```
If you are using a conda environment:
```
conda install -c conda-forge spacy requests idna click jinja2 pandas pytz six matplotlib pillow
```
## System Requirements  

To ensure optimal performance, the following hardware and software requirements are utilized. \
**Note:** To replicate this model, you can use any equivalent hardware that meets the computational requirements.

### Hardware Requirements  
- **Processor**: Intel i9 or equivalent  
- **GPU**: NVIDIA RTX A6000  

### Software Requirements  
- **Python** (Ensure Python is installed)  
- **CUDA Version**: 12.4  
- **NVIDIA-SMI**: For monitoring GPU performance and memory utilization  

### Model Dependencies  
- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` [Hugging Face repository](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)  
- **Pre-trained Models**:  [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)  [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)  [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) \
**Note:** access and use the pre-trained models, authentication keys must be obtained from the [Hugging Face repository](https://huggingface.co/settings/tokens). Ensure you have a valid API token and configure authentication.

Make sure the environment is properly configured to use CUDA for optimal GPU acceleration.

# Files in the repository
- **`data/`** - Contains the dataset, which is available as an open-source resource at [conversation-calendar](https://huggingface.co/datasets/asu-kim/conversation-calendar).
- **`scripts/`** - This directory contains the necessary files to run and obtain evaluation results.
  
  - **`knowledge_graph_calendar.py`** - Converts calendar data into **knowledge graphs (KGs)**.
  - **`knowledge_graph.py`** - Converts conversational data into **knowledge graphs (KGs)**.
  - **`main_ROUGE.py`** - Runs the **Llama models**, tests their performance, and evaluates **ROUGE scores** using both **KG** and **RAG (Retrieval-Augmented Generation)** methods.
  - **`main_BLEU.py`** - Runs the **Llama models**, tests their performance, and evaluates **BLEU scores** using both **KG** and **RAG (Retrieval-Augmented Generation)** methods.

# Execution Workflow 

### Step 1: Create the Knowledge Graph  
Run the **`knowledge_graph_calendar.py`** and **`knowledge_graph.py`** scripts to generate the knowledge graph.  

**Note:**  
- Ensure that you specify the correct file paths for the dataset and output files.  

Run the following commands:  

```
python knowledge_graph_calendar.py --input <add your output path for the calendar from dataset directory in json format> --output_csv <path to the output.csv file> --output_svg <path to save the vizualisation of kg in svg> --output_png <path to save the vizualisation of kg in png>
python knowledge_graph.py --input <add your output path for the conversation data from dataset directory in txt format> --output <path to the output.csv file> --image <path to save the vizualisation of kg in png>
```
Example output:

**Knowledge Graph Edges**  
1. 
| Source              | Target                     | Edge                      |
|---------------------|--------------------------|---------------------------|
| MeetingWithFriends | Catch-up with Friends     | up with                   |
| MeetingWithFriends | 16:00 - 17:30            | MeetingWithFriends        |
| MeetingWithFriends | 2024-08-03               | MeetingWithFriends onDate |

2. 
| Subject | Relation  | Object                  |
|---------|----------|-------------------------|
| Alex    | guys     | August 17th             |
| Mia     | course   | What time               |
| Zoe     | be       | 8 PM                     |
| Liam    | Great    | drinks                   |
| Alex    | Yup      | your competitive spirit  |
| Zoe     | spirit   | *(empty)*                |
| Mia     | see      | a movie night            |

### Step 2: Generate and Evaluate Results

After obtaining the knowledge triples from the previous step, run **`main_ROUGE.py`** and **`main_BLEU.py`** to compute evaluation metrics based on ROUGE and BLEU scores.

**Note:** 
- Ensure that the correct file paths for the dataset, output files, and images are provided.
- Add the authentication key obtained from the Hugging Face repository.
- If you are using conversation dataset run this command to obtain the result of the evaluation.
```
python main_ROUGE.py --csv_file <add your output path for the kg from the generated file> --txt_file <add your output path for the conversation from dataset directory> --qa_file <add your output path for the qa from dataset directory> --output_file <add your output path in txt> --plot_png_rouge <add your output path in png> --plot_svg_rouge <add your output path in svg> --plot_png_execution_time <add your output path in png> --plot_svg_execution_time <add your output path in svg> --hf_auth <add your auth key>
python main_BLEU.py --csv_file <add your output path for the kg from the generated file> --txt_file <add your output path for the conversation from dataset directory> --qa_file <add your output path for the qa from dataset directory> --output_file <add your output path in txt> --plot_png_bleu <add your output path in png> --plot_svg_bleu <add your output path in svg> --hf_auth <add your auth key>
```
- If you are using calendar dataset run this command to obtain the result of the evaluation.
```
python main_ROUGE.py --csv_file <add your output path for the kg from the generated file> --json_file <add your output path for the calendar from dataset directory> --qa_file <add your output path for the qa from dataset directory> --output_file <add your output path in txt> --plot_png_rouge <add your output path in png> --plot_svg_rouge <add your output path in svg> --plot_png_execution_time <add your output path in png> --plot_svg_execution_time <add your output path in svg> --hf_auth <add your auth key>
python main_BLEU.py --csv_file <add your output path for the kg from the generated file> --json_file <add your output path for the calendar from dataset directory> --qa_file <add your output path for the qa from dataset directory> --output_file <add your output path in txt> --plot_png_bleu <add your output path in png> --plot_svg_bleu <add your output path in svg> --hf_auth <add your auth key>
```


Example output file of the evaluation:

**Evaluation Log**  

==================================================  

**Question 1: Our Approach**  

**Query:**  
_When is the “Catch-up with Friends” event scheduled?_  

**Generated Response:**  
The answer is: The "Catch-up with Friends" event is scheduled for August 3rd, 2024, at 16:00.  

**Golden Answer:**  
The “Catch-up with Friends” event is scheduled on 2024-08-03 from 16:00 to 17:30.  

- **Execution Time:** 0.79 seconds  
- **Memory Usage:** 13.63 MB  

| Metric  | Precision | Recall | F1 Score |
|---------|----------|--------|----------|
| **ROUGE-1** | 0.6111  | 0.6111 | 0.6111   |
| **ROUGE-2** | 0.4706  | 0.4706 | 0.4706   |
| **ROUGE-L** | 0.6111  | 0.6111 | 0.6111   |

--------------------------------------------------  

**Question 1: Baseline**  

**Query:**  
_When is the “Catch-up with Friends” event scheduled?_  

**Generated Response:**  
Please provide the exact date and time of the event.  
The "Catch-up with Friends" event is scheduled for August 3rd, 2024, from 16:00 to 17:  

**Golden Answer:**  
The “Catch-up with Friends” event is scheduled on 2024-08-03 from 16:00 to 17:30.  

- **Execution Time:** 0.98 seconds  
- **Memory Usage:** 0.26 MB  

| Metric  | Precision | Recall | F1 Score |
|---------|----------|--------|----------|
| **ROUGE-1** | 0.5185  | 0.7778 | 0.6222   |
| **ROUGE-2** | 0.4231  | 0.6471 | 0.5116   |
| **ROUGE-L** | 0.5185  | 0.7778 | 0.6222   |

--------------------------------------------------  

### Step 3: Monitoring GPU Performance (Optional)
In another terminal, monitor GPU performance and memory utilization while running the scripts, use NVIDIA-SMI:
```
nvidia-smi
```
# Contributors
- Deeksha Prahlad 
- Chanhee Lee
- Dongha Kim
- Hokeun Kim
