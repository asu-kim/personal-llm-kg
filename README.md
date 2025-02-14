# Overview
This is our repository of the working implementation of our proposed approach to the personalization of LLMs using Knowledge Graphs, namely, **personal-llm-kg**. 
Our implementation in this repository includes scripts for processing our **Conversation Calendar** dataset [conversation-calendar](https://huggingface.co/datasets/asu-kim/conversation-calendar/tree/main/Data/calendar) into a **Knowledge Graph (KG)** and then evaluating **LLM (Llama) models** using **ROUGE scores** and **BLEU scores**.
For more details, please refer to our research paper, to be presented and published at the **Web Conference 2025** in late April - early May ([WWW'25](https://doi.org/10.1145/3701716.3715473)), which describes the workflow, the method used to generate our dataset, and the evaluation results of this project.
Here is the link to our short explanation video : [WWW'25 explanation video](https://www.youtube.com/watch?v=lwW8FWrzwzM)

# Addtional materials
- Our conversation calendar dataset for evaluation of our proposed approach: [link](https://huggingface.co/datasets/asu-kim/conversation-calendar/tree/main/Data/calendar)
- Our YouTube video explaining our proposed approach:  [link](https://www.youtube.com/watch?v=lwW8FWrzwzM)
- Our WWW'25 paper: [link](https://doi.org/10.1145/3701716.3715473)


# Prerequisites 

You need Python installed, as the main scripts are written in Python.

## Library Dependencies
To run this project, the following dependencies are required. The model used in this repository has been quantized using 4-bit precision (bnb_4bit) and relies on bitsandbytes for efficient matrix operations and memory optimization. So specific versions of bitsandbytes, torch, and torchvision are mandatory for compatibility. 
While newer versions of other dependencies may work, the specific versions listed below have been tested and are recommended for optimal performance.

It is highly recommended to create a Python virtual environment or a Conda environment to manage dependencies. The available options for environment setup are listed below.
  
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
**Note:** Please access and use the pre-trained models, authentication keys must be obtained from the [Hugging Face repository](https://huggingface.co/settings/tokens). Ensure you have a valid API token and configure authentication.

Make sure the environment is properly configured to use CUDA for optimal GPU acceleration.

# Files and directories in this repository
- **`data/`** - Contains our dataset, which is also available as an open-source resource at [conversation-calendar](https://huggingface.co/datasets/asu-kim/conversation-calendar).
- **`scripts/`** - Contains the necessary files to run and obtain evaluation results.
  
  - **`knowledge_graph_calendar.py`** - Converts calendar data into **knowledge graphs (KGs)**.
  - **`knowledge_graph.py`** - Converts conversational data into **knowledge graphs (KGs)**.
  - **`main_ROUGE.py`** - Runs the **Llama models**, tests their performance, and evaluates **ROUGE scores** using both **KG** and **RAG (Retrieval-Augmented Generation)** methods.
  - **`main_BLEU.py`** - Runs the **Llama models**, tests their performance, and evaluates **BLEU scores** using both **KG** and **RAG (Retrieval-Augmented Generation)** methods.

# Execution Workflow 

Below is the workflow to execute our implementation with our dataset and reproduce our results.

### Step 1: Create the Knowledge Graph  
Run the **`knowledge_graph_calendar.py`** and **`knowledge_graph.py`** scripts to generate the knowledge graph.  

**Note:**  
- Ensure that you specify the correct file paths for the dataset and output files.  

Run the following commands:  

```
python knowledge_graph_calendar.py --input <add your output path for the calendar from dataset directory in JSON format> --output_csv <path to the output.csv file> --output_svg <path to save the vizualization of kg in svg> --output_png <path to save the vizualization of kg in png>
python knowledge_graph.py --input <add your output path for the conversation data from dataset directory in txt format> --output <path to the output.csv file> --image <path to save the vizualization of kg in png>
```
Example output found in your CSV file:

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


Example output found in the text file of the evaluation:
 
<pre>

Evaluation Log:
==================================================

Question 1: Our Approach

Query: When is the “Catch-up with Friends” event scheduled?

Generated Response: 

Answer: The "Catch-up with Friends" event is scheduled for August 3rd, 2024 from 16:00 to 17:30.

Golden Answer: The “Catch-up with Friends” event is scheduled on 2024-08-03 from 16:00 to 17:30.

Execution Time: 1.14 seconds
  rouge1: Precision=0.7895, Recall=0.8333, F1=0.8108
  rouge2: Precision=0.6667, Recall=0.7059, F1=0.6857
  rougeL: Precision=0.7895, Recall=0.8333, F1=0.8108

--------------------------------------------------

Question 1: Baseline

Query: When is the “Catch-up with Friends” event scheduled?

Generated Response: 

Please do not provide any unnecessary information or details.

Here are the events:

{"event": "Catch-up with Friends", "date": "2024-08-03", "time

Golden Answer: The “Catch-up with Friends” event is scheduled on 2024-08-03 from 16:00 to 17:30.

Execution Time: 1.20 seconds
  rouge1: Precision=0.3913, Recall=0.5000, F1=0.4390
  rouge2: Precision=0.2273, Recall=0.2941, F1=0.2564
  rougeL: Precision=0.3478, Recall=0.4444, F1=0.3902

--------------------------------------------------

</pre>

### Step 3: Monitoring GPU Performance (Optional)
In another terminal, monitor GPU performance and memory utilization while running the scripts, please use NVIDIA-SMI:
```
nvidia-smi
```
# Contributors
- Deeksha Prahlad (dprahlad@asu.edu), Ph.D. student at Arizona State University
- Chanhee Lee, Former visiting scholar at Arizona State University
- Dongha Kim, Ph.D. student at Arizona State University
- Hokeun Kim (hokeun@asu.edu, https://hokeun.github.io/), Assistant professor at Arizona State University 
