
# HR Policy Assistant — Local RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions
from HR policy documents using local embeddings, ChromaDB, and Ollama.

## Features

- Ingest and chunk HR policy documents
- Generate embeddings using Sentence Transformers
- Store and retrieve document chunks using ChromaDB
- Generate answers using the local `llama3.2:3b` model
- Abstain when relevant policy information is unavailable
- Evaluate retrieval and answer quality
- Interactive dark-themed Streamlit chat interface

## Tech Stack

- Python
- Streamlit
- Sentence Transformers
- ChromaDB
- Ollama
- Llama 3.2 (3B)

## Architecture

Documents → Chunking → Embeddings → ChromaDB  
User Question → Query Embedding → Retrieval → Context Filtering  
Filtered Context → Ollama → Answer or Abstention

## Project Structure

```text
Rag_Project/
├── app.py
├── answer.py
├── rag.py
├── ingest.py
├── evaluate.py
├── evaluate_answers.py
├── evaluation_cases.json
├── inspect_relevance.py
├── calibrate_threshold.py
├── data/
│   └── hr_policy.txt
├── rag_db/
├── requirements.txt
└── README.md
```

## Run Locally

1. Install Python and Ollama.
2. Pull the model:

   `ollama pull llama3.2:3b`

3. Install Python dependencies:

   `python -m pip install -r requirements.txt`

4. Ingest the policy documents:

   `python ingest.py`

5. Start the chatbot:

   `python -m streamlit run app.py`

## Example Questions

- How many annual leave days do employees receive?
- How many unused annual leave days can be carried forward?
- How many days per week can employees work remotely?
- What is the maternity leave policy?

## Evaluation

The project includes scripts for retrieval evaluation, threshold calibration,
and manual review of answer faithfulness, relevance, and abstention.

Evaluation results depend on the test set and should not be interpreted
as a guarantee of accuracy on unseen questions.

## Limitations

- The current knowledge base contains a small HR policy document.
- Retrieval threshold tuning is provisional and requires broader testing.
- Local answer quality depends on the selected embedding and LLM models.
    