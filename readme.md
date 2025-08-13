#  AI Mapping & Language Processing Tool

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-brightgreen)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

##  Overview
The **AI Mapping & Language Processing Tool** is a Python-based application for **semantic search**, **vector mapping**, and **AI-powered language understanding**.  

At its core, it:
- **Processes and stores vector embeddings** (e.g., from text or structured data)
- **Maps queries to relevant items** in a dataset (`mapping.json`)
- **Runs advanced NLP models** using Hugging Face Transformers
- **Provides a Streamlit-based web interface** for interactive exploration

This makes it useful for:
- Semantic search engines
- AI assistants that need quick knowledge retrieval
- Intelligent recommendation systems
- Domain-specific risk or medication lookup
- Educational AI projects

---

##  Key Features
- **Vector Mapping Engine**  
  Reads `mapping.json` and finds the closest vector match to a given input using similarity metrics.
- **Language Understanding**  
  Processes natural language using Transformer-based models for classification, matching, and retrieval.
- **Interactive Web App**  
  Built with Streamlit for real-time, visual interaction without needing to write extra code.
- **Model Downloader**  
  Automates downloading model weights (e.g., LLaMA) so users don’t need to manually manage large files.
- **Customizable Data**  
  Replace `mapping.json` with your own dataset or expand it with new embeddings.
- **Lightweight Setup**  
  Minimal dependencies and runs locally without complex infrastructure.

---

##  How It Works
1. **Data Preparation**  
   - Input datasets are vectorized using pre-trained language models.
   - Vectors and metadata are stored in `mapping.json`.
   
2. **Query Processing**  
   - A user inputs a query (text or other vectorizable content).
   - The system computes its embedding using the same model used for dataset preparation.
   
3. **Vector Matching**  
   - Similarity is calculated between the query vector and all dataset vectors.
   - The most relevant matches are returned.

4. **UI Display**  
   - Streamlit renders the matches in a simple, responsive interface.

---


