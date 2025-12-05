# CrisisFlow AI Engine (FastAPI)

## Overview

The CrisisFlow AI Engine serves as the backend infrastructure for real-time disaster intelligence. It leverages Natural Language Processing (NLP) and Unsupervised Learning to ingest raw communication data, classify critical information, and transform high-dimensional data into visualizable formats for the frontend dashboard.

This system is built to provide high-performance asynchronous request handling and a robust machine learning pipeline for clustering and dimensionality reduction.

* **Frontend Repository:** [Link to Frontend Repo]
* **Live Backend:** [Link to Deployed API]

---

## Technologies & Languages

This project is developed using **Python 3.9+** and relies on the following core libraries and frameworks:

* **Core Language:** Python
* **Web Framework:** FastAPI (Asynchronous Server)
* **Server Interface:** Uvicorn (ASGI)
* **Machine Learning & Data Processing:**
    * **scikit-learn:** K-Means Clustering and PCA (Principal Component Analysis).
    * **Sentence Transformers:** Semantic text embeddings (Model: `all-MiniLM-L6-v2`).
    * **Pandas & NumPy:** Data manipulation and numerical operations.

---

## Technical Architecture

The engine processes data through a five-step pipeline:

### 1. Ingestion
The system loads raw disaster-related text messages from the source dataset (`data/train.csv`).

### 2. Vectorization
Text data is converted into high-dimensional semantic embeddings using the **Sentence Transformers** library.
* **Model:** `all-MiniLM-L6-v2`
* **Purpose:** To capture the semantic meaning of messages beyond simple keyword matching.

### 3. Clustering
The engine applies **K-Means clustering** to semantic embeddings to group messages into logical categories automatically. Common clusters include:
* Medical Assistance
* Search and Rescue
* Fire/Hazard
* Infrastructure Damage

### 4. Dimensionality Reduction
**Principal Component Analysis (PCA)** is utilized to reduce the embedding vectors into 2-dimensional coordinates $(x, y)$. This allows the high-dimensional data to be plotted visually on the frontend map while preserving the relative distance between semantically similar messages.

### 5. API Distribution
Results are served via a RESTful architecture using FastAPI:
* **Endpoint:** `GET /data`
* **Payload:** Returns enriched JSON data containing message content, cluster labels, and visualization coordinates.

---

## Local Development Guide

Follow these steps to set up the environment locally.

### Prerequisites
* Python 3.10.19+
* pip

### 1. Install Dependencies
Navigate to the root directory and install the required packages:

```bash
pip install -r requirements.txt
```