# aws-generative-ai-rag-system
Production-grade Generative AI RAG system using AWS, OpenSearch, and Ollama
# Production-Grade RAG System on AWS

A scalable **Retrieval-Augmented Generation (RAG)** architecture built on AWS that processes documents, performs hybrid search, and generates contextual answers using LLMs.

This project demonstrates how to build a **production-style Generative AI system** using serverless services and vector search.

---

# Architecture Overview

The system has two main pipelines:

### Document Processing Pipeline

1. Documents uploaded to Amazon S3
2. AWS Step Functions orchestrates workflow
3. AWS Lambda triggers document processing
4. Amazon Textract extracts text from PDFs
5. Text is chunked into semantic segments
6. Amazon Bedrock generates embeddings
7. Vectors stored in Amazon OpenSearch

---

### Query Processing Pipeline

1. User sends query from OpenWebUI
2. Request goes through API Gateway
3. Lambda performs hybrid search in OpenSearch
4. Relevant chunks retrieved
5. Context sent to LLM via Ollama
6. Response returned to user

---

# System Architecture

User → OpenWebUI → API Gateway → Lambda → OpenSearch → Ollama LLM → Response

Document Upload → S3 → Step Functions → Lambda → Textract → Bedrock Embeddings → OpenSearch

---

# Technologies Used

AWS Services

- Amazon S3
- AWS Lambda
- AWS Step Functions
- Amazon Textract
- Amazon Bedrock
- Amazon OpenSearch
- Amazon API Gateway
- Amazon ElastiCache
- Amazon CloudWatch

AI Stack

- Ollama
- DeepSeek-R1
- Titan Embeddings
- RAG Architecture

Frameworks

- Python
- Boto3
- OpenSearch Python Client

---

# Key Features

- Retrieval Augmented Generation (RAG)
- Hybrid Search (Vector + Keyword)
- Metadata Filtering
- Semantic Chunking
- Serverless Architecture
- LLM Response Caching
- Prompt Safety with Guardrails
- Multi-document reasoning

---

# Project Structure

```
rag-ai-system
│
├── lambda
│   ├── rag-document-processor
│   ├── rag-query-api
│   └── ollama-connector
│
├── architecture
│   └── architecture-diagram.png
│
├── step-functions
│   └── workflow.json
│
└── README.md
```

---

# Setup Instructions

## 1 Upload Documents

Upload PDFs to Amazon S3 bucket.

## 2 Document Processing

Step Functions triggers Lambda which:

- Extracts text using Textract
- Creates chunks
- Generates embeddings using Bedrock
- Stores vectors in OpenSearch

## 3 Query Execution

User asks question in OpenWebUI.

Lambda performs:

- Hybrid search in OpenSearch
- Context retrieval
- Prompt generation
- LLM response via Ollama

---

# Example Query

User Query:

```
Tell me about Abhishek Dhumal
```

AI Response:

```
Abhishek Dhumal is a Generative AI Engineer with experience in AWS,
RAG systems, LLM integration, and vector search architectures.
```

---

# Future Improvements

- Add streaming responses
- Multi-agent orchestration
- Automatic document classification
- Knowledge graph integration
- Model evaluation pipelines

---

# Author

Abhishek Dhumal  
Generative AI Engineer | AWS | RAG Systems | LLM Architecture

LinkedIn:
https://linkedin.com/in/abhishek-dhumal

---

# License

MIT License
