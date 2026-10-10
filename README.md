### \# Parallax AI Internship — RAG System

### 

### A Retrieval-Augmented Generation (RAG) system built during the Parallax Labs AI/ML internship.

### 

### The project retrieves relevant information from a Wikipedia-based knowledge base using ChromaDB and generates answers using an LLM through OpenRouter. It also includes NLP-based topic modeling, named entity recognition, and topic-filtered retrieval.

### 

### \---

### 

### \# Week 3 — LLM Integration \& Prompt Engineering

### 

### \## Overview

### 

### Week 3 extends the Week 2 retrieval system by adding:

### 

### \- LLM integration through OpenRouter

### \- Context injection

### \- System prompt engineering

### \- Hallucination protection

### \- Out-of-domain query handling

### \- API error handling

### \- Token-limit handling

### \- Retrieval and generation latency measurement

### \- Automated tests

### 

### The goal is to create a more reliable RAG pipeline that answers questions using retrieved information instead of relying on unsupported information.

### 

### \---

### 

### \# RAG Architecture

### 

### The system follows this pipeline:

### 

### ```text

### User Question

### &#x20;     |

### &#x20;     v

### Query Embedding

### &#x20;     |

### &#x20;     v

### ChromaDB Retrieval

### &#x20;     |

### &#x20;     v

### Relevant Context

### &#x20;     |

### &#x20;     v

### Relevance Check

### &#x20;     |

### &#x20;     +---- Not Relevant ----> Safe Refusal

### &#x20;     |

### &#x20;     v

### Context Injection

### &#x20;     |

### &#x20;     v

### OpenRouter LLM

### &#x20;     |

### &#x20;     v

### Generated Answer

### &#x20;     |

### &#x20;     v

### Semantic Hallucination Check

### &#x20;     |

### &#x20;     v

### Final Answer

### ```

### 

### \---

### 

### \# Week 4 — NLP Enrichment \& Topic-Filtered Retrieval

### 

### \## Overview

### 

### Week 4 extends the RAG system by integrating Natural Language Processing (NLP) metadata into the existing vector database.

### 

### The main additions include:

### 

### \- Topic modeling using Latent Dirichlet Allocation (LDA)

### \- Named Entity Recognition (NER) using spaCy

### \- NLP metadata integration into ChromaDB

### \- Preliminary NER evaluation using manually annotated text chunks

### \- Topic-filtered semantic retrieval

### \- Retrieval testing and latency measurement

### 

### These additions help organize the knowledge base and allow searches to be restricted to a selected topic.

### 

### \## 1. Topic Modeling

### 

### Latent Dirichlet Allocation (LDA), implemented using scikit-learn, was used to discover 10 topics from the Wikipedia text chunks.

### 

### The top keywords for each topic were extracted to provide an initial description of its content. Topic IDs and topic names were added to the ChromaDB metadata.

### 

### Examples of discovered topics include:

### 

### | Topic ID | Example keywords |

### |---|---|

### | 0 | film, work, art, book |

### | 1 | Apollo, arsenic, water, acid |

### | 3 | city, Azerbaijan, oil, Armenia |

### | 6 | American, English, player, French |

### | 8 | languages, court, Arabic, Greek |

### | 9 | war, military, army, British |

### 

### The topic descriptions are provisional and based on their top keywords. Some topics may contain overlapping or unrelated subjects.

### 

### \### Output files

### 

### \- `data/nlp/lda\_model.pkl` — saved LDA model.

### \- `data/nlp/topic\_keywords.json` — keywords associated with the discovered topics.

### 

### \## 2. Named Entity Recognition (NER)

### 

### Named Entity Recognition was implemented using spaCy and the `en\_core\_web\_sm` English language model.

### 

### The system extracts entities such as people, locations, organizations, dates, and other entity types recognized by the model.

### 

### Extracted entity information was stored in the NLP-enriched dataset and added to the existing ChromaDB metadata.

### 

### The metadata includes:

### 

### \- `topic\_id`

### \- `topic\_name`

### \- `named\_entities`

### \- `entity\_texts`

### 

### \### Preliminary evaluation

### 

### A manually annotated sample of five text chunks was used to calculate entity-level precision, recall, and F1-score.

### 

### | Metric | Score |

### |---|---:|

### | Chunks evaluated | 5 |

### | Precision | 47.06% |

### | Recall | 53.33% |

### | F1-score | 50.00% |

### 

### These results are preliminary and should not be treated as representative of overall NER performance because the evaluation sample is small. The manual annotations should also be reviewed for consistency, and the sample should be expanded for a more reliable evaluation.

### 

### \### Output files

### 

### \- `data/nlp/ner\_evaluation\_reviewed.csv` — evaluation sample with manual annotations.

### \- `data/nlp/ner\_evaluation\_metrics.json` — calculated evaluation metrics.

### 

### \## 3. ChromaDB Integration

### 

### NLP metadata was added to the existing ChromaDB collection, `wikipedia\_chunks`.

### 

### The collection contains 20,000 text chunks. Existing embeddings and metadata were preserved during the enrichment process.

### 

### This integration allows retrieved chunks to include topic information and named entities alongside their text.

### 

### \### Output file

### 

### \- `data/nlp/nlp\_enriched\_chunks.parquet` — NLP-enriched dataset.

### 

### \## 4. Topic-Filtered Retrieval

### 

### The script `scripts/search\_filtered.py` supports semantic search with an optional topic ID filter.

### 

### When a topic ID is provided, the query results are restricted to chunks assigned to that topic. When no topic ID is provided, the script can search across topics.

### 

### \### Test 1: Arabic language

### 

### \- \*\*Query:\*\* `languages and Arabic`

### \- \*\*Topic filter:\*\* 8

### \- \*\*Results returned:\*\* 5

### \- \*\*Topic IDs in results:\*\* All five results had topic ID 8.

### \- \*\*Search latency:\*\* 316.47 ms

### \- \*\*Top result distance:\*\* 0.5169

### 

### The retrieved results included relevant information about the Arabic language, its history, and its influence on other languages.

### 

### \### Test 2: Military history

### 

### \- \*\*Query:\*\* `military history and wars`

### \- \*\*Topic filter:\*\* 9

### \- \*\*Results returned:\*\* 5

### \- \*\*Topic IDs in results:\*\* All five results had topic ID 9.

### \- \*\*Search latency:\*\* 378.39 ms

### 

### The retrieved results included material about the American Civil War and military history.

### 

### Both tests provide qualitative evidence that topic-filtered retrieval works for the tested queries. However, two test queries are not sufficient to establish overall retrieval accuracy.

### 

### Retrieval distances represent distances in the embedding space; they are not accuracy percentages.

### 

### \## 5. Project Structure

### 

### The main files and directories used in the project include:

### 

### ```text

### parallax-ai-internship/

### ├── data/

### │   ├── chroma\_db/

### │   └── nlp/

### │       ├── lda\_model.pkl

### │       ├── topic\_keywords.json

### │       ├── nlp\_enriched\_chunks.parquet

### │       ├── ner\_evaluation\_reviewed.csv

### │       └── ner\_evaluation\_metrics.json

### ├── scripts/

### │   ├── rag\_pipeline.py

### │   ├── search\_filtered.py

### │   ├── enrich\_nlp.py

### │   ├── evaluate\_ner.py

### │   ├── benchmark\_retrieval.py

### │   ├── create\_chunks.py

### │   ├── download\_dataset.py

### │   ├── generate\_embeddings.py

### │   ├── ingest\_chroma.py

### │   ├── process\_dataset.py

### │   ├── search\_chroma.py

### │   └── verify\_env.py

### ├── src/

### │   ├── chunking/

### │   └── preprocessing/

### ├── .env

### ├── README.md

### └── requirements.txt

### ```

### 

### This structure illustrates the main components; some generated files and local environment files may not be committed to Git.

### 

### \## 6. Setup and Execution

### 

### \### Prerequisites

### 

### \- Python 3.11

### \- Git

### \- An OpenRouter API key for LLM generation

### \- The project's Python dependencies

### 

### \### Activate the virtual environment

### 

### On Windows PowerShell:

### 

### ```powershell

### .\\.venv\\Scripts\\Activate.ps1

### ```

### 

### \### Install dependencies

### 

### ```powershell

### pip install -r requirements.txt

### ```

### 

### Configure the required OpenRouter environment variables in `.env` according to the project's configuration. Never commit API keys or other secrets to GitHub.

### 

### \### Run the RAG pipeline

### 

### ```powershell

### python scripts\\rag\_pipeline.py

### ```

### 

### \### Run topic-filtered retrieval

### 

### ```powershell

### python scripts\\search\_filtered.py

### ```

### 

### Enter a query when prompted. Then enter a topic ID to restrict results to a topic, or press Enter to search without a topic filter.

### 

### \### Run NER evaluation

### 

### ```powershell

### python scripts\\evaluate\_ner.py

### ```

### 

### The script calculates precision, recall, and F1-score from the available manually annotated evaluation sample.

### 

### \### Run NLP enrichment

### 

### ```powershell

### python scripts\\enrich\_nlp.py

### ```

### 

### \*\*Warning:\*\* This script performs NLP processing and updates the ChromaDB metadata. It may take several minutes. Do not rerun it unnecessarily. Avoid running the original ingestion script unless you intentionally want to rebuild the vector database.

### 

### \---

### 

### \# Limitations and Future Improvements

### 

### The current implementation has several limitations:

### 

### \- The NER evaluation uses only five manually annotated chunks.

### \- Manual entity annotations require further review.

### \- LDA topics are automatically generated and may not always correspond to a single coherent subject.

### \- Topic-filtered retrieval has been tested on a limited number of queries.

### \- Retrieval quality should be evaluated using a larger set of queries and manually assessed relevant documents.

### \- NER performance could be improved through additional evaluation and model comparison.

### 

### Future improvements include expanding the manual evaluation dataset, refining topic descriptions, comparing filtered and unfiltered retrieval, and measuring retrieval quality across more topics.

### 

### \---

### 

### \# Project Summary

### 

### The project has developed from a basic Wikipedia-based retrieval system into a RAG pipeline with LLM integration and NLP-based metadata enrichment.

### 

### The implemented components include:

### 

### \- Wikipedia text preprocessing and chunking

### \- Text embeddings using `all-MiniLM-L6-v2`

### \- Vector storage and semantic retrieval using ChromaDB

### \- LLM integration through OpenRouter

### \- Context injection and prompt engineering

### \- Relevance checks and safe refusal handling

### \- Retrieval and generation latency measurement

### \- LDA-based topic modeling

### \- spaCy-based named entity recognition

### \- NLP metadata integration into ChromaDB

### \- Preliminary NER evaluation

### \- Topic-filtered semantic retrieval

### 

### The Week 4 tests demonstrate that the system can retrieve text chunks under selected topic filters. Further evaluation is needed to measure general retrieval quality and NER performance reliably.



