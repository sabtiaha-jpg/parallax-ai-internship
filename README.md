### \# Parallax AI Internship — RAG System

### 

### A Retrieval-Augmented Generation (RAG) system built during the Parallax Labs AI/ML internship.

### 

### The project retrieves relevant information from a Wikipedia-based knowledge base using ChromaDB and generates answers using an LLM through OpenRouter.

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

