# \# Parallax AI/ML Internship

# 

# \## Week 1 — Environment, Data Ingestion \& Preprocessing

# 

# This repository contains my Week 1 work for the Parallax AI/ML Engineering Internship.

# 

# The goal of Week 1 was to set up a clean Python development environment, collect a real-world text dataset, build a modular preprocessing pipeline, validate the pipeline using unit tests, and generate a cleaned dataset for later stages of the project.

# 

# \---

# 

# \## Objectives

# 

# The Week 1 objectives were:

# 

# \- Initialize a clean Git repository

# \- Create and use a Python virtual environment

# \- Configure `.gitignore`

# \- Create a locked `requirements.txt`

# \- Verify required libraries and CUDA/GPU availability

# \- Acquire a dataset containing more than 5,000 real-world text documents

# \- Implement text preprocessing functions

# \- Add unit tests

# \- Generate and validate a cleaned dataset

# 

# \---

# 

# \## Project Structure

# 

# ```text

# parallax-ai-internship/

# │

# ├── data/

# │   ├── raw/

# │   │   └── wikipedia\_raw.parquet

# │   │

# │   └── processed/

# │       └── clean\_corpus.parquet

# │

# ├── scripts/

# │   ├── verify\_env.py

# │   ├── download\_dataset.py

# │   └── process\_dataset.py

# │

# ├── src/

# │   └── preprocessing.py

# │

# ├── tests/

# │   └── test\_preprocessing.py

# │

# ├── .gitignore

# ├── requirements.txt

# └── README.md

