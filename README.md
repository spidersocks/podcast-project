# Podcasting the News — Topic, Sentiment, and Stance Analysis

SIADS 699 Capstone Project (Aug 2025)

## Overview
This repo contains Jupyter notebooks, Python scripts, and small data samples used to analyze topics, sentiment, stance, and framing across U.S. public news (NPR, PBS) and top podcasts (Q1 2025). Final project blog can be found [HERE](https://www.seanfontaine.dev/podcast-project). Full GitHub repos hosting the site's [frontend](https://github.com/spidersocks/developer-site) and [backend](https://github.com/spidersocks/podcast-project-backend) are also linked.

## Contents
.
├─ Jupyter Notebooks # modeling, analysis, data processing
├─ Python script # for stance ananysis pipeline
├─ data/ # only files under 100 MB included -- details on how to obtain the full dataset 
|          included in the data acquisition notebooks
├─ requirements.txt # Python dependencies
└─ README.md

## Quick Start
- Python (3.10+)
  - python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
  - pip install -r requirements.txt
  - jupyter lab  # open notebooks/ and run as needed

## Data Notes
Large/raw datasets (e.g., full transcripts) are not tracked here. The data/ folder includes small samples and lookups. Notebooks document where larger inputs should be placed and how such large data can be obtained.

## Citation
Fontaine, S., & Simpson, C. (2025). Podcasting the News — A Topic, Sentiment, and Stance Analysis of U.S. Podcasts and Public News Media. SIADS 699 Capstone Project.