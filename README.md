# Telegram-ECommerce-NER

## Overview

This project aims to develop a Named Entity Recognition (NER) system for EthioMart, a centralized e-commerce platform in Ethiopia. The system will extract key business entities such as product names, prices, and locations from Amharic text, images, and documents shared across multiple Telegram channels.


## Business Need

EthioMart's vision is to become the primary hub for all Telegram-based e-commerce activities in Ethiopia. By consolidating real-time data from multiple e-commerce Telegram channels, EthioMart aims to provide a seamless experience for customers to explore and interact with multiple vendors in one place.

## Key Objectives

1. Real-time data extraction from Telegram channels
2. Fine-tuning Large Language Models (LLMs) for Amharic Named Entity Recognition
3. Extraction of entities such as Product names, Prices, and Locations

## Project Structure

The project is divided into the following main tasks:

1. Data Ingestion and Preprocessing
2. Data Labeling in CoNLL Format
3. Fine-tuning NER Models
4. Model Comparison and Selection
5. Model Interpretability


## Folder Structure

```plaintext

Telegram-ECommerce-NER/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml   # GitHub Actions
├── .gitignore              # files and folders to be ignored by git
├── requirements.txt        # contains dependencies for the project
├── README.md               # Documentation for the projects
├── src/
│   └── __init__.py
├── notebooks/
│   ├── __init__.py
│   └── README.md                       # Description of notebooks directory 
├── tests/
│   └── __init__.py
└── scripts/
    ├── __init__.py
    └── README.md                   # Description of scripts directory
    
```


## Installation and Setup
1. Clone this repository:
   ```
   git clone https://github.com/Alpha-Mintamir/Telegram-ECommerce-NER.git
   cd Telegram-ECommerce-NER
   ```
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
## Data

- Source: Messages and data from Ethiopian-based e-commerce Telegram channel([SINA KIDS/ሲና ኪድስⓇ](https://t.me/sinayelj))
- scraped data: [SINA KIDS](link-to-sample-data)
- labeled NER dataset: [SINA KIDS](link-to-dataset)



## Acknowledgments

- 10 Academy for providing the project requirements and guidance