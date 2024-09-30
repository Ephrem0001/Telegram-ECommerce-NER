# Telegram-ECommerce-NER

This repository contains a set of Python scripts designed for processing and analyzing data from Telegram messages. Each script is responsible for a specific aspect of the data pipeline, including entity labeling, data preprocessing, and model comparison.

## Scripts Overview

### 1. `ner_labeling.py`

This script is responsible for labeling named entities within the Telegram messages. It defines sets of product, location, and price entities, and provides functionality to label tokens based on these entities.

#### Key Functions:

- `label_entities(tokens)`: Labels tokens with entity tags such as O (other), B-Product, I-Product, B-Price, I-Price, B-LOC, and I-LOC.
- `create_conll_format(tokens, labels)`: Converts tokens and their corresponding labels into a format suitable for CoNLL files.
- `process_messages(df, num_messages=50)`: Processes a DataFrame containing Telegram messages, labels the tokens, and prepares the output.
- `main()`: The main entry point of the script that loads data and writes the labeled output to a file.

### 2. `pre_process.py`

This script handles the preprocessing of Telegram messages, including cleaning and labeling messages for further analysis.

#### Key Functions:

- `check_and_remove_nan(column_name)`: Checks for NaN values in a specified column and removes rows with NaN values.
- `remove_emojis(text)`: Removes emojis from the given text to ensure clean message data.
- `clean_messages()`: Cleans the message data by applying the emoji removal function and saves the cleaned data to a CSV file.
- `label_message(message)`: Labels messages with prices, locations, and children’s products using a rule-based approach.

### 3. `telegram_scrapper.py`

This script is intended for scraping data from Telegram channels. (Assuming scraping functionality will be implemented in the future or if you have existing functionality that needs to be added here.)

### 4. `model_comparison.py`

This script allows for the comparison of different models used in processing and classifying the Telegram message data. (Add specific functionalities and key functions as needed.)

## Usage Instructions

1. **Installation**:
   Ensure you have Python installed along with the required libraries. You can install the necessary libraries using pip:
   ```bash
   pip install pandas transformers
   ```
