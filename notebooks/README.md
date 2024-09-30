# Model Comparison & Selection

## Overview
This project compares different models for Named Entity Recognition (NER) on Amharic text data from the Telegram channels @sinayelj, @qnashcom, and @leyueqa. The objective is to fine-tune multiple models, evaluate them on a validation set, and select the best-performing model for production.

### Key Features
- **Fine-tune multiple models:** Train various models on the dataset.
- **Evaluate fine-tuned models:** Assess model performance using a validation set.
- **Model comparison:** Compare models based on accuracy, speed, and robustness.
- **Model selection:** Identify the best model for deployment.

## Model Comparison Script: `qenashcom_sinayelj_leyueqa_model_comparison.ipynb`

### Steps in the Script
1. **Import necessary libraries**
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    ```

2. **Mount Google Drive** (if using Google Colab)
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3. **Load and prepare the data**
    - Load the labeled data in CoNLL format.
    - Split the dataset into training and evaluation sets.
    ```python
    file_path = '/content/qenashcom_sinayelj_leyueqa_labeled.conll'
    sentences, labels = load_conll_data(file_path)
    dataset = prepare_dataset(sentences, labels)
    ```

4. **Define models to compare**
    ```python
    models_to_compare = [
        "xlm-roberta-base",
        "distilbert-base-multilingual-cased",
        "bert-base-multilingual-cased"
    ]
    ```

5. **Fine-tune and evaluate models**
    ```python
    for model_name in models_to_compare:
        result = finetune_and_evaluate(model_name, train_dataset, eval_dataset, label_list, label2id, id2label)
        results.append(result)
    ```

6. **Visualize results**: Generate plots comparing evaluation loss and training time for each model.

7. **Select the best model**: Identify and save the model with the lowest evaluation loss.
   
8. **Test the best model on sample text** to verify its predictions.

## Tokenization Script: `tokenization.ipynb`

### Overview
This script utilizes various tokenizers to process Amharic text. It focuses on leveraging Hugging Face's tokenizers and Google’s SentencePiece to enhance text processing capabilities.

### Steps in the Script
1. **Install necessary packages**
    ```bash
    !pip install datasets sentencepiece
    ```

2. **Mount Google Drive**
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3. **Load and process text data**
    - Read text data from a file and prepare it for tokenization.
    ```python
    with open('/content/drive/My Drive/KAIM 2/WEEK-5/labeled_telegram_product_price_location.txt', 'r') as file:
        lines = file.readlines()
    ```

4. **Tokenize using Hugging Face Tokenizers**
    ```python
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    ```

5. **Align tokens with labels** to ensure correct mapping for NER tasks.

6. **Utilize SentencePiece for improved tokenization** of Amharic text.

### Custom Amharic Tokenizer
The Amharic SentencePiece Tokenizer segments Amharic text into subwords, improving the model's ability to handle rare words and morphological variations.

## Dataset
The dataset used for training the custom tokenizer can be found [here](https://huggingface.co/datasets/israel/Amharic-News-Text-classification-Dataset?row=23).
