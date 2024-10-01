import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class TelegramDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.model = None
        self.tokenizer = None
        self.categories = {
            'price': ['ብር', 'ETB', '$', 'Birr'],
            'location': ['ገርጂ', '4ኪሎ', 'ብስራተ ገብርኤል'],
            'kids': [
                'toy', 'children', 'kids', 'መጫወቻ', 'play', 'games', 'fun', 'educational', 
                'puzzle', 'doll', 'action figure', 'stuffed animal', 'arts and crafts', 
                'books', 'outdoor toys', 'building blocks', 'baby', 'toddler', 'Baby', 
                'መጫወቻዎች'
            ]
        }

    def check_and_remove_nan(self, column_name):
        print(f"Checking for NaN values in the '{column_name}' column:")
        nan_count = self.df[column_name].isnull().sum()
        print(f"Number of NaN values in '{column_name}' column: {nan_count}")
        self.df = self.df.dropna(subset=[column_name])
        print(f"Dataset shape after dropping NaN values in '{column_name}' column: {self.df.shape}")

    def remove_emojis(self, text):
        """Removes all emojis from text."""
        emoji_pattern = re.compile(
            "[" 
            "\U0001F600-\U0001F64F" 
            "\U0001F300-\U0001F5FF" 
            "\U0001F680-\U0001F6FF"  
            "\U0001F700-\U0001F77F"  
            "\U0001F1E0-\U0001F1FF"   
            "\U00002500-\U00002BEF"  
            "\U00002702-\U000027B0"  
            "\U0001F900-\U0001F9FF"  
            "\U00002600-\U000026FF"  
             "\u0031\uFE0F\u20E3"     
            "]+", 
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)

    def clean_messages(self):
        self.df['Message'] = self.df['Message'].apply(self.remove_emojis)
        self.df.to_csv('/content/drive/MyDrive/telegram_preprocess/datas/clean_data.csv', index=False)
        print("Cleaned data saved.")

    def label_message(self, message):
        # Define multi-word entities (locations, products, etc.)
        multi_word_entities = {
            'ብስራተ ገብርኤል': 'I-LOC',
        }

        # First, check for multi-word entities in the message
        for entity, label in multi_word_entities.items():
            if entity in message:
                message = message.replace(entity, f"{entity.replace(' ', '_')}")  # Replace spaces with underscores

        tokens = re.findall(r'\S+', message)  # Tokenize after replacing multi-word entities
        labeled_tokens = []

        for token in tokens:
            if token.startswith('@'):
                continue  # Skip usernames

            # After tokenizing, replace underscores with spaces again for multi-word entities
            token = token.replace('_', ' ')

            # Check if token is a multi-word entity (location, product, etc.)
            if token in multi_word_entities:
                labeled_tokens.append(f"{token} {multi_word_entities[token]}")
            # Check if token is a location (single-word locations)
            elif any(loc in token for loc in ['ገርጂ', '4ኪሎ']):
                labeled_tokens.append(f"{token} I-LOC")
            # Check if token is a phone number (exclude numbers longer than 9 digits)
            elif re.match(r'^\+?\d{10,15}$', token):
                labeled_tokens.append(f"{token} O")
            # Check if token is a price (e.g., 500 ETB, $100, or ብር)
            elif re.match(r'^\d+(\.\d{1,2})?$', token) and len(token) < 9:
                labeled_tokens.append(f"{token} I-PRICE")
            elif 'ብር' in token or 'Birr' in token or 'ETB' in token:
                labeled_tokens.append(f"{token} I-PRICE")
            # Check if token matches kids products category
            elif any(token.lower() == item.lower() for item in self.categories['kids']):
                labeled_tokens.append(f"{token} B-PRODUCT")
            # Otherwise, treat it as outside any entity
            else:
                labeled_tokens.append(f"{token} O")

        return "\n".join(labeled_tokens)

    def apply_labeling(self):
        self.df['Labeled_Message'] = self.df['Message'].apply(self.label_message)
        labeled_data_path = '/content/drive/MyDrive/telegram_preprocess/datas/labeled_telegram_data.txt'
        with open(labeled_data_path, 'w', encoding='utf-8') as f:
            for _, row in self.df.iterrows():
                f.write(f"{row['Labeled_Message']}\n\n")
        print(f"Labeled data saved to {labeled_data_path}")

    def load_ner_model(self):
        # Loads a pretrained NER model.
        self.tokenizer = AutoTokenizer.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
        self.model = AutoModelForTokenClassification.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def apply_ner(self):
        # Applies NER to the 'Message' column using a pretrained model.
        if self.model is None or self.tokenizer is None:
            self.load_ner_model()
        ner_results = self.nlp(self.df['Message'].tolist())
        print(ner_results)

    def is_amharic(self, message):
        # Checks if a string contains Amharic characters.
        return bool(re.search(r'[\u1200-\u137F]', message))

    def classify_message(self, message):
        # Classifies messages based on predefined categories.
        if pd.isna(message):
            return 'uncategorized'

        if self.is_amharic(message):
            for category, keywords in self.categories.items():
                if any(keyword in message for keyword in keywords):
                    return category
        else:
            for category, keywords in self.categories.items():
                if any(keyword in message.lower() for keyword in keywords):
                    return category
        return 'uncategorized'

    def apply_classification(self):
        # Applies the classification to the 'Message' column.
        self.df['Category'] = self.df['Message'].apply(self.classify_message)
        print(self.df[['Message', 'Category']])

    def save_classified_data(self):
        # Displays category counts and saves uncategorized messages.
        category_counts = self.df['Category'].value_counts()
        print(category_counts)
        uncategorized_items = self.df[self.df['Category'] == 'uncategorized']
        uncategorized_items.to_csv('/content/drive/MyDrive/telegram_preprocess/datas/uncategorized_data.csv', index=False)
        self.df.to_csv('/content/drive/MyDrive/telegram_preprocess/datas/labeled_data.csv', index=False)
        print("Labeled and uncategorized data saved.")
