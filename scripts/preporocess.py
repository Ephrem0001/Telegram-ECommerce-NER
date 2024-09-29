import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class TelegramDataProcessor:
    def __init__(self, file_path):
        # Initialize the class with the file path, load the CSV data, and prepare placeholders for the model and tokenizer
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.model = None
        self.tokenizer = None
        # Define keyword categories for price and location identification
        self.categories = {
            'price': ['ብር', 'ETB', '$', 'Birr'],
            'location': ['ገርጂ', '4ኪሎ', 'ብስራተ ገብርኤል'],
        }

    def check_and_remove_nan(self, column_name):
        # Check for NaN values in a specified column, remove rows with NaN, and print relevant information
        print(f"Checking for NaN values in the '{column_name}' column:")
        nan_count = self.df[column_name].isnull().sum()
        print(f"Number of NaN values in '{column_name}' column: {nan_count}")
        self.df = self.df.dropna(subset=[column_name])
        print(f"Dataset shape after dropping NaN values in '{column_name}' column: {self.df.shape}")

    def remove_emojis(self, text):
        """Removes emojis from the provided text."""
        # Regular expression pattern to match emoji characters, including '1️⃣'
        emoji_pattern = re.compile(
            "[" 
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
            "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
            "\U0001F700-\U0001F77F"  # Alchemical Symbols
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\u0031\uFE0F\u20E3"     # Combining 1️⃣ Emoji
            "]+", 
            flags=re.UNICODE
        )
        # Remove emojis from the text
        return emoji_pattern.sub(r'', text)

    def clean_messages(self):
        # Apply the emoji removal to the 'Message' column and save the cleaned data to a new CSV file
        self.df['Message'] = self.df['Message'].apply(self.remove_emojis)
        self.df.to_csv('../data/clean_data.csv', index=False)
        print("Cleaned data saved.")

    def label_message(self, message):
        """Labels messages with prices and locations using a rule-based approach."""
        # Multi-word entities that need special handling (e.g., locations with multiple words)
        multi_word_entities = {
            'ብስራተ ገብርኤል': 'I-LOC',
        }
        # Replace multi-word entities in the message with underscored versions
        for entity, label in multi_word_entities.items():
            if entity in message:
                message = message.replace(entity, f"{entity.replace(' ', '_')}")  # Replace spaces with underscores

        # Tokenize the message after replacing multi-word entities
        tokens = re.findall(r'\S+', message)
        labeled_tokens = []

        # Iterate through tokens and assign labels based on the token content (e.g., location, price, or phone number)
        for token in tokens:
            token = token.replace('_', ' ')  # Replace underscores back to spaces for multi-word entities

            # Check if token matches a multi-word entity
            if token in multi_word_entities:
                labeled_tokens.append(f"{token} {multi_word_entities[token]}")
            # Label as location if token matches certain keywords
            elif any(loc in token for loc in ['ገርጂ', '4ኪሎ']):
                labeled_tokens.append(f"{token} I-LOC")
            # Label as a phone number if the token is a number with 10-15 digits
            elif re.match(r'^\+?\d{10,15}$', token):
                labeled_tokens.append(f"{token} O")
            # Label as price if token matches a numeric pattern and isn't too long
            elif re.match(r'^\d+(\.\d{1,2})?$', token) and len(token) < 9:
                labeled_tokens.append(f"{token} I-PRICE")
            elif 'ብር' in token or 'Birr' in token or 'ETB' in token:
                labeled_tokens.append(f"{token} I-PRICE")
            # Label as outside any entity if no other conditions are met
            else:
                labeled_tokens.append(f"{token} O")

        # Return the labeled tokens joined with newline characters
        return "\n".join(labeled_tokens)

    def apply_labeling(self):
        # Apply the message labeling function and save the results to a text file
        self.df['Labeled_Message'] = self.df['Message'].apply(self.label_message)
        labeled_data_path = '../data/labeled_telegram_data.txt'
        with open(labeled_data_path, 'w', encoding='utf-8') as f:
            for _, row in self.df.iterrows():
                f.write(f"{row['Labeled_Message']}\n\n")
        print(f"Labeled data saved to {labeled_data_path}")

    def load_ner_model(self):
        # Load the pre-trained NER model and tokenizer from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
        self.model = AutoModelForTokenClassification.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def apply_ner(self):
        # Apply the pre-trained NER model to the 'Message' column
        if self.model is None or self.tokenizer is None:
            self.load_ner_model()
        # Perform named entity recognition on the messages
        ner_results = self.nlp(self.df['Message'].tolist())
        print(ner_results)

    def is_amharic(self, message):
        # Check if the message contains Amharic characters
        return bool(re.search(r'[\u1200-\u137F]', message))

    def classify_message(self, message):
        # Classify messages based on predefined categories (price, location, etc.)
        if pd.isna(message):
            return 'uncategorized'

        if self.is_amharic(message):
            # Check if the message contains Amharic keywords for each category
            for category, keywords in self.categories.items():
                if any(keyword in message for keyword in keywords):
                    return category
        else:
            # Check if the message contains non-Amharic keywords for each category
            for category, keywords in self.categories.items():
                if any(keyword in message.lower() for keyword in keywords):
                    return category
        # Return 'uncategorized' if no match is found
        return 'uncategorized'

    def apply_classification(self):
        # Apply the message classification function and print the results
        self.df['Category'] = self.df['Message'].apply(self.classify_message)
        print(self.df[['Message', 'Category']])

    def save_classified_data(self):
        # Display category counts and save uncategorized messages to a CSV file
        category_counts = self.df['Category'].value_counts()
        print(category_counts)
        uncategorized_items = self.df[self.df['Category'] == 'uncategorized']
        uncategorized_items.to_csv('../data/uncategorized_data.csv', index=False)
        self.df.to_csv('../data/labeled_data.csv', index=False)
        print("Labeled and uncategorized data saved.")
