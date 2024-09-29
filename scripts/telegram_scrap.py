from telethon import TelegramClient
import csv
import os
from dotenv import load_dotenv

# Load environment variables from the .env file containing API credentials
load_dotenv('.env')
api_id = os.getenv('TG_API_ID')  # Fetch the Telegram API ID from environment variables
api_hash = os.getenv('TG_API_HASH')  # Fetch the Telegram API hash from environment variables
phone = os.getenv('phone')  # Fetch the phone number from environment variables

# Function to scrape data from a single Telegram channel
async def scrape_channel(client, channel_username, writer, media_dir):
    # Retrieve information about the channel, such as its title
    entity = await client.get_entity(channel_username)
    channel_title = entity.title  # Extract the channel's title
    # Iterate over the messages in the channel (up to 10,000 messages)
    async for message in client.iter_messages(entity, limit=10000):
        media_path = None  # Initialize media_path to None
        # Check if the message has any media, specifically a photo
        if message.media and hasattr(message.media, 'photo'):
            # Create a unique filename for the photo using the message ID
            filename = f"{channel_username}_{message.id}.jpg"
            media_path = os.path.join(media_dir, filename)  # Define the path to save the photo
            # Download the photo media to the specified directory
            await client.download_media(message.media, media_path)
        
        # Write the scraped data to the CSV file, including the media path (if any)
        writer.writerow([channel_title, channel_username, message.id, message.message, message.date, media_path])

# Initialize the Telegram client with the session name and API credentials
client = TelegramClient('scraping_session', api_id, api_hash)

# Main function to start the scraping process
async def main():
    await client.start()  # Start the Telegram client
    
    # Create a directory for media files (e.g., photos) if it doesn't exist
    media_dir = 'photos'
    os.makedirs(media_dir, exist_ok=True)

    # Open a CSV file to save the scraped data, and prepare the CSV writer
    with open('telegram_data.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header row in the CSV file
        writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])  # Include channel title in the header
        
        # List of channels to scrape
        channels = [
            '@sinayelj'  # Add channel username to scrape data from
        ]
        
        # Iterate over each channel in the list and scrape its data into the CSV
        for channel in channels:
            await scrape_channel(client, channel, writer, media_dir)
            print(f"Scraped data from {channel}")  # Log the progress to the console

# Run the main function within the client context
with client:
    client.loop.run_until_complete(main())
