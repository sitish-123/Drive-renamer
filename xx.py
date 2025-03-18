import os
import re
import shutil
import tempfile
import time
from difflib import SequenceMatcher

import gdown
import pandas as pd
import speech_recognition as sr
from google.oauth2 import service_account
from googleapiclient.discovery import build
from pydub import AudioSegment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def setup_drive_api():
    """Set up and authenticate Google Drive API."""
    try:
        if not os.path.exists('credentials.json'):
            print("Error: credentials.json file not found.")
            return None
        SCOPES = ['https://www.googleapis.com/auth/drive']
        credentials = service_account.Credentials.from_service_account_file(
            'credentials.json', scopes=SCOPES)
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        print(f"Error setting up Drive API: {e}")
        return None


def setup_sheets_api():
    """Set up and authenticate Google Sheets API."""
    try:
        if not os.path.exists('credentials.json'):
            print("Error: credentials.json file not found.")
            return None
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
        credentials = service_account.Credentials.from_service_account_file(
            'credentials.json', scopes=SCOPES)
        return build('sheets', 'v4', credentials=credentials)
    except Exception as e:
        print(f"Error setting up Sheets API: {e}")
        return None


def extract_folder_id(folder_url):
    """Extract folder ID from Google Drive folder URL."""
    match = re.search(r"/folders/([a-zA-Z0-9-_]+)", folder_url)
    return match.group(1) if match else None


def extract_sheet_details(sheet_url):
    """Extract Sheet ID and GID from Google Sheets URL."""
    # Updated regex to handle URLs with .xlsx in them
    sheet_id_match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not sheet_id_match:
        # Try alternate pattern for .xlsx URLs
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)/", sheet_url)

    sheet_id = sheet_id_match.group(1) if sheet_id_match else None

    # Extract GID if available
    gid_match = re.search(r"gid=([0-9]+)", sheet_url)
    gid = gid_match.group(1) if gid_match else None

    return sheet_id, gid

def fetch_titles_from_sheets(sheet_url):
    """Fetch video titles from a Google Sheet (A = Reference, B = Topics)."""
    sheet_id, gid = extract_sheet_details(sheet_url)
    if not sheet_id:
        print("Invalid Google Sheets URL.")
        return []

    sheets_service = setup_sheets_api()
    if not sheets_service:
        return []

    try:
        sheet = sheets_service.spreadsheets()

        # If we have a GID, get the sheet name first
        sheet_range = "A:B"  # Get both reference and topics columns
        if gid:
            # Get the sheet name from the GID
            sheet_metadata = sheet.get(spreadsheetId=sheet_id).execute()
            sheets = sheet_metadata.get('sheets', [])

            for s in sheets:
                if s.get('properties', {}).get('sheetId') == int(gid):
                    sheet_name = s.get('properties', {}).get('title')
                    sheet_range = f"'{sheet_name}'!A:B"
                    break

            # If we couldn't find the sheet by GID, warn the user
            if sheet_range == "A:B":
                print(f"Warning: Could not find sheet with GID {gid}. Using first sheet.")

        # Fetch the data from the specified sheet
        result = sheet.values().get(spreadsheetId=sheet_id, range=sheet_range).execute()
        values = result.get('values', [])

        # Skip header row if it exists
        if values and values[0][0] == "Reference" and values[0][1] == "Topics":
            values = values[1:]

        # Return just the Topics column (column B)
        return [row[1] for row in values if len(row) > 1]
    except Exception as e:
        print(f"Error fetching data from Sheets: {e}")
        return []


def list_files_in_folder(drive_service, folder_id):
    """List all video files inside a Google Drive folder."""
    query = f"'{folder_id}' in parents and mimeType contains 'video/'"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    return results.get('files', [])


def rename_file_in_drive(drive_service, file_id, new_name):
    """Rename a file in Google Drive."""
    try:
        file_metadata = {'name': new_name}
        drive_service.files().update(fileId=file_id, body=file_metadata).execute()
        print(f"✅ Renamed file: {new_name}")
        return True
    except Exception as e:
        print(f"❌ Error renaming file: {e}")
        return False



def preprocess_text(text):
    """Preprocess text for better matching, excluding everything from 'hello' onwards."""
    # Convert to lowercase
    text = text.lower()

    # Find the position of "hello" and trim the text to exclude everything after it
    hello_index = text.find('hello')
    if hello_index != -1:
        text = text[:hello_index]

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove common stopwords that might interfere with topic matching
    stopwords = ['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when', 'where', 'how', 'why',
                 'which', 'who', 'whom']
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]

    return ' '.join(filtered_words)


def find_best_match(transcribed_text, titles):
    """Find the best-matching title using TF-IDF and cosine similarity."""
    # Preprocess all texts
    processed_titles = [preprocess_text(title) for title in titles]
    processed_transcription = preprocess_text(transcribed_text)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Special case: if transcription is empty or too short
    if len(processed_transcription.split()) < 2:
        print("⚠️ Transcription is too short for reliable matching.")
        return None

    # Fit and transform all texts
    all_texts = processed_titles + [processed_transcription]
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError as e:
        print(f"⚠️ Vectorization error: {e}")
        return None

    # Calculate cosine similarity between transcription and each title
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Get the index of the best match
    best_match_idx = np.argmax(cosine_sim)
    best_match_score = cosine_sim[0][best_match_idx]

    # Get the matched title
    best_match = titles[best_match_idx]

    # Print top 3 matches for debugging
    indices = np.argsort(cosine_sim[0])[::-1][:3]  # Top 3 indices
    print("\n🔍 Top 3 potential matches:")
    for i, idx in enumerate(indices, 1):
        print(f"{i}. {titles[idx]} (Score: {cosine_sim[0][idx]:.2f})")

    # Return the best match if score is above threshold
    threshold = 0.2  # Lower threshold for TF-IDF cosine similarity
    return best_match if best_match_score > threshold else None


def download_audio_preview(file_id):
    """Download a short audio preview of a video file."""
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
    download_url = f"https://drive.google.com/uc?id={file_id}"

    try:
        print(f"Downloading preview of video...")
        gdown.download(download_url, temp_video_path, quiet=False)
        print("Extracting audio...")
        video_audio = AudioSegment.from_file(temp_video_path)

        # Take the first 30 seconds instead of 15 for better transcription
        preview_duration = min(10000, len(video_audio))
        preview_audio = video_audio[:preview_duration]

        temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
        preview_audio.export(temp_audio_path, format='wav')
        return temp_audio_path, temp_dir
    except Exception as e:
        print(f"❌ Error downloading video preview: {e}")
        shutil.rmtree(temp_dir)
        return None, None


def speech_to_text(audio_path):
    """Convert speech from an audio file to text."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source)
        try:
            # Use language="en-US" for clearer articulation
            return recognizer.recognize_google(audio_data, language="en-US")
        except sr.UnknownValueError:
            print("⚠️ Speech recognition could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"❌ Speech Recognition error: {e}")
            return ""


def process_folder(folder_url, sheet_url):
    """Process all videos in a Google Drive folder and rename them."""
    drive_service = setup_drive_api()
    if not drive_service:
        print("❌ Failed to connect to Google Drive API.")
        return

    folder_id = extract_folder_id(folder_url)
    if not folder_id:
        print("❌ Invalid folder URL.")
        return

    video_files = list_files_in_folder(drive_service, folder_id)
    if not video_files:
        print("❌ No videos found in the folder.")
        return

    titles = fetch_titles_from_sheets(sheet_url)
    if not titles:
        print("❌ No titles found in the Google Sheet.")
        return

    print(f"📋 Found {len(titles)} titles from the Topics column in your sheet.")
    print(f"🎬 Processing {len(video_files)} videos...")

    for video in video_files:
        file_id = video['id']
        original_name = video['name']

        print(f"\n🔍 Processing: {original_name}...")

        audio_path, temp_dir = download_audio_preview(file_id)
        if not audio_path:
            continue

        print("🎙️ Transcribing audio...")
        transcribed_text = speech_to_text(audio_path)
        print(f"📝 Transcribed: {transcribed_text}")

        best_match = find_best_match(transcribed_text, titles)

        # Clean up temporary files immediately
        shutil.rmtree(temp_dir)
        print("🧹 Temporary files cleaned up.")

        if best_match:
            new_name = f"{best_match}.mp4"
            rename_file_in_drive(drive_service, file_id, new_name)
        else:
            print(f"⚠️ No suitable match found for: {original_name}")
            print("💡 Would you like to select a title manually? (y/n)")
            manual_select = input().strip().lower()
            if manual_select == 'y':
                print("\n📋 Available titles:")
                for i, title in enumerate(titles, 1):
                    print(f"{i}. {title}")
                print("\nEnter the number of the correct title:")
                try:
                    selection = int(input().strip())
                    if 1 <= selection <= len(titles):
                        selected_title = titles[selection - 1]
                        new_name = f"{selected_title}.mp4"
                        rename_file_in_drive(drive_service, file_id, new_name)
                    else:
                        print("Invalid selection. Skipping this file.")
                except ValueError:
                    print("Invalid input. Skipping this file.")


# Run script
print("📹 Google Drive Video Renaming Tool 📹")
print(
    "This tool renames videos based on speech content matched with titles from the Topics column in your Google Sheet.")
folder_url = input("📂 Enter Google Drive folder URL: ").strip()
sheet_url = input("📄 Enter Google Sheets URL (include gid=### for specific sheet): ").strip()
process_folder(folder_url, sheet_url)