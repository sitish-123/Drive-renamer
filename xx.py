import os
import re
import shutil
import tempfile
import gdown
import speech_recognition as sr
from google.oauth2 import service_account
from googleapiclient.discovery import build
from pydub import AudioSegment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def drive_service_setup():
    if not os.path.exists('credentials.json'):
        print("credentials.json not found")
        return None
    creds = service_account.Credentials.from_service_account_file(
        'credentials.json', scopes=['https://www.googleapis.com/auth/drive'])
    return build('drive', 'v3', credentials=creds)

def sheets_service_setup():
    if not os.path.exists('credentials.json'):
        print("credentials.json not found")
        return None
    creds = service_account.Credentials.from_service_account_file(
        'credentials.json', scopes=['https://www.googleapis.com/auth/spreadsheets.readonly'])
    return build('sheets', 'v4', credentials=creds)

def get_folder_id(url):
    match = re.search(r"/folders/([a-zA-Z0-9-_]+)", url)
    return match.group(1) if match else None

def get_sheet_details(url):
    sheet_id_match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    sheet_id = sheet_id_match.group(1) if sheet_id_match else None
    gid_match = re.search(r"gid=([0-9]+)", url)
    gid = gid_match.group(1) if gid_match else None
    return sheet_id, gid

def get_titles(sheet_url):
    sheet_id, gid = get_sheet_details(sheet_url)
    if not sheet_id:
        print("Invalid Sheet URL")
        return []

    service = sheets_service_setup()
    if not service:
        return []

    sheet = service.spreadsheets()
    sheet_range = "A:B"
    if gid:
        meta = sheet.get(spreadsheetId=sheet_id).execute()
        sheets = meta.get('sheets', [])
        for s in sheets:
            if s.get('properties', {}).get('sheetId') == int(gid):
                sheet_name = s.get('properties', {}).get('title')
                sheet_range = f"'{sheet_name}'!A:B"
                break

    try:
        res = sheet.values().get(spreadsheetId=sheet_id, range=sheet_range).execute()
        vals = res.get('values', [])
        if vals and vals[0][0] == "Reference" and vals[0][1] == "Topics":
            vals = vals[1:]
        return [row[1] for row in vals if len(row) > 1]
    except:
        print("Error reading sheet")
        return []

def list_videos(drive, folder_id):
    res = drive.files().list(q=f"'{folder_id}' in parents and mimeType contains 'video/'",
                             fields="files(id,name)").execute()
    return res.get('files', [])

def rename_video(drive, file_id, new_name):
    try:
        drive.files().update(fileId=file_id, body={'name': new_name}).execute()
        print(f"Renamed: {new_name}")
        return True
    except:
        print(f"Failed to rename: {new_name}")
        return False

def clean_text(text):
    text = text.lower()
    hello_idx = text.find('hello')
    if hello_idx != -1:
        text = text[:hello_idx]
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stopwords = ['a','an','the','and','or','but','if','because','as','what','when','where','how','why','which','who','whom']
    words = [w for w in text.split() if w not in stopwords]
    return ' '.join(words)

def match_title(transcribed, titles):
    processed_titles = [clean_text(t) for t in titles]
    processed_trans = clean_text(transcribed)
    if len(processed_trans.split()) < 2:
        return None
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(processed_titles + [processed_trans])
    sim = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = np.argmax(sim)
    score = sim[0][idx]
    top3 = np.argsort(sim[0])[::-1][:3]
    print("\nTop 3 matches:")
    for i, j in enumerate(top3, 1):
        print(f"{i}. {titles[j]} ({sim[0][j]:.2f})")
    return titles[idx] if score > 0.2 else None

def download_preview(file_id):
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "video.mp4")
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, video_path, quiet=False)
        audio = AudioSegment.from_file(video_path)
        preview = audio[:10000]
        audio_path = os.path.join(temp_dir, "audio.wav")
        preview.export(audio_path, format='wav')
        return audio_path, temp_dir
    except:
        shutil.rmtree(temp_dir)
        return None, None

def transcribe_audio(audio_path):
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as src:
        r.adjust_for_ambient_noise(src)
        data = r.record(src)
        try:
            return r.recognize_google(data, language="en-US")
        except:
            return ""

def process_folder(folder_url, sheet_url):
    drive = drive_service_setup()
    if not drive:
        return
    folder_id = get_folder_id(folder_url)
    if not folder_id:
        print("Invalid folder URL")
        return
    videos = list_videos(drive, folder_id)
    if not videos:
        print("No videos found")
        return
    titles = get_titles(sheet_url)
    if not titles:
        print("No titles found")
        return

    print(f"{len(titles)} titles found. Processing {len(videos)} videos...")

    for vid in videos:
        print(f"\nProcessing {vid['name']}...")
        audio_path, temp_dir = download_preview(vid['id'])
        if not audio_path:
            continue
        text = transcribe_audio(audio_path)
        print(f"Transcribed: {text}")
        best = match_title(text, titles)
        shutil.rmtree(temp_dir)
        if best:
            rename_video(drive, vid['id'], f"{best}.mp4")
        else:
            print("No match found. Select manually? (y/n)")
            if input().strip().lower() == 'y':
                for i, t in enumerate(titles, 1):
                    print(f"{i}. {t}")
                try:
                    sel = int(input())
                    if 1 <= sel <= len(titles):
                        rename_video(drive, vid['id'], f"{titles[sel-1]}.mp4")
                except:
                    print("Invalid input. Skipping.")

print("Google Drive Video Renamer")
folder_url = input("Enter Drive folder URL: ").strip()
sheet_url = input("Enter Sheet URL (include gid= if needed): ").strip()
process_folder(folder_url, sheet_url)
