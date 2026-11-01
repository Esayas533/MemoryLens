import cv2
import numpy as np
import os
import sys
import math
import io
import wave
import certifi
import json
import threading
import time
import random
from collections import deque
from datetime import datetime
import sounddevice as sd
from dotenv import load_dotenv
from pymongo import MongoClient
import onnxruntime
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo as insightface_model_zoo
from google import genai
from PIL import Image, ImageDraw, ImageFont

project_dir = os.path.dirname(os.path.abspath(__file__))

env_candidates = [
    os.path.join(project_dir, "audio recognition", ".env"),
    os.path.join(project_dir, ".env"),
]
env_candidates.extend(
    os.path.join(project_dir, filename)
    for filename in os.listdir(project_dir)
    if filename.endswith(".env")
)

loaded_env_path = next(
    (path for path in env_candidates if os.path.isfile(path)),
    None,
)

if loaded_env_path:
    load_dotenv(loaded_env_path)

OFFLINE = (
    "--offline" in sys.argv
    or os.getenv("MEMORYLENS_OFFLINE", "").strip().lower() in ("1", "true", "yes")
)

MONGO_URL = os.getenv("MONGO_URL")
api_key = os.getenv("GEMINI_API_KEY")

if not MONGO_URL and not OFFLINE:
    raise ValueError(
        f"MONGO_URL is missing. Checked: {', '.join(env_candidates)}"
    )

if not api_key and not OFFLINE:
    raise ValueError(
        f"GEMINI_API_KEY is missing. Checked: {', '.join(env_candidates)}"
    )

if OFFLINE:
    print("OFFLINE MODE: no database.")
    print("Anyone registered now is kept in memory only and lost on exit.")
    people = None
else:
    try:
        mongo_client = MongoClient(
            MONGO_URL,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=8000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
        )
        print(mongo_client.admin.command("ping"))
        db = mongo_client["MemoryLens"]
        people = db["people"]
    except Exception as error:
        raise RuntimeError(
            "MongoDB is unreachable. Check your internet/DNS and Atlas network "
            "access, or run with --offline for camera-only mode."
        ) from error

gemini_client = genai.Client(api_key=api_key) if api_key else None

DETECTOR_THREADS = min(8, max(1, (os.cpu_count() or 4) // 2))
_session_options = onnxruntime.SessionOptions()
_session_options.intra_op_num_threads = DETECTOR_THREADS
_session_options.inter_op_num_threads = 1
_original_session_init = insightface_model_zoo.PickableInferenceSession.__init__


def _limited_session_init(self, model_path, **kwargs):
    kwargs.setdefault("sess_options", _session_options)
    _original_session_init(self, model_path, **kwargs)


insightface_model_zoo.PickableInferenceSession.__init__ = _limited_session_init

app = FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"],
    allowed_modules=["detection", "recognition"]
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320),
    det_thresh=0.5
)

insightface_model_zoo.PickableInferenceSession.__init__ = _original_session_init

print(f"Detector using {DETECTOR_THREADS} of {os.cpu_count()} cores.")

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\nyala.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]

FONT_PATH = next(
    (path for path in FONT_CANDIDATES if os.path.isfile(path)),
    None,
)

try:
    FONT_LG = ImageFont.truetype(FONT_PATH, 34)
    FONT_SM = ImageFont.truetype(FONT_PATH, 21)
    FONT_XS = ImageFont.truetype(FONT_PATH, 18)
except:
    FONT_LG = ImageFont.load_default()
    FONT_SM = ImageFont.load_default()
    FONT_XS = ImageFont.load_default()

sample_rate = 16000
duration = 10

GEMINI_ATTEMPTS = 4
GEMINI_BACKOFF = 2.0
MAX_CONVERSATIONS = 50
MAX_TEXT_LENGTH = 2000
TRANSIENT_MARKERS = (
    "503", "500", "unavailable", "overloaded",
    "internal error", "deadline", "timeout",
)
MESSAGE_TTL = 20.0
REGISTRATION_ANGLES = ["front", "left", "right"]
REGISTRATION_MESSAGES = {
    "front": "Look straight at the camera",
    "left": "Slowly turn your face LEFT",
    "right": "Slowly turn your face RIGHT",
}
REGISTRATION_CAPTURE_DELAY = 1.5
REGISTRATION_MAX_TIME = 20.0
registration_lock = threading.Lock()
registration_state = {
    "active": False,
    "angle_index": 0,
    "embeddings": {},
    "started": 0.0,
    "angle_started": 0.0,
}


def set_status(message, error=False):
    with audio_lock:
        audio_status["message"] = message
        audio_status["error"] = error
        audio_status["stamp"] = time.time()


def is_transient(error):
    text = f"{type(error).__name__} {error}".lower()
    if "429" in text or "quota" in text or "resource_exhausted" in text:
        return False
    return any(marker in text for marker in TRANSIENT_MARKERS)


def call_gemini(label, func):
    """Run a Gemini call, retrying transient outages with backoff."""
    for attempt in range(1, GEMINI_ATTEMPTS + 1):
        try:
            return func()
        except Exception as error:
            if not is_transient(error) or attempt == GEMINI_ATTEMPTS:
                raise
            delay = GEMINI_BACKOFF ** (attempt - 1) + random.uniform(0, 0.5)
            set_status(f"{label} busy, retrying in {delay:.0f}s "
                       f"({attempt}/{GEMINI_ATTEMPTS - 1})")
            print(f"[Gemini] {label} attempt {attempt} failed, "
                  f"retrying in {delay:.1f}s: {str(error)[:120]}", flush=True)
            time.sleep(delay)

def record_audio():
    print("Recording...")
    set_status("Recording...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )
    sd.wait()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())
    buffer.seek(0)
    return buffer

def transcribe_audio(audio_buffer):
    print("Transcribing...", flush=True)
    set_status("Transcribing...")

    def upload():
        audio_buffer.seek(0)
        return gemini_client.files.upload(
            file=audio_buffer,
            config={
                "mime_type": "audio/wav"
            }
        )

    audio_file = call_gemini("Upload", upload)

    def transcribe():
        return gemini_client.interactions.create(
            model="gemini-3.5-transcribe",
            input=[
                {
                    "type": "audio",
                    "uri": audio_file.uri,
                    "mime_type": audio_file.mime_type
                }
            ],
            generation_config={
                "transcription_config": {
                    "language_codes": []
                }
            }
        )

    result = call_gemini("Transcription", transcribe)
    transcript = result.output_text.strip()
    print("TRANSCRIPT:")
    print(transcript)
    return transcript

def analyze_transcript(transcript):
    if not transcript:
        return None

    prompt = f"""
You are an information extraction system.

Do NOT talk to the user.
Do NOT explain anything.

Analyze this transcript and extract:
1. name
2. relationship
3. summary

IMPORTANT:
- Keep the SAME LANGUAGE as the transcript.
- If the transcript is Amharic, relationship and summary must be in Amharic.
- If the transcript is English, relationship and summary must be in English.
- Keep names exactly as spoken.
- Do not invent information.
- If the name is not mentioned, use "Unknown".
- If the relationship is not mentioned, use "Unknown".

Return ONLY valid JSON.
Use exactly this format:
{{
    "name": "person's name",
    "relationship": "relationship to the speaker",
    "summary": "short summary"
}}

Transcript:
{transcript}
"""

    response = call_gemini(
        "Extraction",
        lambda: gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
    )

    text = response.text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print("Could not parse Gemini JSON.")
        return None

    if not isinstance(data, dict):
        print("Gemini returned a JSON value instead of an object.")
        return None

    cleaned = {}
    for field in ("name", "relationship", "summary"):
        value = data.get(field, "Unknown")
        if not isinstance(value, str):
            value = "Unknown"
        cleaned[field] = value.strip()[:MAX_TEXT_LENGTH] or "Unknown"
    return cleaned


def clean_text(value, fallback="Unknown"):
    if not isinstance(value, str):
        return fallback
    value = value.strip()[:MAX_TEXT_LENGTH]
    return value or fallback

def get_person_information():
    audio_buffer = record_audio()
    try:
        transcript = transcribe_audio(audio_buffer)
    finally:
        audio_buffer.close()

    if not transcript:
        return None

    return analyze_transcript(transcript)

def save_conversation(person_id, transcript, summary):
    people.update_one(
        {"_id": person_id},
        {
            "$push": {
                "conversations": {
                    "$each": [{
                        "transcript": str(transcript)[:MAX_TEXT_LENGTH],
                        "summary": str(summary or "Unknown")[:MAX_TEXT_LENGTH],
                        "timestamp": datetime.now().isoformat()
                    }],
                    "$slice": -MAX_CONVERSATIONS,
                },
            }
        }
    )

def summarize_conversation(transcript):
    if not transcript:
        return None

    prompt = f"""
Summarize this conversation in one short sentence.
Do not invent information.
Keep the same language as the transcript.

Transcript:
{transcript}
"""

    response = call_gemini(
        "Summary",
        lambda: gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
    )
    return response.text.strip()

def generate_starter_question(person):
    current_person = people.find_one({"_id": person["_id"]}) if people is not None else None
    if current_person is None:
        return "How have you been?"

    conversations = current_person.get("conversations", [])
    if conversations:
        latest = conversations[-1]
        context = latest.get("summary", "")
    else:
        context = current_person.get("summary", "")

    prompt = f"""
You are helping a person remember how to start a conversation with someone they know.

Person:
Name: {current_person.get("name", "")}
Relationship: {current_person.get("relationship", "")}

Previous conversation:
{context}

Generate ONE natural conversation starter based on the previous conversation.

Rules:
- Keep it appropriate and natural.
- Do not mention that you are an AI.
- Do not invent facts.
- If there is no previous conversation, ask a simple question based on the person's stored information.
- Return ONLY the question.
- Use the same language as the previous conversation when possible.
"""

    response = call_gemini(
        "Starter",
        lambda: gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
    )
    return response.text.strip()

embedding_lock = threading.Lock()

known_people = list(people.find()) if people is not None else []
valid_people = []
embedding_dimension = None

for person in known_people:
    stored_embeddings = person.get("embeddings")
    entries = []

    if isinstance(stored_embeddings, dict):
        for angle in REGISTRATION_ANGLES:
            value = stored_embeddings.get(angle)
            if value is not None:
                entries.append((angle, value))
    else:
        value = person.get("embedding")
        if value is not None:
            entries.append(("legacy", value))

    if not entries:
        print(f"Skipping person {person.get('_id')}: no stored embedding.")
        continue

    for angle, value in entries:
        try:
            embedding = np.asarray(value, dtype=np.float32)

            if (
                embedding.ndim != 1
                or embedding.size == 0
                or not np.isfinite(embedding).all()
            ):
                raise ValueError("embedding must be a finite one-dimensional array")

            if embedding_dimension is None:
                embedding_dimension = embedding.size
            elif embedding.size != embedding_dimension:
                raise ValueError("embedding dimensions do not match")

            norm = np.linalg.norm(embedding)

            if norm == 0:
                raise ValueError("embedding has zero norm")

            entry = {
                "name": clean_text(person.get("name")),
                "relationship": clean_text(person.get("relationship")),
                "summary": clean_text(person.get("summary"), "No summary available."),
                "embedding": embedding / norm,
                "conversations": person.get("conversations", []),
                "embedding_angle": angle,
            }

            if "_id" in person:
                entry["_id"] = person["_id"]

            valid_people.append(entry)

        except (TypeError, ValueError) as error:
            print(
                f"Skipping invalid {angle} embedding for {person.get('_id')}: {error}"
            )

known_people = valid_people

if known_people:
    known_embeddings = np.array(
        [person["embedding"] for person in known_people],
        dtype=np.float32
    )
else:
    known_embeddings = np.empty(
        (0, embedding_dimension or 512),
        dtype=np.float32
    )

print(f"Loaded {len(known_people)} face embeddings.")

def match_embedding(embed):
    query_norm = np.linalg.norm(embed)
    with embedding_lock:
        people_snapshot = known_people
        matrix_snapshot = known_embeddings

    best_match = None
    best_score = -1
    if query_norm != 0 and matrix_snapshot.shape[0] > 0:
        query = embed / query_norm
        scores = matrix_snapshot @ query
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_match = people_snapshot[best_idx]
    return best_match, best_score

audio_lock = threading.Lock()
audio_status = {
    "processing": False,
    "message": "",
    "error": False,
    "stamp": 0.0
}

def process_audio_registration(embeddings):
    global known_people, known_embeddings

    try:
        if gemini_client is None:
            set_status("No GEMINI_API_KEY: cannot register.", error=True)
            return

        info = get_person_information()

        if info is None:
            set_status("Could not get information.", error=True)
            return

        name = clean_text(info.get("name"))
        relationship = clean_text(info.get("relationship"))
        summary = clean_text(info.get("summary"))

        if name == "Unknown":
            set_status(
                "Did not catch the name. Press r and say it again.",
                error=True
            )
            return

        if relationship == "Unknown":
            set_status(
                "Did not catch the relationship. Say name AND relation.",
                error=True
            )
            return

        normalized_embeddings = {}

        for angle in REGISTRATION_ANGLES:
            embed = np.asarray(embeddings.get(angle), dtype=np.float32)

            if (
                embed.ndim != 1
                or embed.size == 0
                or not np.isfinite(embed).all()
            ):
                raise ValueError(f"Invalid {angle} embedding")

            norm = np.linalg.norm(embed)

            if norm == 0:
                raise ValueError(f"Zero-length {angle} embedding")

            normalized_embeddings[angle] = embed / norm

        person = {
            "name": name,
            "relationship": relationship,
            "summary": summary,
            "embeddings": {
                angle: normalized_embeddings[angle].tolist()
                for angle in REGISTRATION_ANGLES
            },
            "conversations": []
        }

        if people is not None:
            result = people.insert_one(person)
            person["_id"] = result.inserted_id

        with embedding_lock:
            people_snapshot = list(known_people)
            matrix_snapshot = known_embeddings.copy()

        new_people = list(people_snapshot)
        new_vectors = []

        for angle in REGISTRATION_ANGLES:
            new_entry = {
                "name": name,
                "relationship": relationship,
                "summary": summary,
                "embedding": normalized_embeddings[angle].copy(),
                "conversations": [],
                "embedding_angle": angle,
            }

            if "_id" in person:
                new_entry["_id"] = person["_id"]

            new_people.append(new_entry)
            new_vectors.append(normalized_embeddings[angle])

        new_vectors = np.array(new_vectors, dtype=np.float32)

        if matrix_snapshot.shape[0] > 0:
            new_matrix = np.vstack([matrix_snapshot, new_vectors])
        else:
            new_matrix = new_vectors

        with embedding_lock:
            known_people = new_people
            known_embeddings = new_matrix

        if people is None:
            message = (
                f"{name} registered in memory with front, left and right face angles."
            )
        else:
            message = (
                f"{name} registered with front, left and right face angles."
            )

        set_status(message)
        print(f"[Audio Thread] {message}", flush=True)

    except Exception as e:
        error_text = str(e)
        lowered = error_text.lower()

        if "429" in error_text or "quota" in lowered:
            error_message = (
                "Gemini quota exceeded. Wait for the reset or use another key."
            )
        elif is_transient(e):
            error_message = (
                f"Gemini overloaded after {GEMINI_ATTEMPTS} tries. "
                "Press r to try again."
            )
        else:
            error_message = f"Audio error: {type(e).__name__}: {error_text[:70]}"

        set_status(error_message, error=True)
        print(f"[Audio Thread] {error_message}", flush=True)
        print(f"[Audio Thread] full error: {error_text}", flush=True)

    finally:
        with audio_lock:
            audio_status["processing"] = False

def process_audio_update(person_entry, embed):
    try:
        if gemini_client is None:
            set_status("No GEMINI_API_KEY: cannot update.", error=True)
            return

        audio_buffer = record_audio()
        try:
            transcript = transcribe_audio(audio_buffer)
        finally:
            audio_buffer.close()

        if not transcript:
            set_status("No conversation detected.", error=True)
            return

        conversation_summary = summarize_conversation(transcript)

        if people is not None and "_id" in person_entry:
            save_conversation(
                person_entry["_id"],
                transcript,
                conversation_summary
            )

        with embedding_lock:
            person_entry["summary"] = conversation_summary
            person_entry.setdefault("conversations", [])
            person_entry["conversations"].append({
                "transcript": str(transcript)[:MAX_TEXT_LENGTH],
                "summary": str(conversation_summary or "Unknown")[:MAX_TEXT_LENGTH],
                "timestamp": datetime.now().isoformat()
            })
            del person_entry["conversations"][:-MAX_CONVERSATIONS]

        message = f"Conversation updated. New summary: {conversation_summary}"
        set_status(message)
        print("Conversation updated.", flush=True)
        print(f"New summary: {conversation_summary}", flush=True)
    except Exception as e:
        error_text = str(e)
        lowered = error_text.lower()
        if "429" in error_text or "quota" in lowered:
            error_message = (
                "Gemini quota exceeded. Wait for the reset or use another key."
            )
        elif is_transient(e):
            error_message = (
                f"Gemini overloaded after {GEMINI_ATTEMPTS} tries. "
                "Press u to try again."
            )
        else:
            error_message = f"Audio error: {type(e).__name__}: {error_text[:70]}"

        set_status(error_message, error=True)
        print(f"[Audio Thread] {error_message}", flush=True)
        print(f"[Audio Thread] full error: {error_text}", flush=True)
    finally:
        with audio_lock:
            audio_status["processing"] = False

def start_audio_thread(embeddings):
    with audio_lock:
        if audio_status["processing"]:
            return False

        audio_status["processing"] = True
        audio_status["message"] = "Listening..."
        audio_status["error"] = False
        audio_status["stamp"] = time.time()

    embeddings_copy = {
        angle: np.asarray(embeddings[angle], dtype=np.float32).copy()
        for angle in REGISTRATION_ANGLES
    }

    thread = threading.Thread(
        target=process_audio_registration,
        args=(embeddings_copy,),
        daemon=True
    )
    thread.start()
    return True


def start_three_angle_registration():
    with registration_lock:
        if registration_state["active"]:
            return False

        registration_state["active"] = True
        registration_state["angle_index"] = 0
        registration_state["embeddings"] = {}
        registration_state["started"] = time.time()
        registration_state["angle_started"] = time.time()

    angle = REGISTRATION_ANGLES[0]
    set_status(f"{angle.upper()}: {REGISTRATION_MESSAGES[angle]}")
    print(f"{angle.upper()}: {REGISTRATION_MESSAGES[angle]}", flush=True)
    return True


def cancel_three_angle_registration(
    message="3-angle registration cancelled.",
    error=True
):
    with registration_lock:
        registration_state["active"] = False
        registration_state["angle_index"] = 0
        registration_state["embeddings"] = {}
        registration_state["started"] = 0.0
        registration_state["angle_started"] = 0.0

    set_status(message, error=error)


def registration_capture(track):
    start_audio = False
    embeddings = None
    next_message = None

    with registration_lock:
        if not registration_state["active"]:
            return

        angle_index = registration_state["angle_index"]

        if angle_index >= len(REGISTRATION_ANGLES):
            return

        angle = REGISTRATION_ANGLES[angle_index]
        elapsed = time.time() - registration_state["angle_started"]

        if elapsed < REGISTRATION_CAPTURE_DELAY:
            return

        embed = np.asarray(track["embedding"], dtype=np.float32).copy()
        norm = np.linalg.norm(embed)

        if norm == 0 or not np.isfinite(embed).all():
            return

        embed = embed / norm
        registration_state["embeddings"][angle] = embed

        print(f"Captured {angle} face embedding.", flush=True)

        next_index = angle_index + 1

        if next_index >= len(REGISTRATION_ANGLES):
            embeddings = {
                key: value.copy()
                for key, value in registration_state["embeddings"].items()
            }

            registration_state["active"] = False
            registration_state["angle_index"] = 0
            registration_state["embeddings"] = {}
            registration_state["started"] = 0.0
            registration_state["angle_started"] = 0.0
            start_audio = True
        else:
            registration_state["angle_index"] = next_index
            registration_state["angle_started"] = time.time()

            next_angle = REGISTRATION_ANGLES[next_index]
            next_message = (
                f"{next_angle.upper()}: "
                f"{REGISTRATION_MESSAGES[next_angle]}"
            )

    if start_audio:
        set_status(
            "3 angles captured. Now say the person's name and relationship."
        )
        print("All 3 face angles captured.", flush=True)
        print("Now say the person's name and relationship.", flush=True)
        start_audio_thread(embeddings)
    elif next_message:
        set_status(next_message)
        print(next_message, flush=True)

def start_update_thread(person_entry, embed):
    with audio_lock:
        if audio_status["processing"]:
            return False
        audio_status["processing"] = True
        audio_status["message"] = "Listening..."
        audio_status["error"] = False
        audio_status["stamp"] = time.time()

    thread = threading.Thread(
        target=process_audio_update,
        args=(person_entry, embed.copy()),
        daemon=True
    )
    thread.start()
    return True


def process_starter_question(person_entry):
    try:
        if gemini_client is None:
            set_status("No GEMINI_API_KEY: cannot generate a starter.", error=True)
            return
        question = generate_starter_question(person_entry)
        if not question:
            set_status("Could not generate a starter question.", error=True)
            return
        question = clean_text(question, "How have you been?")
        print("\nSTARTER QUESTION:")
        print(question)
        set_status(f"Starter: {question}")
    except Exception as error:
        error_text = str(error)
        set_status(
            f"Starter error: {type(error).__name__}: {error_text[:70]}",
            error=True,
        )
        print(f"[Starter Thread] full error: {error_text}", flush=True)
    finally:
        with audio_lock:
            audio_status["processing"] = False


def start_starter_thread(person_entry):
    with audio_lock:
        if audio_status["processing"]:
            return False
        audio_status["processing"] = True
        audio_status["message"] = "Generating starter..."
        audio_status["error"] = False
        audio_status["stamp"] = time.time()

    thread = threading.Thread(
        target=process_starter_question,
        args=(person_entry,),
        daemon=True,
    )
    thread.start()
    return True


THRESHOLD = 0.45           # cosine similarity needed to call it a match
BOX_SMOOTHING_TAU = 0.08   # seconds; lower = snappier boxes, higher = smoother
TRACK_IOU_MATCH = 0.3      # min overlap to consider it the same face
TRACK_TTL = 0.7            # seconds a face stays on screen without a detection
IDENTITY_HITS = 2          # consecutive agreeing matches before the label flips
CARD_W, CARD_H = 460, 170
CARD_RADIUS = 22
COMPACT_W, COMPACT_H = 320, 108
COMPACT_RADIUS = 16


class CameraStream:
    """Grabs frames in the background so read() never blocks on the driver."""
    def __init__(self, index=0, width=640, height=480):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            raise RuntimeError("Could not open webcam.")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture = capture
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            success, frame = self.capture.read()
            if not success:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def read(self):
        with self.lock:
            return self.frame, self.frame_id

    def stop(self):
        self.running = False
        self.capture.release()
        self.thread.join(timeout=1.0)
class Detector:
    """Runs face detection + embedding on the newest frame, off the UI thread."""
    def __init__(self, stream):
        self.stream = stream
        self.lock = threading.Lock()
        self.faces = []
        self.stamp = 0.0
        self.fps = 0.0
        self.latency_ms = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    def _loop(self):
        seen_frame_id = -1
        finish_times = deque(maxlen=20)
        while self.running:
            frame, frame_id = self.stream.read()
            if frame is None or frame_id == seen_frame_id:
                time.sleep(0.002)
                continue
            seen_frame_id = frame_id
            started = time.perf_counter()
            try:
                faces = app.get(frame)
            except Exception as error:
                print(f"[Detector] {error}")
                continue
            finished = time.perf_counter()
            finish_times.append(finished)
            span = finish_times[-1] - finish_times[0]
            with self.lock:
                self.faces = faces
                self.stamp = finished
                self.latency_ms = (finished - started) * 1000.0
                if len(finish_times) > 1 and span > 0:
                    self.fps = (len(finish_times) - 1) / span
    def latest(self):
        with self.lock:
            return self.faces, self.stamp, self.fps, self.latency_ms
    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)
def bbox_iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
def identify(embed):
    best_match, best_score = match_embedding(embed)
    if best_match is not None and best_score >= THRESHOLD:
        return (
            best_match["name"],
            best_match["relationship"],
            best_match.get("summary", "No summary available."),
            True,
        )
    return (
        "Unknown",
        "Stranger",
        "Face not recognized in the database.",
        False,
    )
def update_tracks(tracks, faces, now):
    """Attach each detection to an existing track (IoU) or start a new one."""
    claimed = set()
    for face in faces:
        bbox = face.bbox.astype(np.float32)
        embed = np.asarray(face.embedding, dtype=np.float32)
        identity = identify(embed)
        best_track = None
        best_score = TRACK_IOU_MATCH
        for track in tracks:
            if id(track) in claimed:
                continue
            score = bbox_iou(bbox, track["target"])
            if score >= best_score:
                best_score = score
                best_track = track
        if best_track is None:
            best_track = {
                "target": bbox,
                "display": bbox.copy(),
                "identity": identity,
                "pending": identity,
                "pending_hits": 0,
                "embedding": embed,
                "last_seen": now,
            }
            tracks.append(best_track)
        else:
            best_track["target"] = bbox
            best_track["embedding"] = embed
            best_track["last_seen"] = now
            if identity == best_track["identity"]:
                best_track["pending_hits"] = 0
            elif identity == best_track["pending"]:
                best_track["pending_hits"] += 1
                if best_track["pending_hits"] >= IDENTITY_HITS:
                    best_track["identity"] = identity
                    best_track["pending_hits"] = 0
            else:
                best_track["pending"] = identity
                best_track["pending_hits"] = 1
        claimed.add(id(best_track))
    return [track for track in tracks if now - track["last_seen"] <= TRACK_TTL]
def wrap_text(draw, text, font, max_width, max_lines):
    lines = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if not current or draw.textlength(candidate, font=font) <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) == max_lines:
            current = ""
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if lines and len(" ".join(lines)) < len(text):
        lines[-1] = lines[-1] + "..."
    return lines
CARD_CACHE = {}
NAME_CACHE = {}
COMPACT_CARD_CACHE = {}
LEGEND_CACHE = {}
def build_legend():
    """Render the R / U / Q keyboard shortcut legend once, then reuse it."""
    cached = LEGEND_CACHE.get("panel")
    if cached is not None:
        return cached
    rows = [
        ("R", "Register", (0, 200, 90, 255)),
        ("U", "Update", (0, 165, 255, 255)),
        ("Q", "Exit", (60, 60, 220, 255)),
    ]
    row_h = 40
    width = 190
    height = row_h * len(rows) + 20
    radius = 16
    panel = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle(
        [(0, 0), (width - 1, height - 1)],
        radius=radius,
        fill=(18, 19, 24, 210)
    )
    draw.rounded_rectangle(
        [(0, 0), (width - 1, height - 1)],
        radius=radius,
        outline=(255, 255, 255, 40),
        width=1
    )
    for index, (key, label, accent) in enumerate(rows):
        row_y = 10 + index * row_h
        chip_size = 28
        chip_x = 12
        chip_y = row_y + (row_h - chip_size) // 2
        draw.rounded_rectangle(
            [(chip_x, chip_y), (chip_x + chip_size, chip_y + chip_size)],
            radius=8,
            fill=accent
        )
        key_bbox = draw.textbbox((0, 0), key, font=FONT_SM)
        key_w = key_bbox[2] - key_bbox[0]
        key_h = key_bbox[3] - key_bbox[1]
        draw.text(
            (chip_x + (chip_size - key_w) / 2, chip_y + (chip_size - key_h) / 2 - key_bbox[1]),
            key,
            font=FONT_SM,
            fill=(255, 255, 255, 255)
        )
        label_bbox = draw.textbbox((0, 0), label, font=FONT_XS)
        label_h = label_bbox[3] - label_bbox[1]
        draw.text(
            (chip_x + chip_size + 12, row_y + (row_h - label_h) / 2 - label_bbox[1]),
            label,
            font=FONT_XS,
            fill=(220, 220, 225, 255)
        )
    rgba = np.array(panel, dtype=np.float32)
    alpha = rgba[:, :, 3:4] / 255.0
    premultiplied = rgba[:, :, 2::-1] * alpha
    inverse_alpha = 1.0 - alpha
    LEGEND_CACHE["panel"] = (premultiplied, inverse_alpha, width, height)
    return LEGEND_CACHE["panel"]
def paste_legend(frame, x, y):
    """Blend the keyboard shortcut legend onto the frame at (x, y)."""
    premultiplied, inverse_alpha, width, height = build_legend()
    region = frame[y:y + height, x:x + width]
    region[:] = (premultiplied + region.astype(np.float32) * inverse_alpha).astype(np.uint8)
    return x, y, width, height
def build_card(identity, box_color):
    """Render the full information card once per identity, then reuse it."""
    key = (identity, box_color)
    cached = CARD_CACHE.get(key)
    if cached is not None:
        return cached
    name, relationship, summary, _known = identity
    accent = (box_color[2], box_color[1], box_color[0], 255)
    card = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle(
        [(0, 0), (CARD_W - 1, CARD_H - 1)],
        radius=CARD_RADIUS,
        fill=(18, 19, 24, 220)
    )
    gradient = Image.new("L", (1, CARD_H), 0)
    for row in range(CARD_H):
        gradient.putpixel((0, row), int(50 * (1 - row / CARD_H)))
    highlight = Image.new("RGBA", (CARD_W, CARD_H), (255, 255, 255, 0))
    highlight.putalpha(gradient.resize((CARD_W, CARD_H)))
    card = Image.alpha_composite(card, highlight)
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle(
        [(0, 0), (CARD_W - 1, CARD_H - 1)],
        radius=CARD_RADIUS,
        outline=(255, 255, 255, 40),
        width=1
    )
    draw.rounded_rectangle(
        [(0, 0), (7, CARD_H - 1)],
        radius=4,
        fill=accent
    )
    text_x = 30
    draw.text((text_x, 18), name, font=FONT_LG, fill=(255, 255, 255, 255))
    rel_w = draw.textlength(relationship, font=FONT_SM)
    badge_h = 32
    badge_w = rel_w + 28
    badge_x = text_x
    badge_y = 56
    draw.rounded_rectangle(
        [(badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h)],
        radius=badge_h / 2,
        fill=accent
    )
    draw.text((badge_x + 14, badge_y + 6), relationship, font=FONT_SM, fill=(255, 255, 255, 255))

    divider_y = badge_y + badge_h + 14
    draw.line(
        [(text_x, divider_y), (CARD_W - 24, divider_y)],
        fill=(255, 255, 255, 35),
        width=1
    )
    text_y = divider_y + 12
    for line in wrap_text(draw, summary, FONT_XS, CARD_W - text_x - 24, 2):
        draw.text((text_x, text_y), line, font=FONT_XS, fill=(205, 207, 212, 255))
        text_y += 24

    rgba = np.array(card, dtype=np.float32)
    alpha = rgba[:, :, 3:4] / 255.0
    premultiplied = rgba[:, :, 2::-1] * alpha
    inverse_alpha = 1.0 - alpha

    if len(CARD_CACHE) > 64:
        CARD_CACHE.clear()
    CARD_CACHE[key] = (premultiplied, inverse_alpha)
    return CARD_CACHE[key]


def build_compact_card(identity, box_color):
    """Render a smaller name + relationship card for two people."""
    key = (identity, box_color)
    cached = COMPACT_CARD_CACHE.get(key)
    if cached is not None:
        return cached

    name, relationship, _summary, _known = identity
    width, height = COMPACT_W, COMPACT_H
    accent = (box_color[2], box_color[1], box_color[0], 255)

    card = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card)

    draw.rounded_rectangle(
        [(0, 0), (width - 1, height - 1)],
        radius=COMPACT_RADIUS,
        fill=(18, 19, 24, 225)
    )
    draw.rounded_rectangle(
        [(0, 0), (width - 1, height - 1)],
        radius=COMPACT_RADIUS,
        outline=(255, 255, 255, 40),
        width=1
    )
    draw.rounded_rectangle(
        [(0, 0), (6, height - 1)],
        radius=3,
        fill=accent
    )

    draw.text((22, 14), name, font=FONT_SM, fill=(255, 255, 255, 255))

    rel_w = draw.textlength(relationship, font=FONT_XS)
    badge_h = 28
    badge_w = rel_w + 24
    badge_x = 22
    badge_y = 52
    draw.rounded_rectangle(
        [(badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h)],
        radius=badge_h / 2,
        fill=accent
    )
    draw.text((badge_x + 12, badge_y + 4), relationship, font=FONT_XS, fill=(255, 255, 255, 255))

    rgba = np.array(card, dtype=np.float32)
    alpha = rgba[:, :, 3:4] / 255.0
    premultiplied = rgba[:, :, 2::-1] * alpha
    inverse_alpha = 1.0 - alpha

    if len(COMPACT_CARD_CACHE) > 64:
        COMPACT_CARD_CACHE.clear()
    COMPACT_CARD_CACHE[key] = (premultiplied, inverse_alpha, width, height)
    return COMPACT_CARD_CACHE[key]


def paste_card(frame, identity, box_color, x, y):
    """Alpha-blend only the card region instead of the whole frame."""
    height, width = frame.shape[:2]
    x = max(0, min(x, width - CARD_W))
    y = max(0, min(y, height - CARD_H))

    premultiplied, inverse_alpha = build_card(identity, box_color)
    region = frame[y:y + CARD_H, x:x + CARD_W]
    region[:] = (premultiplied + region.astype(np.float32) * inverse_alpha).astype(np.uint8)
    return x, y


def paste_compact_card(frame, identity, box_color, x, y):
    """Place a small information card above or below a face."""
    height, width = frame.shape[:2]
    premultiplied, inverse_alpha, card_width, card_height = build_compact_card(identity, box_color)
    x = max(5, min(x, width - card_width - 5))
    y = max(5, min(y, height - card_height - 5))

    region = frame[y:y + card_height, x:x + card_width]
    region[:] = (premultiplied + region.astype(np.float32) * inverse_alpha).astype(np.uint8)
    return x, y, card_width, card_height


def build_name_label(identity, box_color):
    """Render only the person's name for the 3+ people layout."""
    key = (identity[0], box_color)
    cached = NAME_CACHE.get(key)
    if cached is not None:
        return cached

    name = identity[0]
    accent = (box_color[2], box_color[1], box_color[0], 255)
    bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), name, font=FONT_SM)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    width = text_width + 44
    height = max(40, text_height + 20)
    radius = height / 2

    label = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(label)
    draw.rounded_rectangle(
        [(0, 0), (width - 1, height - 1)],
        radius=radius,
        fill=(18, 19, 24, 225)
    )
    draw.rounded_rectangle(
        [(0, 0), (width - 1, height - 1)],
        radius=radius,
        outline=(255, 255, 255, 40),
        width=1
    )
    dot_r = 6
    dot_cx = 18
    dot_cy = height / 2
    draw.ellipse(
        [(dot_cx - dot_r, dot_cy - dot_r), (dot_cx + dot_r, dot_cy + dot_r)],
        fill=accent
    )
    draw.text((32, (height - text_height) / 2 - bbox[1]), name, font=FONT_SM, fill=(255, 255, 255, 255))

    rgba = np.array(label, dtype=np.float32)
    alpha = rgba[:, :, 3:4] / 255.0
    premultiplied = rgba[:, :, 2::-1] * alpha
    inverse_alpha = 1.0 - alpha

    if len(NAME_CACHE) > 64:
        NAME_CACHE.clear()
    NAME_CACHE[key] = (premultiplied, inverse_alpha, width, height)
    return NAME_CACHE[key]


def paste_name_label(frame, identity, box_color, x, y):
    """Place the name label directly above the face box."""
    height, width = frame.shape[:2]
    premultiplied, inverse_alpha, label_width, label_height = build_name_label(identity, box_color)
    x = max(5, min(x, width - label_width - 5))
    y = max(5, min(y, height - label_height - 5))

    region = frame[y:y + label_height, x:x + label_width]
    region[:] = (premultiplied + region.astype(np.float32) * inverse_alpha).astype(np.uint8)
    return x, y, label_width, label_height


_fitted_cache = {}


def fit_text(text, scale, thickness, max_width):
    """Trim a status line so it never runs off the side of the window."""
    key = (text, max_width)
    cached = _fitted_cache.get(key)
    if cached is not None:
        return cached

    font = cv2.FONT_HERSHEY_SIMPLEX
    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_width:
        fitted = text
    else:
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            width = cv2.getTextSize(text[:mid] + "...", font, scale, thickness)[0][0]
            if width <= max_width:
                low = mid
            else:
                high = mid - 1
        fitted = text[:low].rstrip() + "..."

    if len(_fitted_cache) > 32:
        _fitted_cache.clear()
    _fitted_cache[key] = fitted
    return fitted


def handle_key(key, tracks):
    if key == ord("q"):
        return False

    if key == ord("r"):
        if len(tracks) != 1:
            print("Please make sure one face is visible only.")
            return True

        with audio_lock:
            if audio_status["processing"]:
                print("Audio processing is already running.")
                return True

        with registration_lock:
            if registration_state["active"]:
                print("3-angle registration is already running.")
                return True

        embed = tracks[0]["embedding"]
        best_match, best_score = match_embedding(embed)

        if best_match is not None and best_score >= THRESHOLD:
            print(f"{best_match['name']} is already registered.")
            set_status(f"{best_match['name']} is already registered.")
            return True

        print("Unknown person detected.")
        print("Starting 3-angle registration.")
        print("Look FRONT, then LEFT, then RIGHT.")

        start_three_angle_registration()
        return True

    if key == ord("u"):
        if len(tracks) != 1:
            print("Please make sure one face is visible only.")
            return True

        with audio_lock:
            if audio_status["processing"]:
                print("Audio processing is already running.")
                return True

        embed = tracks[0]["embedding"]
        best_match, best_score = match_embedding(embed)

        if best_match is None or best_score < THRESHOLD:
            print("Person is not registered.")
            return True

        print(f"Recording conversation with {best_match['name']}...")

        if start_update_thread(best_match, embed):
            print("Camera continues running...")
        else:
            print("Audio processing is already running.")

    if key == ord("s"):
        if len(tracks) != 1:
            print("Please make sure one face is visible only.")
            return True

        with audio_lock:
            if audio_status["processing"]:
                print("Audio processing is already running.")
                return True

        embed = tracks[0]["embedding"]
        best_match, best_score = match_embedding(embed)

        if best_match is None or best_score < THRESHOLD:
            print("Person is not registered.")
            return True

        if start_starter_thread(best_match):
            print("Starter question is being generated...")
        else:
            print("Audio processing is already running.")

    return True

camera = CameraStream(0, 1280, 720)
detector = Detector(camera)

tracks = []
last_frame_id = -1
last_detection_stamp = 0.0
render_times = deque()
previous_render = time.perf_counter()
display_fps = 0.0
FPS_WINDOW = 1.0      # seconds of history averaged for the display FPS
DETECTOR_STALE = 1.5  # seconds without a detection before Detect reads 0.0

print("Running. Press 'r' to register a face, 'u' to update conversation, 's' for a starter question, 'q' to quit.")

try:
    while True:
        raw_frame, frame_id = camera.read()
        if raw_frame is None:
            if not handle_key(cv2.waitKey(10) & 0xFF, tracks):
                break
            continue

        if frame_id == last_frame_id:
            if not handle_key(cv2.waitKey(1) & 0xFF, tracks):
                break
            continue
        last_frame_id = frame_id

        frame = raw_frame.copy()
        now = time.perf_counter()

        faces, detection_stamp, detector_fps, detector_latency = detector.latest()
        if detection_stamp != last_detection_stamp:
            last_detection_stamp = detection_stamp
            tracks = update_tracks(tracks, faces, now)
        else:
            tracks = [t for t in tracks if now - t["last_seen"] <= TRACK_TTL]

        delta = now - previous_render
        previous_render = now
        smoothing = 1.0 - math.exp(-delta / BOX_SMOOTHING_TAU) if delta > 0 else 1.0
        for track in tracks:
            track["display"] += (track["target"] - track["display"]) * smoothing

        with registration_lock:
            registration_active = registration_state["active"]
            registration_started = registration_state["started"]

        if registration_active:
            if time.time() - registration_started > REGISTRATION_MAX_TIME:
                cancel_three_angle_registration(
                    "Registration timed out. Press r to try again."
                )
            elif len(tracks) == 1:
                registration_capture(tracks[0])
            else:
                with registration_lock:
                    if registration_state["active"]:
                        registration_state["angle_started"] = time.time()
                set_status(
                    "Registration waiting: keep exactly ONE face visible.",
                    error=True
                )

        person_count = len(tracks)
        frame_height, frame_width = frame.shape[:2]

        paste_legend(frame, 20, 20)

        if person_count == 1:
            track = tracks[0]
            x1, y1, x2, y2 = track["display"].astype(int)
            identity = track["identity"]
            box_color = (0, 255, 0) if identity[3] else (0, 0, 255)

            card_x = max(10, (frame_width - CARD_W) // 2)
            card_y = 20

            card_x, card_y = paste_card(frame, identity, box_color, card_x, card_y)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            if card_y > y2:
                cv2.line(
                    frame,
                    ((x1 + x2) // 2, y2),
                    ((x1 + x2) // 2, card_y),
                    box_color,
                    1,
                    cv2.LINE_AA
                )
            elif card_y + CARD_H < y1:
                cv2.line(
                    frame,
                    ((x1 + x2) // 2, y1),
                    ((x1 + x2) // 2, card_y + CARD_H),
                    box_color,
                    1,
                    cv2.LINE_AA
                )

        elif person_count == 2:
            compact_width = COMPACT_W
            compact_height = COMPACT_H
            for track in tracks:
                x1, y1, x2, y2 = track["display"].astype(int)
                identity = track["identity"]
                box_color = (0, 255, 0) if identity[3] else (0, 0, 255)

                card_x = ((x1 + x2) // 2) - compact_width // 2
                below_y = y2 + 12
                above_y = y1 - compact_height - 12

                if below_y + compact_height <= frame_height - 8:
                    card_y = below_y
                else:
                    card_y = max(8, above_y)

                card_x, card_y, _, _ = paste_compact_card(
                    frame,
                    identity,
                    box_color,
                    card_x,
                    card_y
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
                cv2.line(
                    frame,
                    ((x1 + x2) // 2, y2 if card_y > y2 else y1),
                    ((x1 + x2) // 2, card_y if card_y > y2 else card_y + compact_height),
                    box_color,
                    1,
                    cv2.LINE_AA
                )

        else:
            for track in tracks:
                x1, y1, x2, y2 = track["display"].astype(int)
                identity = track["identity"]
                box_color = (0, 255, 0) if identity[3] else (0, 0, 255)

                label_data = build_name_label(identity, box_color)
                label_width = label_data[2]
                label_height = label_data[3]

                label_x = ((x1 + x2) // 2) - label_width // 2
                label_y = y1 - label_height - 8
                if label_y < 5:
                    label_y = y2 + 8

                paste_name_label(
                    frame,
                    identity,
                    box_color,
                    label_x,
                    label_y
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        render_times.append(now)
        while len(render_times) > 2 and now - render_times[0] > FPS_WINDOW:
            render_times.popleft()
        if len(render_times) > 1:
            span = render_times[-1] - render_times[0]
            if span > 0:
                display_fps = (len(render_times) - 1) / span
        if now - detection_stamp > DETECTOR_STALE:
            detector_fps = 0.0
            detector_latency = 0.0
        fps_text = f"{display_fps:.0f} FPS"
        fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(
            frame,
            fps_text,
            (frame.shape[1] - fps_size[0] - 24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        with registration_lock:
            registration_active = registration_state["active"]
            registration_angle_index = registration_state["angle_index"]
        if (
            registration_active
            and registration_angle_index < len(REGISTRATION_ANGLES)
        ):
            current_angle = REGISTRATION_ANGLES[registration_angle_index]
            registration_text = (
                f"{current_angle.upper()}  •  "
                f"{REGISTRATION_MESSAGES[current_angle]}"
            )
            registration_size = cv2.getTextSize(
                registration_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )[0]
            cv2.putText(
                frame,
                registration_text,
                ((frame.shape[1] - registration_size[0]) // 2, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )
        with audio_lock:
            audio_processing = audio_status["processing"]
            audio_message = audio_status["message"]
            audio_error = audio_status["error"]
            audio_stamp = audio_status["stamp"]
        if audio_message and time.time() - audio_stamp > MESSAGE_TTL:
            audio_message = ""
        if audio_processing:
            status_text = audio_message if audio_message else "Listening..."
            status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0]
            cv2.putText(
                frame,
                status_text,
                ((frame.shape[1] - status_size[0]) // 2, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )
        elif audio_message:
            fitted_message = fit_text(audio_message, 0.8, 2, frame.shape[1] - 48)
            message_size = cv2.getTextSize(fitted_message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.putText(
                frame,
                fitted_message,
                ((frame.shape[1] - message_size[0]) // 2, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255) if audio_error else (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        cv2.imshow("MemoryLens Face Detection", frame)

        if not handle_key(cv2.waitKey(1) & 0xFF, tracks):
            break
finally:
    detector.stop()
    camera.stop()
    if people is not None:
        mongo_client.close()
    cv2.destroyAllWindows()    