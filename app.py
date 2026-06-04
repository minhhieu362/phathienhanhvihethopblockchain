import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, send_from_directory

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional runtime dependency
    OpenAI = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
SNAPSHOT_DIR = LOG_DIR / "snapshots"
DB_PATH = LOG_DIR / "behavior.db"
ETH_LEDGER = LOG_DIR / "ethereum_ledger.jsonl"
TRAINED_YOLO_MODEL = BASE_DIR / "models" / "classroom_behavior_yolo11n_best.pt"
DEFAULT_CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "behaviorId", "type": "string"},
            {"internalType": "string", "name": "hashData", "type": "string"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
        ],
        "name": "storeBehaviorHash",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]
DEFAULT_CONTRACT_ADDRESS = "0xd1Fd789e7D5d11685CFf19B09f6A29212e6B5372"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
CHIN = 152
LEFT_FACE = 234
RIGHT_FACE = 454
FOREHEAD = 10
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

BEHAVIOR_LABELS = {
    "Eating": "An uong",
    "Sleeping": "Ngu gat",
    "Using Phone": "Dung dien thoai",
    "Distracted": "Mat tap trung",
    "Left Seat": "Roi khoi vi tri",
    "Raised Hand": "Gio tay phat bieu",
    "Standing Up": "Dung day",
    "Talking": "Noi chuyen",
}


@dataclass
class BehaviorConfig:
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    camera_backend: str = os.getenv("CAMERA_BACKEND", "DSHOW").upper()
    camera_width: int = int(os.getenv("CAMERA_WIDTH", "640"))
    camera_height: int = int(os.getenv("CAMERA_HEIGHT", "480"))
    camera_read_fail_limit: int = int(os.getenv("CAMERA_READ_FAIL_LIMIT", "30"))
    eye_closed_ear: float = float(os.getenv("EYE_CLOSED_EAR", "0.21"))
    eye_closed_adaptive_ratio: float = float(os.getenv("EYE_CLOSED_ADAPTIVE_RATIO", "0.68"))
    drowsy_seconds: float = float(os.getenv("DROWSY_SECONDS", "3.0"))
    drowsy_eye_seconds: float = float(os.getenv("DROWSY_EYE_SECONDS", os.getenv("DROWSY_SECONDS", "0.75")))
    drowsy_head_seconds: float = float(os.getenv("DROWSY_HEAD_SECONDS", os.getenv("DROWSY_SECONDS", "1.1")))
    drowsy_signal_grace_seconds: float = float(os.getenv("DROWSY_SIGNAL_GRACE_SECONDS", "0.30"))
    head_down_ratio: float = float(os.getenv("HEAD_DOWN_RATIO", "0.58"))
    head_down_delta: float = float(os.getenv("HEAD_DOWN_DELTA", "0.07"))
    face_lost_drowsy_grace_seconds: float = float(os.getenv("FACE_LOST_DROWSY_GRACE_SECONDS", "1.5"))
    haar_head_down_ratio: float = float(os.getenv("HAAR_HEAD_DOWN_RATIO", "0.88"))
    distracted_yaw_ratio: float = float(os.getenv("DISTRACTED_YAW_RATIO", "0.14"))
    distracted_seconds: float = float(os.getenv("DISTRACTED_SECONDS", "2.0"))
    distracted_face_lost: bool = os.getenv("DISTRACTED_FACE_LOST", "1") != "0"
    left_seat_seconds: float = float(os.getenv("LEFT_SEAT_SECONDS", "3.0"))
    phone_confidence: float = float(os.getenv("PHONE_CONFIDENCE", "0.35"))
    phone_hold_seconds: float = float(os.getenv("PHONE_HOLD_SECONDS", "0.7"))
    phone_detect_every: int = int(os.getenv("PHONE_DETECT_EVERY", "3"))
    eating_seconds: float = float(os.getenv("EATING_SECONDS", "1.0"))
    yolo_behavior_confidence: float = float(os.getenv("YOLO_BEHAVIOR_CONFIDENCE", "0.45"))
    use_yolo_behavior_classes: bool = os.getenv("USE_YOLO_BEHAVIOR_CLASSES", "1") == "1"
    raised_hand_seconds: float = float(os.getenv("RAISED_HAND_SECONDS", "0.5"))
    raised_hand_margin: float = float(os.getenv("RAISED_HAND_MARGIN", "0.04"))
    standing_seconds: float = float(os.getenv("STANDING_SECONDS", "1.0"))
    standing_person_height_ratio: float = float(os.getenv("STANDING_PERSON_HEIGHT_RATIO", "0.62"))
    standing_person_aspect_ratio: float = float(os.getenv("STANDING_PERSON_ASPECT_RATIO", "1.55"))
    standing_body_span_ratio: float = float(os.getenv("STANDING_BODY_SPAN_RATIO", "0.48"))
    standing_baseline_delta: float = float(os.getenv("STANDING_BASELINE_DELTA", "0.12"))
    use_box_standing: bool = os.getenv("USE_BOX_STANDING", "0") == "1"
    pose_visibility: float = float(os.getenv("POSE_VISIBILITY", "0.55"))
    talking_seconds: float = float(os.getenv("TALKING_SECONDS", "1.0"))
    talking_mouth_ratio: float = float(os.getenv("TALKING_MOUTH_RATIO", "0.10"))
    talking_mouth_delta: float = float(os.getenv("TALKING_MOUTH_DELTA", "0.035"))
    talking_window_seconds: float = float(os.getenv("TALKING_WINDOW_SECONDS", "1.2"))
    talking_turn_yaw_ratio: float = float(os.getenv("TALKING_TURN_YAW_RATIO", "0.10"))
    audio_talking_hold_seconds: float = float(os.getenv("AUDIO_TALKING_HOLD_SECONDS", "7.0"))
    audio_private_hold_seconds: float = float(os.getenv("AUDIO_PRIVATE_HOLD_SECONDS", "10.0"))
    audio_private_threshold: int = int(os.getenv("AUDIO_PRIVATE_THRESHOLD", "1"))
    openai_classifier_model: str = os.getenv("OPENAI_CLASSIFIER_MODEL", "gpt-5.4-mini")
    openai_classifier_timeout: float = float(os.getenv("OPENAI_CLASSIFIER_TIMEOUT", "8.0"))
    openai_private_confidence: float = float(os.getenv("OPENAI_PRIVATE_CONFIDENCE", "0.65"))
    led_gpio_pin: int = int(os.getenv("LED_GPIO_PIN", "-1"))
    led_active_high: bool = os.getenv("LED_ACTIVE_HIGH", "1") != "0"
    led_serial_port: str = os.getenv("LED_SERIAL_PORT", "").strip()
    led_serial_baud: int = int(os.getenv("LED_SERIAL_BAUD", "9600"))
    yolo_model: str = os.getenv("YOLO_MODEL", str(TRAINED_YOLO_MODEL) if TRAINED_YOLO_MODEL.exists() else "yolo11n.pt")
    person_yolo_model: str = os.getenv("PERSON_YOLO_MODEL", "yolo11n.pt")
    use_yolo: bool = os.getenv("USE_YOLO", "1") != "0"
    event_cooldown: float = float(os.getenv("EVENT_COOLDOWN", "6.0"))


class HoldTimer:
    def __init__(self, seconds: float, false_grace_seconds: float = 0.0):
        self.seconds = seconds
        self.false_grace_seconds = false_grace_seconds
        self.started_at: Optional[float] = None
        self.last_true_at: Optional[float] = None

    def update(self, condition: bool, now: float) -> bool:
        if condition:
            if self.started_at is None:
                self.started_at = now
            self.last_true_at = now
            return now - self.started_at >= self.seconds
        if (
            self.false_grace_seconds > 0
            and self.started_at is not None
            and self.last_true_at is not None
            and now - self.last_true_at <= self.false_grace_seconds
        ):
            return now - self.started_at >= self.seconds
        self.started_at = None
        self.last_true_at = None
        return False


class FpsMeter:
    def __init__(self):
        self.last_time = time.monotonic()
        self.fps = 0.0

    def update(self) -> float:
        now = time.monotonic()
        delta = now - self.last_time
        self.last_time = now
        if delta > 0:
            current = 1.0 / delta
            self.fps = current if self.fps == 0 else (self.fps * 0.9) + (current * 0.1)
        return self.fps


class BehaviorDatabase:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS behavior_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    behavior TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    hash TEXT NOT NULL UNIQUE,
                    ethereum_tx TEXT,
                    status TEXT NOT NULL DEFAULT 'confirmed',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def insert_event(self, behavior: str, timestamp: str, image_path: str, hash_data: str) -> int:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO behavior_logs (behavior, timestamp, image_path, hash)
                VALUES (?, ?, ?, ?)
                """,
                (behavior, timestamp, image_path, hash_data),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def update_ethereum_tx(self, event_id: int, ethereum_tx: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE behavior_logs
                SET ethereum_tx = ?
                WHERE id = ?
                """,
                (ethereum_tx, event_id),
            )
            conn.commit()

    def list_events(self, limit: int = 50) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, behavior, timestamp, image_path, hash, ethereum_tx, status
                FROM behavior_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_event(self, event_id: int) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, behavior, timestamp, image_path, hash, ethereum_tx, status
                FROM behavior_logs
                WHERE id = ?
                """,
                (event_id,),
            ).fetchone()
        return dict(row) if row else None


class EthereumClient:
    def __init__(self, ledger_path: Path):
        self.ledger_path = ledger_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def submit_hash(self, event_id: int, hash_data: str, timestamp: str) -> str:
        if os.getenv("ETHEREUM_ENABLED", "0") == "1":
            return self._submit_web3(event_id, hash_data, timestamp)
        return self._submit_mock(event_id, hash_data, timestamp)

    def _submit_mock(self, event_id: int, hash_data: str, timestamp: str) -> str:
        tx_hash = f"eth-mock-{event_id}-{int(time.time())}"
        payload = {
            "tx_hash": tx_hash,
            "network": os.getenv("ETHEREUM_NETWORK", "mock-sepolia"),
            "behaviorId": str(event_id),
            "hashData": hash_data,
            "timestamp": timestamp,
        }
        with self._lock, self.ledger_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return tx_hash

    def _submit_web3(self, event_id: int, hash_data: str, timestamp: str) -> str:
        try:
            from web3 import Web3
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("web3 chua duoc cai dat") from exc

        rpc_url = os.environ["ETHEREUM_RPC_URL"]
        private_key = os.environ["ETHEREUM_PRIVATE_KEY"]
        contract_address = os.getenv("ETHEREUM_CONTRACT_ADDRESS", DEFAULT_CONTRACT_ADDRESS)
        contract_abi = json.loads(os.getenv("ETHEREUM_CONTRACT_ABI", json.dumps(DEFAULT_CONTRACT_ABI)))

        web3 = Web3(Web3.HTTPProvider(rpc_url))
        account = web3.eth.account.from_key(private_key)
        contract = web3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=contract_abi)
        unix_ts = int(datetime.fromisoformat(timestamp).timestamp())
        txn = contract.functions.storeBehaviorHash(str(event_id), hash_data, unix_ts).build_transaction(
            {
                "from": account.address,
                "nonce": web3.eth.get_transaction_count(account.address),
                "gas": int(os.getenv("ETHEREUM_GAS", "180000")),
                "gasPrice": web3.eth.gas_price,
                "chainId": int(os.getenv("ETHEREUM_CHAIN_ID", "11155111")),
            }
        )
        signed = account.sign_transaction(txn)
        tx_hash = web3.eth.send_raw_transaction(signed.rawTransaction)
        return web3.to_hex(tx_hash)


class ConversationClassifier:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string", "enum": ["private", "study", "unknown"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["label", "confidence", "reason"],
    }

    instructions = """
Ban la bo phan loai transcript trong lop hoc.
Hay xac dinh cau noi la:
- private: noi chuyen rieng/off-topic, noi ve viec ca nhan, giai tri, mua ban, hen gap, mang xa hoi, viec khong lien quan bai hoc.
- study: noi ve bai hoc, cau hoi/giai thich kien thuc, bai tap, giao vien, thuyet trinh, thao luan nhom phuc vu hoc tap.
- unknown: transcript qua ngan, nhieu loi nghe, hoac khong du ngu canh.
Chi tra ve JSON theo schema. Khong dua keyword list vao ly do.
""".strip()

    def __init__(self, cfg: BehaviorConfig):
        self.cfg = cfg
        self.client = None
        if OpenAI is not None and os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(timeout=cfg.openai_classifier_timeout)

    def classify(self, text: str) -> Dict[str, object]:
        if self.client is None:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "academic_score": 0,
                "private_score": 0,
                "reason": "Chua cau hinh OPENAI_API_KEY hoac chua cai package openai.",
                "error": "openai_not_configured",
            }

        try:
            response = self.client.responses.create(
                model=self.cfg.openai_classifier_model,
                instructions=self.instructions,
                input=f"Transcript: {text}",
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "classroom_conversation_classification",
                        "schema": self.schema,
                        "strict": True,
                    }
                },
                max_output_tokens=120,
            )
            payload = json.loads(response.output_text)
        except Exception as exc:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "academic_score": 0,
                "private_score": 0,
                "reason": f"Loi goi OpenAI API: {exc}",
                "error": "openai_api_error",
            }

        label = str(payload.get("label", "unknown"))
        if label not in {"private", "study", "unknown"}:
            label = "unknown"
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
        private_score = int(round(confidence * 100)) if label == "private" else 0
        academic_score = int(round(confidence * 100)) if label == "study" else 0
        return {
            "label": label,
            "confidence": confidence,
            "academic_score": academic_score,
            "private_score": private_score,
            "reason": str(payload.get("reason", ""))[:180],
            "error": "",
        }


class LedController:
    def __init__(self, cfg: BehaviorConfig):
        self.pin = cfg.led_gpio_pin
        self.active_high = cfg.led_active_high
        self.is_on = False
        self.mode = "virtual"
        self._led = None
        self._serial = None
        self._last_sent: Optional[bool] = None
        if cfg.led_serial_port:
            try:
                import serial

                self._serial = serial.Serial(cfg.led_serial_port, cfg.led_serial_baud, timeout=1)
                time.sleep(2.0)
                self.mode = f"Serial {cfg.led_serial_port}"
            except Exception as exc:
                self.mode = f"virtual (Serial loi: {exc})"
        elif self.pin >= 0:
            try:
                from gpiozero import LED

                self._led = LED(self.pin, active_high=self.active_high)
                self.mode = f"GPIO {self.pin}"
            except Exception as exc:
                self.mode = f"virtual (GPIO loi: {exc})"

    def set(self, enabled: bool) -> None:
        self.is_on = bool(enabled)
        if self._serial is not None and self._last_sent != self.is_on:
            try:
                self._serial.write(b"ON\n" if self.is_on else b"OFF\n")
                self._serial.flush()
                self._last_sent = self.is_on
            except Exception as exc:
                self.mode = f"virtual (Serial loi: {exc})"
                self._serial = None
        if self._led is not None:
            if self.is_on:
                self._led.on()
            else:
                self._led.off()


class HaarAttentionDetector:
    def __init__(self):
        face_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        eye_path = str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml")
        self.face_cascade = cv2.CascadeClassifier(face_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_path)
        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise RuntimeError("Khong tai duoc Haar Cascade cua OpenCV.")

    def process(self, frame, cfg: BehaviorConfig) -> Tuple[bool, Dict[str, bool], Dict[str, float], list]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return False, {"eyes_closed": False, "head_down": False, "looking_away": False}, {}, []

        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        upper_face = gray[y : y + int(h * 0.62), x : x + w]
        eyes = self.eye_cascade.detectMultiScale(upper_face, scaleFactor=1.1, minNeighbors=5, minSize=(18, 18))
        frame_center_x = frame.shape[1] / 2.0
        face_center_x = x + (w / 2.0)
        yaw_ratio = abs(face_center_x - frame_center_x) / max(frame.shape[1], 1)
        face_down_ratio = (y + h) / max(frame.shape[0], 1)
        metrics = {
            "ear": 0.0 if len(eyes) == 0 else 0.30,
            "yaw_ratio": yaw_ratio,
            "nose_down_ratio": face_down_ratio,
        }
        attention = {
            "eyes_closed": len(eyes) == 0,
            "head_down": face_down_ratio > cfg.haar_head_down_ratio,
            "looking_away": yaw_ratio > cfg.distracted_yaw_ratio,
        }
        return True, attention, metrics, [(x, y, w, h, eyes)]


def has_mediapipe_face_mesh() -> bool:
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")


def has_mediapipe_pose() -> bool:
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")


def point(landmarks, idx: int, width: int, height: int) -> Tuple[int, int]:
    lm = landmarks[idx]
    return int(lm.x * width), int(lm.y * height)


def distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(landmarks, indices, width: int, height: int) -> float:
    p1, p2, p3, p4, p5, p6 = [point(landmarks, i, width, height) for i in indices]
    vertical = distance(p2, p6) + distance(p3, p5)
    horizontal = distance(p1, p4)
    return 0.0 if horizontal <= 1 else vertical / (2.0 * horizontal)


def mouth_open_ratio(landmarks, width: int, height: int) -> float:
    top = point(landmarks, MOUTH_TOP, width, height)
    bottom = point(landmarks, MOUTH_BOTTOM, width, height)
    left = point(landmarks, MOUTH_LEFT, width, height)
    right = point(landmarks, MOUTH_RIGHT, width, height)
    horizontal = max(distance(left, right), 1.0)
    return distance(top, bottom) / horizontal


def estimate_attention(landmarks, width: int, height: int, cfg: BehaviorConfig) -> Dict[str, float]:
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, width, height)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, width, height)
    ear = (left_ear + right_ear) / 2.0
    nose = point(landmarks, NOSE_TIP, width, height)
    chin = point(landmarks, CHIN, width, height)
    forehead = point(landmarks, FOREHEAD, width, height)
    left_face = point(landmarks, LEFT_FACE, width, height)
    right_face = point(landmarks, RIGHT_FACE, width, height)
    face_width = max(distance(left_face, right_face), 1.0)
    face_height = max(distance(forehead, chin), 1.0)
    face_center_x = (left_face[0] + right_face[0]) / 2.0
    yaw_ratio = abs(nose[0] - face_center_x) / face_width
    nose_down_ratio = (nose[1] - forehead[1]) / face_height
    mouth_ratio = mouth_open_ratio(landmarks, width, height)
    return {
        "ear": ear,
        "yaw_ratio": yaw_ratio,
        "nose_down_ratio": nose_down_ratio,
        "mouth_open_ratio": mouth_ratio,
        "eyes_closed": ear < cfg.eye_closed_ear,
        "head_down": nose_down_ratio > cfg.head_down_ratio,
        "looking_away": yaw_ratio > cfg.distracted_yaw_ratio,
    }


def create_face_mesh():
    if not has_mediapipe_face_mesh():
        return None
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def create_pose():
    if not has_mediapipe_pose():
        return None
    return mp.solutions.pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def load_yolo(cfg: BehaviorConfig):
    if not cfg.use_yolo or YOLO is None:
        return None
    try:
        return YOLO(cfg.yolo_model)
    except Exception as exc:
        print(f"[WARN] Khong tai duoc YOLO model '{cfg.yolo_model}': {exc}")
        return None


def load_person_yolo(cfg: BehaviorConfig, behavior_yolo=None):
    if not cfg.use_yolo or YOLO is None:
        return None
    if str(cfg.person_yolo_model) == str(cfg.yolo_model):
        return behavior_yolo
    try:
        return YOLO(cfg.person_yolo_model)
    except Exception as exc:
        print(f"[WARN] Khong tai duoc person YOLO model '{cfg.person_yolo_model}': {exc}")
        return None


def open_camera(cfg: BehaviorConfig):
    backends = {
        "DSHOW": cv2.CAP_DSHOW,
        "MSMF": cv2.CAP_MSMF,
        "ANY": cv2.CAP_ANY,
    }
    preferred = cfg.camera_backend if cfg.camera_backend in backends else "DSHOW"
    ordered = [preferred] + [name for name in ("DSHOW", "MSMF", "ANY") if name != preferred]
    for name in ordered:
        cap = cv2.VideoCapture(cfg.camera_index, backends[name])
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera_height)
        return cap, name
    return None, ""


def box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def phone_near_person(phone_box: Tuple[int, int, int, int], person_boxes: List[Tuple[int, int, int, int]]) -> bool:
    if not person_boxes:
        return True
    px, py = box_center(phone_box)
    for x1, y1, x2, y2 in person_boxes:
        margin_x = (x2 - x1) * 0.25
        margin_y = (y2 - y1) * 0.25
        if x1 - margin_x <= px <= x2 + margin_x and y1 - margin_y <= py <= y2 + margin_y:
            return True
    return False


def normalize_label(label: str) -> str:
    return " ".join(str(label).replace("_", " ").replace("-", " ").strip().lower().split())


def behavior_from_yolo_label(label: str) -> Optional[str]:
    normalized = normalize_label(label)
    aliases = {
        "eating": "Eating",
        "an uong": "Eating",
        "sleeping": "Sleeping",
        "using phone": "Using Phone",
        "using mobile": "Using Phone",
        "usingmobile": "Using Phone",
        "mobile phone": "Using Phone",
        "phone": "Using Phone",
        "distracted": "Distracted",
        "left seat": "Left Seat",
        "leaving seat": "Left Seat",
        "raised hand": "Raised Hand",
        "raise hand": "Raised Hand",
        "hand raising": "Raised Hand",
        "gio tay": "Raised Hand",
        "standing up": "Standing Up",
        "stand up": "Standing Up",
        "standing": "Standing Up",
        "dung day": "Standing Up",
    }
    return aliases.get(normalized)


def detect_yolo(model, frame, cfg: BehaviorConfig) -> Tuple[bool, bool, list, list, Dict[str, bool], list]:
    if model is None:
        return False, False, [], [], {}, []
    confidence = min(cfg.phone_confidence, cfg.yolo_behavior_confidence)
    results = model.predict(frame, imgsz=640, conf=confidence, verbose=False)
    phone_boxes = []
    person_boxes = []
    behavior_signals = {name: False for name in BEHAVIOR_LABELS}
    behavior_boxes = []
    for result in results:
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = names.get(cls_id, str(cls_id))
            score = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            behavior = behavior_from_yolo_label(label)
            if cfg.use_yolo_behavior_classes and behavior and score >= cfg.yolo_behavior_confidence:
                behavior_signals[behavior] = True
                behavior_boxes.append((x1, y1, x2, y2, behavior, score))
            if normalize_label(label) == "cell phone" and score >= cfg.phone_confidence:
                phone_boxes.append((x1, y1, x2, y2, score))
            if normalize_label(label) == "person" and score >= cfg.phone_confidence:
                person_boxes.append((x1, y1, x2, y2))
    phone_found = any(phone_near_person(box[:4], person_boxes) for box in phone_boxes)
    phone_found = phone_found or behavior_signals.get("Using Phone", False)
    return bool(person_boxes or behavior_boxes), phone_found, phone_boxes, person_boxes, behavior_signals, behavior_boxes


def compute_hash(behavior: str, timestamp: str, image_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(behavior.encode("utf-8"))
    digest.update(timestamp.encode("utf-8"))
    if image_path.exists():
        digest.update(image_path.read_bytes())
    else:
        digest.update(str(image_path).encode("utf-8"))
    return digest.hexdigest()


def public_metamask_config() -> Dict:
    abi_text = os.getenv("METAMASK_CONTRACT_ABI") or os.getenv("ETHEREUM_CONTRACT_ABI")
    if abi_text:
        try:
            abi = json.loads(abi_text)
        except json.JSONDecodeError:
            abi = DEFAULT_CONTRACT_ABI
    else:
        abi = DEFAULT_CONTRACT_ABI
    return {
        "chain_id": int(os.getenv("METAMASK_CHAIN_ID", os.getenv("ETHEREUM_CHAIN_ID", "11155111"))),
        "contract_address": os.getenv(
            "METAMASK_CONTRACT_ADDRESS",
            os.getenv("ETHEREUM_CONTRACT_ADDRESS", DEFAULT_CONTRACT_ADDRESS),
        ),
        "abi": abi,
    }


class DetectionEngine:
    def __init__(self, cfg: BehaviorConfig, db: BehaviorDatabase, ethereum: EthereumClient):
        self.cfg = cfg
        self.db = db
        self.ethereum = ethereum
        self.lock = threading.Lock()
        self.running = False
        self.frame_jpeg: Optional[bytes] = None
        self.current_status = "Starting"
        self.current_behaviors: Dict[str, bool] = {name: False for name in BEHAVIOR_LABELS}
        self.metrics: Dict[str, float] = {}
        self.fps = 0.0
        self.last_event_at: Dict[str, float] = {}
        self.smoothed_metrics: Dict[str, float] = {}
        self.neutral_nose_down_ratio: Optional[float] = None
        self.neutral_eye_ear: Optional[float] = None
        self.seated_body_span: Optional[float] = None
        self.mouth_history = deque()
        self.last_drowsy_signal_at: float = 0.0
        self.last_drowsy_attention: Dict[str, bool] = {"eyes_closed": False, "head_down": False}
        self.classifier = ConversationClassifier(cfg)
        self.led = LedController(cfg)
        self.audio_state: Dict[str, object] = {
            "enabled": False,
            "listening": False,
            "last_text": "",
            "classification": "idle",
            "confidence": 0.0,
            "academic_score": 0,
            "private_score": 0,
            "reason": "",
            "error": "",
            "updated_at": "",
            "led_on": False,
            "led_mode": self.led.mode,
        }
        self.last_audio_talking_at: float = 0.0
        self.last_audio_private_at: float = 0.0

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        thread = threading.Thread(target=self._run_safe, daemon=True)
        thread.start()

    def stop(self) -> None:
        self.running = False

    def snapshot_status(self) -> Dict:
        with self.lock:
            self._refresh_audio_led_locked()
            behaviors = self.current_behaviors.copy()
            if self._audio_talking_active(time.monotonic()):
                behaviors["Talking"] = True
            return {
                "running": self.running,
                "status": self.current_status,
                "behaviors": behaviors,
                "metrics": self.metrics.copy(),
                "fps": round(self.fps, 1),
                "yolo": self.metrics.get("yolo_enabled", 0) == 1,
                "audio": self.audio_state.copy(),
            }

    def get_frame(self) -> Optional[bytes]:
        with self.lock:
            return self.frame_jpeg

    def submit_transcript(self, text: str) -> Dict:
        clean_text = " ".join(str(text or "").strip().split())
        if not clean_text:
            return {"ok": False, "error": "empty_text"}

        now = time.monotonic()
        result = self.classifier.classify(clean_text)
        label = str(result["label"])
        confidence = float(result.get("confidence", 0.0))
        with self.lock:
            self.last_audio_talking_at = now
            if label == "private" and confidence >= self.cfg.openai_private_confidence:
                self.last_audio_private_at = now
            self.audio_state.update(
                {
                    "enabled": True,
                    "listening": True,
                    "last_text": clean_text,
                    "classification": label,
                    "confidence": confidence,
                    "academic_score": int(result["academic_score"]),
                    "private_score": int(result["private_score"]),
                    "reason": str(result.get("reason", "")),
                    "error": str(result.get("error", "")),
                    "updated_at": datetime.now().replace(microsecond=0).isoformat(sep=" "),
                }
            )
            self._refresh_audio_led_locked()
            return {"ok": True, "audio": self.audio_state.copy()}

    def _audio_talking_active(self, now: float) -> bool:
        return now - self.last_audio_talking_at <= self.cfg.audio_talking_hold_seconds

    def _audio_private_active(self, now: float) -> bool:
        return now - self.last_audio_private_at <= self.cfg.audio_private_hold_seconds

    def _refresh_audio_led_locked(self) -> None:
        led_on = self._audio_private_active(time.monotonic())
        self.led.set(led_on)
        self.audio_state["led_on"] = led_on
        self.audio_state["led_mode"] = self.led.mode

    def _run_safe(self) -> None:
        try:
            self._run()
        except Exception as exc:
            with self.lock:
                self.current_status = f"Camera error: {exc}"
                self.frame_jpeg = self._status_frame(self.current_status)
            self.running = False

    def _status_frame(self, message: str) -> bytes:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, message[:70], (48, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (245, 245, 245), 2)
        cv2.putText(frame, "Hay tat app/camera khac roi bam Start lai.", (48, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 220, 255), 2)
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return buffer.tobytes() if ok else b""

    def _run(self) -> None:
        with self.lock:
            self.current_status = "Opening camera"
        cap, backend = open_camera(self.cfg)
        if cap is None:
            with self.lock:
                self.current_status = f"Cannot open webcam {self.cfg.camera_index}"
                self.frame_jpeg = self._status_frame(self.current_status)
            self.running = False
            return

        with self.lock:
            self.current_status = f"Camera opened ({backend})"
            self.frame_jpeg = self._status_frame(f"Camera opened ({backend})")
        face_mesh = create_face_mesh()
        with self.lock:
            self.current_status = "FaceMesh ready" if face_mesh is not None else "FaceMesh OFF"
        pose = create_pose()
        with self.lock:
            self.current_status = "Pose ready" if pose is not None else "Pose OFF"
        haar_detector = None if face_mesh is not None else HaarAttentionDetector()
        with self.lock:
            self.current_status = "Loading YOLO" if self.cfg.use_yolo else "YOLO OFF"
        yolo = load_yolo(self.cfg)
        person_yolo = load_person_yolo(self.cfg, yolo)
        with self.lock:
            self.current_status = "Reading camera"
            self.frame_jpeg = self._status_frame("Reading camera")
        eye_closed_timer = HoldTimer(self.cfg.drowsy_eye_seconds, self.cfg.drowsy_signal_grace_seconds)
        head_down_timer = HoldTimer(self.cfg.drowsy_head_seconds, self.cfg.drowsy_signal_grace_seconds)
        distracted_timer = HoldTimer(self.cfg.distracted_seconds)
        phone_timer = HoldTimer(self.cfg.phone_hold_seconds)
        eating_timer = HoldTimer(self.cfg.eating_seconds)
        left_seat_timer = HoldTimer(self.cfg.left_seat_seconds)
        raised_hand_timer = HoldTimer(self.cfg.raised_hand_seconds)
        standing_timer = HoldTimer(self.cfg.standing_seconds)
        talking_timer = HoldTimer(self.cfg.talking_seconds, 0.25)
        fps_meter = FpsMeter()
        frame_index = 0
        phone_boxes = []
        person_boxes = []
        behavior_boxes = []
        yolo_behavior_signals = {name: False for name in BEHAVIOR_LABELS}
        phone_signal = False
        person_seen = False
        read_failures = 0

        while self.running:
            ok, frame = cap.read()
            if not ok:
                read_failures += 1
                if read_failures >= self.cfg.camera_read_fail_limit:
                    with self.lock:
                        self.current_status = f"Cannot read webcam {self.cfg.camera_index} ({backend})"
                        self.frame_jpeg = self._status_frame(self.current_status)
                    self.running = False
                    break
                time.sleep(0.05)
                continue
            read_failures = 0

            frame = cv2.flip(frame, 1)
            if frame_index == 0:
                ok_preview, preview_buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if ok_preview:
                    with self.lock:
                        self.frame_jpeg = preview_buffer.tobytes()
                        self.current_status = "Processing frame"
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.monotonic()
            frame_index += 1
            face_seen, attention, metrics, face_boxes = self._analyze_face(frame, rgb, face_mesh, haar_detector)
            if face_mesh is not None and face_seen:
                attention, metrics = self._stabilize_attention(attention, metrics)
            elif not face_seen:
                self.smoothed_metrics = {}
                self.neutral_nose_down_ratio = None
                self.neutral_eye_ear = None

            if yolo is not None and frame_index % max(1, self.cfg.phone_detect_every) == 0:
                person_seen, phone_signal, phone_boxes, person_boxes, yolo_behavior_signals, behavior_boxes = detect_yolo(
                    yolo,
                    frame,
                    self.cfg,
                )
                if person_yolo is not None and person_yolo is not yolo:
                    person_seen_coco, phone_signal_coco, phone_boxes_coco, person_boxes_coco, _, _ = detect_yolo(
                        person_yolo,
                        frame,
                        self.cfg,
                    )
                    person_seen = person_seen or person_seen_coco
                    phone_signal = phone_signal or phone_signal_coco
                    phone_boxes = phone_boxes_coco or phone_boxes
                    person_boxes = person_boxes_coco or person_boxes

            pose_seen, pose_signals, pose_metrics, pose_points = self._analyze_pose(frame, rgb, pose, person_boxes)
            metrics.update(pose_metrics)
            visible_person = bool(face_seen or person_seen or pose_seen)
            eyes_closed_signal = bool(attention.get("eyes_closed"))
            head_down_signal = bool(attention.get("head_down"))
            if face_seen and (eyes_closed_signal or head_down_signal):
                self.last_drowsy_signal_at = now
                self.last_drowsy_attention = {
                    "eyes_closed": eyes_closed_signal,
                    "head_down": head_down_signal,
                }
            elif not face_seen and visible_person and now - self.last_drowsy_signal_at <= self.cfg.face_lost_drowsy_grace_seconds:
                eyes_closed_signal = eyes_closed_signal or self.last_drowsy_attention.get("eyes_closed", False)
                head_down_signal = head_down_signal or self.last_drowsy_attention.get("head_down", False)

            face_lost_distracted = self.cfg.distracted_face_lost and not face_seen and bool(person_seen or pose_seen)
            distracted_signal = (bool(attention.get("looking_away")) and visible_person) or face_lost_distracted
            metrics["face_seen"] = 1.0 if face_seen else 0.0
            metrics["person_seen"] = 1.0 if person_seen else 0.0
            metrics["person_count"] = float(max(len(person_boxes), 1 if face_seen else 0))
            metrics["face_lost_distracted"] = 1.0 if face_lost_distracted else 0.0
            eyes_closed_active = eye_closed_timer.update(eyes_closed_signal, now)
            head_down_active = head_down_timer.update(head_down_signal, now)
            sleeping = eyes_closed_active or head_down_active
            metrics["eyes_closed_signal"] = 1.0 if eyes_closed_signal else 0.0
            metrics["head_down_signal"] = 1.0 if head_down_signal else 0.0
            metrics["eyes_closed_active"] = 1.0 if eyes_closed_active else 0.0
            metrics["head_down_active"] = 1.0 if head_down_active else 0.0
            talking_signal = self._detect_talking(face_seen, metrics, person_boxes, distracted_signal, now)
            statuses = {
                "Eating": eating_timer.update(yolo_behavior_signals.get("Eating", False), now),
                "Sleeping": sleeping or yolo_behavior_signals.get("Sleeping", False),
                "Using Phone": phone_timer.update(phone_signal or yolo_behavior_signals.get("Using Phone", False), now),
                "Distracted": distracted_timer.update(distracted_signal, now),
                "Left Seat": left_seat_timer.update(not visible_person, now),
                "Raised Hand": raised_hand_timer.update(
                    pose_signals.get("raised_hand", False) or yolo_behavior_signals.get("Raised Hand", False),
                    now,
                ),
                "Standing Up": standing_timer.update(
                    pose_signals.get("standing", False) or yolo_behavior_signals.get("Standing Up", False),
                    now,
                ),
                "Talking": talking_timer.update(talking_signal, now),
            }
            if self._audio_talking_active(now):
                statuses["Talking"] = True

            self._draw(
                frame,
                statuses,
                metrics,
                face_boxes,
                phone_boxes,
                person_boxes,
                behavior_boxes,
                pose_points,
                yolo is not None,
                pose is not None,
            )
            self._record_new_events(statuses, frame, now)
            fps = fps_meter.update()
            metrics["yolo_enabled"] = 1.0 if yolo is not None or person_yolo is not None else 0.0
            metrics["pose_enabled"] = 1.0 if pose is not None else 0.0
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                with self.lock:
                    self._refresh_audio_led_locked()
                    self.frame_jpeg = buffer.tobytes()
                    self.current_behaviors = statuses.copy()
                    self.current_status = self._status_text(statuses)
                    self.metrics = metrics.copy()
                    self.fps = fps

        cap.release()
        if face_mesh is not None:
            face_mesh.close()
        if pose is not None:
            pose.close()
        self.running = False

    def _analyze_face(self, frame, rgb, face_mesh, haar_detector):
        if face_mesh is not None:
            height, width = frame.shape[:2]
            result = face_mesh.process(rgb)
            if not result.multi_face_landmarks:
                return False, {"eyes_closed": False, "head_down": False, "looking_away": False}, {}, []
            landmarks = result.multi_face_landmarks[0].landmark
            attention = estimate_attention(landmarks, width, height, self.cfg)
            metrics = {k: float(v) for k, v in attention.items() if isinstance(v, float)}
            for idx in [NOSE_TIP, LEFT_FACE, RIGHT_FACE, FOREHEAD, CHIN]:
                cv2.circle(frame, point(landmarks, idx, width, height), 2, (255, 255, 255), -1)
            return True, attention, metrics, []
        return haar_detector.process(frame, self.cfg)

    def _analyze_pose(self, frame, rgb, pose, person_boxes) -> Tuple[bool, Dict[str, bool], Dict[str, float], list]:
        height, width = frame.shape[:2]
        box_standing = self.cfg.use_box_standing and self._standing_from_person_boxes(person_boxes, width, height)
        if pose is None:
            return bool(person_boxes), {"raised_hand": False, "standing": box_standing}, {"standing_box": float(box_standing)}, []

        result = pose.process(rgb)
        if not result.pose_landmarks:
            return bool(person_boxes), {"raised_hand": False, "standing": box_standing}, {"standing_box": float(box_standing)}, []

        landmarks = result.pose_landmarks.landmark
        pose_lm = mp.solutions.pose.PoseLandmark

        def visible(item) -> bool:
            return landmarks[item.value].visibility >= self.cfg.pose_visibility

        def lm(item):
            value = landmarks[item.value]
            return value.x, value.y, value.visibility

        shoulder_items = [pose_lm.LEFT_SHOULDER, pose_lm.RIGHT_SHOULDER]
        hip_items = [pose_lm.LEFT_HIP, pose_lm.RIGHT_HIP]
        knee_items = [pose_lm.LEFT_KNEE, pose_lm.RIGHT_KNEE]
        ankle_items = [pose_lm.LEFT_ANKLE, pose_lm.RIGHT_ANKLE]
        wrist_items = [pose_lm.LEFT_WRIST, pose_lm.RIGHT_WRIST]

        shoulders = [lm(item) for item in shoulder_items if visible(item)]
        hips = [lm(item) for item in hip_items if visible(item)]
        knees = [lm(item) for item in knee_items if visible(item)]
        ankles = [lm(item) for item in ankle_items if visible(item)]
        wrists = [lm(item) for item in wrist_items if visible(item)]
        points = []
        for item in shoulder_items + hip_items + knee_items + ankle_items + wrist_items:
            if visible(item):
                value = landmarks[item.value]
                points.append((int(value.x * width), int(value.y * height)))

        raised_hand = False
        if shoulders:
            shoulder_line_y = min(item[1] for item in shoulders)
            raised_hand = any(wrist[1] < shoulder_line_y - self.cfg.raised_hand_margin for wrist in wrists)

        standing_pose = False
        body_span = 0.0
        if shoulders and hips:
            shoulder_y = sum(item[1] for item in shoulders) / len(shoulders)
            hip_y = sum(item[1] for item in hips) / len(hips)
            lower_body = knees + ankles
            visible_body = shoulders + hips + lower_body
            body_span = max(item[1] for item in visible_body) - min(item[1] for item in visible_body)
            legs_below_hips = bool(lower_body) and max(item[1] for item in lower_body) > hip_y + 0.12
            upright_torso = hip_y > shoulder_y + 0.12
            baseline_standing = self.seated_body_span is not None and body_span > self.seated_body_span + self.cfg.standing_baseline_delta
            absolute_standing = body_span > self.cfg.standing_body_span_ratio
            standing_pose = upright_torso and legs_below_hips and (absolute_standing or baseline_standing)
            if not standing_pose and body_span > 0.18:
                if self.seated_body_span is None:
                    self.seated_body_span = body_span
                else:
                    self.seated_body_span = (self.seated_body_span * 0.98) + (body_span * 0.02)

        metrics = {
            "raised_hand_signal": float(raised_hand),
            "standing_pose": float(standing_pose),
            "standing_box": float(box_standing),
            "body_span": float(body_span),
            "seated_body_span": float(self.seated_body_span or 0.0),
        }
        return True, {"raised_hand": raised_hand, "standing": standing_pose or box_standing}, metrics, points

    def _detect_talking(self, face_seen: bool, metrics: Dict[str, float], person_boxes, distracted_signal: bool, now: float) -> bool:
        mouth_ratio = float(metrics.get("mouth_open_ratio", 0.0))
        if face_seen:
            self.mouth_history.append((now, mouth_ratio))
        while self.mouth_history and now - self.mouth_history[0][0] > self.cfg.talking_window_seconds:
            self.mouth_history.popleft()

        mouth_values = [value for _, value in self.mouth_history]
        mouth_delta = (max(mouth_values) - min(mouth_values)) if len(mouth_values) >= 3 else 0.0
        mouth_moving = mouth_ratio > self.cfg.talking_mouth_ratio and mouth_delta > self.cfg.talking_mouth_delta
        turned_sideways = float(metrics.get("yaw_ratio", 0.0)) > self.cfg.talking_turn_yaw_ratio
        has_nearby_person = len(person_boxes) >= 2
        talking = mouth_moving and (turned_sideways or has_nearby_person or distracted_signal)
        metrics["mouth_movement_delta"] = float(mouth_delta)
        metrics["talking_signal"] = 1.0 if talking else 0.0
        return talking

    def _standing_from_person_boxes(self, person_boxes, width: int, height: int) -> bool:
        if not person_boxes:
            return False
        for x1, y1, x2, y2 in person_boxes:
            box_width = max(x2 - x1, 1)
            box_height = max(y2 - y1, 1)
            height_ratio = box_height / max(height, 1)
            aspect_ratio = box_height / box_width
            if height_ratio >= self.cfg.standing_person_height_ratio and aspect_ratio >= self.cfg.standing_person_aspect_ratio:
                return True
        return False

    def _stabilize_attention(self, attention: Dict[str, bool], metrics: Dict[str, float]) -> Tuple[Dict[str, bool], Dict[str, float]]:
        if not metrics:
            return attention, metrics

        alpha = 0.35
        for key in ("ear", "yaw_ratio", "nose_down_ratio"):
            value = metrics.get(key)
            if value is None:
                continue
            previous = self.smoothed_metrics.get(key)
            self.smoothed_metrics[key] = value if previous is None else (previous * (1.0 - alpha)) + (value * alpha)

        stable = metrics.copy()
        stable.update(self.smoothed_metrics)
        ear = stable.get("ear", 0.0)
        yaw_ratio = stable.get("yaw_ratio", 0.0)
        nose_down_ratio = stable.get("nose_down_ratio", 0.0)

        if ear > 0.12 and not bool(attention.get("head_down")):
            if self.neutral_eye_ear is None:
                self.neutral_eye_ear = ear
            elif ear > self.neutral_eye_ear:
                self.neutral_eye_ear = (self.neutral_eye_ear * 0.85) + (ear * 0.15)
            else:
                self.neutral_eye_ear = (self.neutral_eye_ear * 0.995) + (ear * 0.005)

        adaptive_eye_threshold = self.cfg.eye_closed_ear
        if self.neutral_eye_ear is not None:
            adaptive_eye_threshold = min(adaptive_eye_threshold, self.neutral_eye_ear * self.cfg.eye_closed_adaptive_ratio)

        if ear >= adaptive_eye_threshold:
            if self.neutral_nose_down_ratio is None:
                self.neutral_nose_down_ratio = nose_down_ratio
            elif nose_down_ratio < self.cfg.head_down_ratio + 0.12:
                self.neutral_nose_down_ratio = (self.neutral_nose_down_ratio * 0.995) + (nose_down_ratio * 0.005)

        adaptive_head_threshold = self.cfg.head_down_ratio
        if self.neutral_nose_down_ratio is not None:
            adaptive_head_threshold = max(adaptive_head_threshold, self.neutral_nose_down_ratio + self.cfg.head_down_delta)

        stable["neutral_eye_ear"] = float(self.neutral_eye_ear or 0.0)
        stable["eye_closed_threshold"] = float(adaptive_eye_threshold)
        stable["neutral_nose_down_ratio"] = float(self.neutral_nose_down_ratio or 0.0)
        stable["head_down_threshold"] = float(adaptive_head_threshold)
        return {
            "eyes_closed": ear < adaptive_eye_threshold,
            "head_down": nose_down_ratio > adaptive_head_threshold,
            "looking_away": yaw_ratio > self.cfg.distracted_yaw_ratio,
        }, stable

    def _record_new_events(self, statuses: Dict[str, bool], frame, now: float) -> None:
        for behavior, active in statuses.items():
            if not active:
                continue
            if now - self.last_event_at.get(behavior, 0) < self.cfg.event_cooldown:
                continue
            self.last_event_at[behavior] = now
            timestamp = datetime.now().replace(microsecond=0).isoformat(sep=" ")
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{behavior.lower().replace(' ', '_')}.jpg"
            image_path = SNAPSHOT_DIR / filename
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), frame)
            relative_image = str(image_path.relative_to(BASE_DIR)).replace("\\", "/")
            hash_data = compute_hash(behavior, timestamp, image_path)
            event_id = self.db.insert_event(behavior, timestamp, relative_image, hash_data)
            try:
                ethereum_tx = self.ethereum.submit_hash(event_id, hash_data, timestamp)
            except Exception as exc:
                ethereum_tx = f"ethereum-error:{exc}"
            self.db.update_ethereum_tx(event_id, ethereum_tx)

    def _draw(self, frame, statuses, metrics, face_boxes, phone_boxes, person_boxes, behavior_boxes, pose_points, yolo_enabled, pose_enabled) -> None:
        for x, y, w, h, eyes in face_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 1)
            for ex, ey, ew, eh in eyes[:2]:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 255, 255), 1)
        for x1, y1, x2, y2 in person_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 160, 80), 1)
            cv2.putText(frame, "person", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 160, 80), 1)
        for x1, y1, x2, y2, score in phone_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(frame, f"phone {score:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
        for x1, y1, x2, y2, behavior, score in behavior_boxes:
            label = f"{BEHAVIOR_LABELS.get(behavior, behavior)} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 80, 255), 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)
        for x, y in pose_points:
            cv2.circle(frame, (x, y), 4, (255, 180, 0), -1)

        active = [BEHAVIOR_LABELS[key] for key, value in statuses.items() if value]
        if not active:
            active = ["Binh thuong"]
        y = 34
        for label in active:
            cv2.putText(frame, label, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if label != "Binh thuong" else (0, 180, 0), 2)
            y += 30
        footer = (
            f"EAR={metrics.get('ear', 0):.2f}/{metrics.get('eye_closed_threshold', self.cfg.eye_closed_ear):.2f} "
            f"Yaw={metrics.get('yaw_ratio', 0):.2f} "
            f"Down={metrics.get('nose_down_ratio', 0):.2f}/{metrics.get('head_down_threshold', self.cfg.haar_head_down_ratio):.2f} "
            f"Mouth={metrics.get('mouth_open_ratio', 0):.2f} "
            f"Pose={'ON' if pose_enabled else 'OFF'} "
            f"YOLO={'ON' if yolo_enabled else 'OFF'}"
        )
        cv2.putText(frame, footer, (16, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)

    @staticmethod
    def _status_text(statuses: Dict[str, bool]) -> str:
        active = [BEHAVIOR_LABELS[key] for key, value in statuses.items() if value]
        return ", ".join(active) if active else "Binh thuong"


HTML = """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BEHAVIOR AI</title>
  <script src="https://cdn.jsdelivr.net/npm/ethers@6.13.4/dist/ethers.umd.min.js"></script>
  <style>
    :root {
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d8dee9;
      --accent: #0f766e;
      --warn: #b42318;
      --blue: #175cd3;
      --amber: #b54708;
      --shadow: 0 14px 40px rgba(23, 32, 51, .10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif;
      letter-spacing: 0;
    }
    header {
      background: #111827;
      color: white;
      border-bottom: 1px solid #263244;
    }
    .wrap { width: min(1180px, calc(100vw - 32px)); margin: 0 auto; }
    .topbar { min-height: 76px; display: flex; align-items: center; justify-content: space-between; gap: 18px; }
    h1 { margin: 0; font-size: 22px; font-weight: 760; }
    .sub { margin-top: 4px; color: #b7c0d1; font-size: 13px; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; }
    button {
      height: 38px; border: 0; border-radius: 7px; padding: 0 14px; font-weight: 700; cursor: pointer;
      background: var(--accent); color: white;
    }
    button.secondary { background: #344054; }
    main { padding: 24px 0 34px; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(330px, .75fr); gap: 18px; align-items: start; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); overflow: hidden; }
    .video { aspect-ratio: 16 / 9; background: #0b1020; display: grid; place-items: center; }
    .video img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .panel-head { padding: 14px 16px; border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; align-items: center; gap: 10px; }
    .panel-title { font-size: 14px; font-weight: 760; text-transform: uppercase; color: #344054; }
    .status-pill { display: inline-flex; align-items: center; min-height: 30px; border-radius: 999px; padding: 5px 11px; background: #ecfdf3; color: #067647; font-size: 13px; font-weight: 750; }
    .status-pill.alert { background: #fef3f2; color: var(--warn); }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--line); }
    .metric { background: white; padding: 13px 14px; min-width: 0; }
    .metric span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }
    .metric strong { font-size: 18px; }
    .cards { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; padding: 14px; }
    .behavior { border: 1px solid var(--line); border-radius: 8px; padding: 13px; min-height: 76px; display: flex; flex-direction: column; justify-content: space-between; }
    .behavior strong { font-size: 15px; }
    .behavior span { font-size: 12px; color: var(--muted); }
    .behavior.active { border-color: #fda29b; background: #fff4f2; }
    .events { margin-top: 18px; }
    table { width: 100%; border-collapse: collapse; table-layout: fixed; }
    th, td { padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; font-size: 13px; }
    th { color: #475467; font-size: 12px; text-transform: uppercase; background: #f8fafc; }
    td.hash { font-family: Consolas, ui-monospace, monospace; font-size: 12px; overflow-wrap: anywhere; color: #344054; }
    .thumb { width: 72px; aspect-ratio: 16 / 10; border-radius: 6px; object-fit: cover; border: 1px solid var(--line); background: #eef2f6; }
    .chain { display: block; color: var(--blue); overflow-wrap: anywhere; font-family: Consolas, ui-monospace, monospace; font-size: 12px; }
    .wallet { color: #d0d5dd; font-size: 12px; align-self: center; font-family: Consolas, ui-monospace, monospace; }
    .mini-btn { height: 32px; border-radius: 6px; padding: 0 10px; font-size: 12px; white-space: nowrap; }
    .mini-btn:disabled { cursor: not-allowed; opacity: .55; }
    .command-strip { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }
    .kpi { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); padding: 15px 16px; min-height: 104px; display: flex; flex-direction: column; justify-content: space-between; }
    .kpi span { color: var(--muted); font-size: 12px; text-transform: uppercase; font-weight: 750; }
    .kpi strong { font-size: 30px; line-height: 1; }
    .kpi small { color: var(--muted); font-size: 12px; }
    .kpi.danger { border-color: #fecdca; background: #fff7f7; }
    .kpi.warn { border-color: #fedf89; background: #fffbeb; }
    .risk { display: flex; align-items: center; gap: 12px; }
    .ring { width: 58px; height: 58px; border-radius: 50%; display: grid; place-items: center; background: conic-gradient(var(--warn) var(--risk, 0deg), #e8edf5 0deg); font-weight: 850; }
    .ring::before { content: ""; position: absolute; }
    .ops-grid { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 18px; margin-top: 18px; align-items: stretch; }
    .chart-wrap { padding: 14px 16px 16px; }
    .chart-canvas { width: 100%; height: 220px; display: block; border: 1px solid #eef2f6; border-radius: 8px; background: #fbfdff; }
    .bars { padding: 4px 16px 16px; display: grid; gap: 11px; }
    .bar-row { display: grid; grid-template-columns: 128px minmax(0, 1fr) 34px; align-items: center; gap: 10px; font-size: 13px; }
    .bar-track { height: 9px; border-radius: 999px; background: #edf1f7; overflow: hidden; }
    .bar-fill { height: 100%; width: 0%; border-radius: inherit; background: var(--blue); transition: width .25s ease; }
    .alert-stack { padding: 14px; display: grid; gap: 10px; border-bottom: 1px solid var(--line); }
    .alert-item { border: 1px solid #fecdca; background: #fff4f2; color: var(--warn); border-radius: 8px; padding: 11px 12px; font-weight: 760; display: flex; justify-content: space-between; gap: 10px; }
    .alert-item.idle { border-color: #d1fadf; background: #f0fdf4; color: #067647; }
    .alert-item small { color: inherit; opacity: .72; font-weight: 650; white-space: nowrap; }
    .toast { position: fixed; right: 24px; top: 96px; z-index: 20; display: grid; gap: 10px; width: min(360px, calc(100vw - 32px)); }
    .toast-card { border-radius: 8px; background: #b42318; color: white; box-shadow: 0 18px 50px rgba(180, 35, 24, .28); padding: 14px 16px; font-weight: 780; animation: slideIn .18s ease-out; }
    .toast-card span { display: block; margin-top: 4px; font-size: 12px; opacity: .86; font-weight: 650; }
    .system-health { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: var(--line); }
    .health-cell { background: #fff; padding: 13px 14px; }
    .health-cell span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 5px; }
    .health-cell strong { font-size: 14px; }
    .audio-grid { display: grid; grid-template-columns: minmax(0, 1fr) 180px 180px; gap: 1px; background: var(--line); }
    .audio-cell { background: white; padding: 14px 16px; min-height: 82px; }
    .audio-cell > span { display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; font-weight: 750; margin-bottom: 7px; }
    .audio-cell strong { display: block; font-size: 16px; overflow-wrap: anywhere; }
    .audio-led { width: 20px; height: 20px; border-radius: 50%; background: #98a2b3; box-shadow: inset 0 0 0 1px rgba(0,0,0,.14); }
    .audio-led.on { background: #f04438; box-shadow: 0 0 0 6px rgba(240,68,56,.14), 0 0 20px rgba(240,68,56,.55); }
    @keyframes slideIn { from { transform: translateY(-8px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
      .command-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .ops-grid { grid-template-columns: 1fr; }
      .audio-grid { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .topbar { align-items: flex-start; flex-direction: column; padding: 16px 0; }
      table { min-width: 820px; }
      .table-scroll { overflow-x: auto; }
    }
  </style>
</head>
<body>
  <div class="toast" id="toast"></div>
  <header>
    <div class="wrap topbar">
      <div>
        <h1>BEHAVIOR AI</h1>
      </div>
      <div class="actions">
        <button onclick="post('/api/start')">Start</button>
        <button class="secondary" onclick="post('/api/stop')">Stop</button>
        <button class="secondary" onclick="toggleMic()" id="micButton">Start mic</button>
        <button class="secondary" onclick="connectWallet()">Connect MetaMask</button>
        <span class="wallet" id="walletStatus">MetaMask chua ket noi</span>
      </div>
    </div>
  </header>
  <main class="wrap">
    <section class="command-strip">
      <div class="kpi">
        <span>Tong hoc sinh</span>
        <strong id="studentCount">0</strong>
      </div>
      <div class="kpi danger">
        <span>Dang ngu gat</span>
        <strong id="sleepingNow">0</strong>
      </div>
      <div class="kpi warn">
        <span>Dung dien thoai</span>
        <strong id="phoneNow">0</strong>
      </div>
      <div class="kpi">
        <span>Mat tap trung</span>
        <strong id="distractedNow">0</strong>
      </div>
      <div class="kpi">
        <span>Risk score</span>
        <div class="risk"><div class="ring" id="riskRing">0</div><small id="riskText">Lop on dinh</small></div>
      </div>
    </section>
    <section class="grid">
      <div class="panel">
        <div class="panel-head">
          <div class="panel-title">Webcam realtime</div>
          <div id="statusPill" class="status-pill">Dang khoi dong</div>
        </div>
        <div class="video"><img src="/video_feed" alt="Realtime webcam stream"></div>
        <div class="metrics">
          <div class="metric"><span>FPS</span><strong id="fps">0</strong></div>
          <div class="metric"><span>EAR</span><strong id="ear">0.00</strong></div>
          <div class="metric"><span>Yaw</span><strong id="yaw">0.00</strong></div>
          <div class="metric"><span>Head down</span><strong id="down">0.00</strong></div>
          <div class="metric"><span>Mouth</span><strong id="mouth">0.00</strong></div>
          <div class="metric"><span>Audio</span><strong id="audioHealth">OFF</strong></div>
          <div class="metric"><span>LED</span><strong id="ledHealth">OFF</strong></div>
          <div class="metric"><span>Pose</span><strong id="pose">OFF</strong></div>
          <div class="metric"><span>YOLO</span><strong id="yolo">OFF</strong></div>
        </div>
      </div>
      <aside class="panel">
        <div class="panel-head"><div class="panel-title">Canh bao realtime</div></div>
        <div class="alert-stack" id="alertStack"></div>
        <div class="panel-head"><div class="panel-title">Trang thai hanh vi</div></div>
        <div class="cards" id="behaviorCards"></div>
      </aside>
    </section>
    <section class="panel events">
      <div class="panel-head">
        <div class="panel-title">Nhan dien noi dung am thanh</div>
        <span class="status-pill" id="audioMode">Mic OFF</span>
      </div>
      <div class="audio-grid">
        <div class="audio-cell"><span>Text nghe duoc</span><strong id="audioText">Chua co transcript</strong></div>
        <div class="audio-cell"><span>AI phan loai</span><strong id="audioClass">Idle</strong></div>
        <div class="audio-cell"><span>Den canh bao</span><strong style="display:flex;align-items:center;gap:10px"><i id="audioLed" class="audio-led"></i><span id="audioLedText">OFF</span></strong></div>
      </div>
    </section>
    <section class="ops-grid">
      <div class="panel">
        <div class="panel-head">
          <div class="panel-title">Bieu do hanh vi theo thoi gian</div>
          <span class="status-pill" id="chartMode">Live window</span>
        </div>
        <div class="chart-wrap"><canvas class="chart-canvas" id="behaviorChart" width="900" height="260"></canvas></div>
      </div>
      <div class="panel">
        <div class="panel-head"><div class="panel-title">Phan bo hanh vi</div></div>
        <div class="bars" id="behaviorBars"></div>
        <div class="system-health">
          <div class="health-cell"><span>Camera</span><strong id="cameraHealth">OFF</strong></div>
          <div class="health-cell"><span>Pose</span><strong id="poseHealth">OFF</strong></div>
          <div class="health-cell"><span>YOLO</span><strong id="yoloHealth">OFF</strong></div>
        </div>
      </div>
    </section>
    <section class="panel events">
      <div class="panel-head">
        <div class="panel-title">Danh sach vi pham va hash blockchain</div>
        <span class="status-pill" id="chainMode">Ethereum ready</span>
      </div>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th style="width:70px">ID</th>
              <th style="width:145px">Hanh vi</th>
              <th style="width:170px">Thoi gian</th>
              <th style="width:100px">Anh</th>
              <th>SHA256</th>
              <th style="width:190px">Ethereum</th>
              <th style="width:130px">MetaMask</th>
            </tr>
          </thead>
          <tbody id="eventsBody"></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    const labels = {
      "Eating": "An uong",
      "Sleeping": "Ngu gat",
      "Using Phone": "Dung dien thoai",
      "Distracted": "Mat tap trung",
      "Left Seat": "Roi khoi vi tri",
      "Raised Hand": "Gio tay phat bieu",
      "Standing Up": "Dung day",
      "Talking": "Noi chuyen"
    };
    const behaviorOrder = ["Eating", "Sleeping", "Using Phone", "Distracted", "Talking", "Raised Hand", "Standing Up", "Left Seat"];
    const severity = {
      "Eating": 10,
      "Sleeping": 32,
      "Using Phone": 30,
      "Distracted": 18,
      "Talking": 12,
      "Standing Up": 10,
      "Raised Hand": 4,
      "Left Seat": 16
    };
    const chartSeries = [];
    let lastAlertKey = "";
    let chainConfig = null;
    let walletAddress = "";
    let speechRecognition = null;
    let micListening = false;

    async function post(url) {
      await fetch(url, { method: "POST" });
      await refresh();
    }
    async function loadChainConfig() {
      if (chainConfig) return chainConfig;
      const res = await fetch("/api/chain_config");
      chainConfig = await res.json();
      return chainConfig;
    }
    function shortAddress(value) {
      return value ? `${value.slice(0, 6)}...${value.slice(-4)}` : "";
    }
    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }[char]));
    }
    function audioLabel(value) {
      if (value === "private") return "Noi chuyen rieng";
      if (value === "study") return "Hoc tap";
      if (value === "unknown") return "Chua ro";
      return "Idle";
    }
    function setupSpeechRecognition() {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) return null;
      const recognition = new SpeechRecognition();
      recognition.lang = "vi-VN";
      recognition.continuous = true;
      recognition.interimResults = false;
      recognition.onresult = async (event) => {
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (!event.results[i].isFinal) continue;
          const text = event.results[i][0].transcript.trim();
          if (!text) continue;
          await fetch("/api/audio_transcript", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
          });
        }
        await refresh();
      };
      recognition.onend = () => {
        if (micListening) recognition.start();
      };
      recognition.onerror = () => {
        document.getElementById("audioMode").textContent = "Mic error";
        document.getElementById("audioMode").className = "status-pill alert";
      };
      return recognition;
    }
    function toggleMic() {
      if (!speechRecognition) speechRecognition = setupSpeechRecognition();
      if (!speechRecognition) {
        alert("Trinh duyet chua ho tro Web Speech API. Hay dung Chrome hoac Edge tren localhost/HTTPS.");
        return;
      }
      micListening = !micListening;
      if (micListening) {
        speechRecognition.start();
      } else {
        speechRecognition.stop();
      }
      document.getElementById("micButton").textContent = micListening ? "Stop mic" : "Start mic";
      document.getElementById("audioMode").textContent = micListening ? "Mic ON" : "Mic OFF";
      document.getElementById("audioMode").className = "status-pill" + (micListening ? "" : " alert");
    }
    async function connectWallet() {
      if (!window.ethereum) {
        alert("Chua cai MetaMask tren trinh duyet.");
        return;
      }
      const config = await loadChainConfig();
      const accounts = await window.ethereum.request({ method: "eth_requestAccounts" });
      walletAddress = accounts[0] || "";
      document.getElementById("walletStatus").textContent = walletAddress ? shortAddress(walletAddress) : "MetaMask chua ket noi";
      const chainHex = await window.ethereum.request({ method: "eth_chainId" });
      const currentChain = parseInt(chainHex, 16);
      if (config.chain_id && currentChain !== config.chain_id) {
        alert(`MetaMask dang o chain ${currentChain}. Hay chuyen sang chain ${config.chain_id}.`);
      }
    }
    function toUnixTimestamp(value) {
      const parsed = new Date(String(value).replace(" ", "T"));
      const ms = parsed.getTime();
      return Number.isNaN(ms) ? Math.floor(Date.now() / 1000) : Math.floor(ms / 1000);
    }
    async function sendMetamask(eventId) {
      if (!window.ethereum) {
        alert("Chua cai MetaMask tren trinh duyet.");
        return;
      }
      const config = await loadChainConfig();
      if (!config.contract_address) {
        alert("Chua cau hinh METAMASK_CONTRACT_ADDRESS hoac ETHEREUM_CONTRACT_ADDRESS.");
        return;
      }
      if (!window.ethers) {
        alert("Khong tai duoc ethers.js. Kiem tra ket noi internet cua trinh duyet.");
        return;
      }
      if (!walletAddress) {
        await connectWallet();
      }
      const eventRes = await fetch(`/api/events/${eventId}`);
      const row = await eventRes.json();
      const provider = new ethers.BrowserProvider(window.ethereum);
      const signer = await provider.getSigner();
      const contract = new ethers.Contract(config.contract_address, config.abi, signer);
      const tx = await contract.storeBehaviorHash(String(row.id), row.hash, toUnixTimestamp(row.timestamp));
      await fetch(`/api/events/${eventId}/metamask_tx`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tx_hash: tx.hash })
      });
      await refresh();
    }
    function fmt(n) {
      return Number(n || 0).toFixed(2);
    }
    function activeCount(status, key) {
      return status.behaviors && status.behaviors[key] ? 1 : 0;
    }
    function riskScore(behaviors) {
      return Math.min(100, Object.entries(behaviors || {}).reduce((sum, [key, active]) => sum + (active ? (severity[key] || 8) : 0), 0));
    }
    function behaviorCounts(events) {
      const counts = Object.fromEntries(Object.keys(labels).map(key => [key, 0]));
      events.forEach(row => {
        if (counts[row.behavior] !== undefined) counts[row.behavior] += 1;
      });
      return counts;
    }
    function showToast(active) {
      const key = active.join("|");
      if (!active.length || key === lastAlertKey) return;
      lastAlertKey = key;
      const toast = document.getElementById("toast");
      const card = document.createElement("div");
      card.className = "toast-card";
      card.innerHTML = `CANH BAO: ${active.map(key => labels[key] || key).join(", ")}<span>${new Date().toLocaleTimeString("vi-VN")}</span>`;
      toast.prepend(card);
      setTimeout(() => card.remove(), 5200);
    }
    function drawChart() {
      const canvas = document.getElementById("behaviorChart");
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#fbfdff";
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "#e4eaf2";
      ctx.lineWidth = 1;
      for (let i = 0; i < 5; i++) {
        const y = 24 + i * 48;
        ctx.beginPath();
        ctx.moveTo(38, y);
        ctx.lineTo(w - 18, y);
        ctx.stroke();
      }
      const series = ["Sleeping", "Using Phone", "Distracted", "Talking"];
      const colors = {"Sleeping": "#b42318", "Using Phone": "#175cd3", "Distracted": "#b54708", "Talking": "#0f766e"};
      const maxPoints = Math.max(chartSeries.length - 1, 1);
      series.forEach(key => {
        ctx.strokeStyle = colors[key];
        ctx.lineWidth = 3;
        ctx.beginPath();
        chartSeries.forEach((point, idx) => {
          const x = 42 + (idx / maxPoints) * (w - 68);
          const y = h - 34 - (point[key] || 0) * (h - 72);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });
      ctx.font = "12px Segoe UI, Arial";
      let lx = 42;
      series.forEach(key => {
        ctx.fillStyle = colors[key];
        ctx.fillRect(lx, h - 18, 12, 4);
        ctx.fillStyle = "#475467";
        ctx.fillText(labels[key], lx + 18, h - 14);
        lx += 145;
      });
    }
    function updateBars(counts) {
      const max = Math.max(1, ...Object.values(counts));
      document.getElementById("behaviorBars").innerHTML = behaviorOrder.map(key => {
        const value = counts[key] || 0;
        const pct = Math.round((value / max) * 100);
        return `<div class="bar-row"><span>${labels[key]}</span><div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div><strong>${value}</strong></div>`;
      }).join("");
    }
    async function refresh() {
      const [statusRes, eventsRes] = await Promise.all([fetch("/api/status"), fetch("/api/events")]);
      const status = await statusRes.json();
      const events = await eventsRes.json();
      const activeKeys = Object.entries(status.behaviors || {}).filter(([, active]) => active).map(([key]) => key);
      const alerting = activeKeys.length > 0;
      const score = riskScore(status.behaviors);
      const counts = behaviorCounts(events);
      chartSeries.push(Object.fromEntries(Object.keys(labels).map(key => [key, activeCount(status, key)])));
      if (chartSeries.length > 48) chartSeries.shift();
      const pill = document.getElementById("statusPill");
      pill.textContent = status.status;
      pill.className = "status-pill" + (alerting ? " alert" : "");
      document.getElementById("studentCount").textContent = Math.round(status.metrics.person_count || 0);
      document.getElementById("sleepingNow").textContent = activeCount(status, "Sleeping");
      document.getElementById("phoneNow").textContent = activeCount(status, "Using Phone");
      document.getElementById("distractedNow").textContent = activeCount(status, "Distracted");
      const ring = document.getElementById("riskRing");
      ring.textContent = score;
      ring.style.setProperty("--risk", `${Math.round(score * 3.6)}deg`);
      document.getElementById("riskText").textContent = score >= 60 ? "Canh bao cao" : score >= 25 ? "Can theo doi" : "Lop on dinh";
      document.getElementById("fps").textContent = fmt(status.fps);
      document.getElementById("ear").textContent = fmt(status.metrics.ear);
      document.getElementById("yaw").textContent = fmt(status.metrics.yaw_ratio);
      document.getElementById("down").textContent = fmt(status.metrics.nose_down_ratio);
      document.getElementById("mouth").textContent = fmt(status.metrics.mouth_open_ratio);
      const audio = status.audio || {};
      document.getElementById("audioHealth").textContent = micListening ? "ON" : (audio.enabled ? "READY" : "OFF");
      document.getElementById("ledHealth").textContent = audio.led_on ? "ON" : "OFF";
      document.getElementById("pose").textContent = status.metrics.pose_enabled ? "ON" : "OFF";
      document.getElementById("yolo").textContent = status.metrics.yolo_enabled ? "ON" : "OFF";
      document.getElementById("cameraHealth").textContent = status.fps > 0 ? "ONLINE" : "STARTING";
      document.getElementById("poseHealth").textContent = status.metrics.pose_enabled ? "ONLINE" : "OFF";
      document.getElementById("yoloHealth").textContent = status.metrics.yolo_enabled ? "ONLINE" : "OFF";
      document.getElementById("audioText").innerHTML = escapeHtml(audio.last_text || "Chua co transcript");
      const confidence = Math.round((audio.confidence || 0) * 100);
      document.getElementById("audioClass").textContent = audio.error
        ? `GPT loi: ${audio.reason || audio.error}`
        : `${audioLabel(audio.classification)} - ${confidence}%`;
      const led = document.getElementById("audioLed");
      led.className = "audio-led" + (audio.led_on ? " on" : "");
      document.getElementById("audioLedText").textContent = audio.led_on ? `ON - ${audio.led_mode || "virtual"}` : `OFF - ${audio.led_mode || "virtual"}`;
      if (!micListening) {
        document.getElementById("audioMode").textContent = audio.enabled ? "Mic READY" : "Mic OFF";
        document.getElementById("audioMode").className = "status-pill" + (audio.led_on ? " alert" : "");
      }
      document.getElementById("alertStack").innerHTML = activeKeys.length
        ? activeKeys.slice(0, 4).map(key => `<div class="alert-item"><span>CANH BAO: ${labels[key] || key}</span><small>now</small></div>`).join("")
        : `<div class="alert-item idle"><span>Khong co canh bao dang mo</span><small>live</small></div>`;
      document.getElementById("behaviorCards").innerHTML = Object.entries(labels).map(([key, label]) => {
        const active = Boolean(status.behaviors[key]);
        return `<div class="behavior ${active ? "active" : ""}"><strong>${label}</strong><span>${active ? "Dang phat hien" : "Khong phat hien"} - ${counts[key] || 0} lan</span></div>`;
      }).join("");
      updateBars(counts);
      drawChart();
      showToast(activeKeys.filter(key => key !== "Raised Hand"));
      document.getElementById("eventsBody").innerHTML = events.map(row => `
        <tr>
          <td>#${row.id}</td>
          <td>${labels[row.behavior] || row.behavior}</td>
          <td>${row.timestamp}</td>
          <td><img class="thumb" src="/${row.image_path}" alt=""></td>
          <td class="hash">${row.hash}</td>
          <td><span class="chain">${row.ethereum_tx || ""}</span></td>
          <td><button class="mini-btn" onclick="sendMetamask(${row.id})">Gui hash</button></td>
        </tr>
      `).join("");
    }
    refresh();
    setInterval(refresh, 1500);
  </script>
</body>
</html>
"""


cfg = BehaviorConfig()
db = BehaviorDatabase(DB_PATH)
ethereum = EthereumClient(ETH_LEDGER)
engine = DetectionEngine(cfg, db, ethereum)
app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)


@app.route("/")
def index():
    engine.start()
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    engine.start()

    def generate():
        while True:
            frame = engine.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.02)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify(engine.snapshot_status())


@app.route("/api/audio_transcript", methods=["POST"])
def api_audio_transcript():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    result = engine.submit_transcript(text)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/events")
def api_events():
    return jsonify(db.list_events())


@app.route("/api/events/<int:event_id>")
def api_event(event_id: int):
    event = db.get_event(event_id)
    if not event:
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify(event)


@app.route("/api/chain_config")
def api_chain_config():
    return jsonify(public_metamask_config())


@app.route("/api/events/<int:event_id>/metamask_tx", methods=["POST"])
def api_metamask_tx(event_id: int):
    event = db.get_event(event_id)
    if not event:
        return jsonify({"ok": False, "error": "not_found"}), 404
    data = request.get_json(silent=True) or {}
    tx_hash = str(data.get("tx_hash", "")).strip()
    if not tx_hash.startswith("0x") or len(tx_hash) != 66:
        return jsonify({"ok": False, "error": "invalid_tx_hash"}), 400
    db.update_ethereum_tx(event_id, tx_hash)
    return jsonify({"ok": True, "ethereum_tx": tx_hash})


@app.route("/api/events/<int:event_id>/verify")
def api_verify_event(event_id: int):
    event = db.get_event(event_id)
    if not event:
        return jsonify({"ok": False, "error": "not_found"}), 404
    image_path = BASE_DIR / event["image_path"]
    actual_hash = compute_hash(event["behavior"], event["timestamp"], image_path)
    return jsonify({"ok": actual_hash == event["hash"], "stored_hash": event["hash"], "actual_hash": actual_hash})


@app.route("/api/start", methods=["POST"])
def api_start():
    engine.start()
    return jsonify({"ok": True, "running": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    engine.stop()
    return jsonify({"ok": True, "running": False})


@app.route("/logs/snapshots/<path:filename>")
def snapshots(filename: str):
    return send_from_directory(SNAPSHOT_DIR, filename)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "5000")), threaded=True)
