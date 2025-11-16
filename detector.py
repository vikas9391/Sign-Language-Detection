import threading
from typing import Tuple, Optional, List, Any
import cv2
import numpy as np
import logging
from collections import deque, Counter
import time

# MediaPipe imports
try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    raise ImportError("MediaPipe not installed. Run: pip install mediapipe") from e

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state with better thread safety
_state_lock = threading.Lock()
_latest_label: Tuple[str, float] = ("", 0.0)
_current_word: str = ""
_word_history: List[str] = []


def get_latest_label() -> Tuple[str, float]:
    """Thread-safe getter for the latest detected label"""
    with _state_lock:
        return _latest_label


def get_current_word() -> str:
    """Thread-safe getter for the current word being formed"""
    with _state_lock:
        return _current_word


def get_word_history() -> List[str]:
    """Thread-safe getter for word history"""
    with _state_lock:
        return _word_history.copy()


class ASLRecognizer:
    """Complete ASL alphabet recognizer (A-Z)"""
    
    def __init__(self):
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_pips = [3, 6, 10, 14, 18]
        self.finger_mcps = [2, 5, 9, 13, 17]
        
    def recognize(self, landmarks: Any) -> Tuple[str, float]:
        """Recognize ASL letter from hand landmarks"""
        if landmarks is None or len(landmarks) != 21:
            return "", 0.0
        
        extended = self._get_extended_fingers(landmarks)
        
        # Check each letter with priority order
        checks = [
            ("Y", self._is_letter_y, landmarks, extended),
            ("I", self._is_letter_i, extended, landmarks),
            ("L", self._is_letter_l, landmarks, extended),
            ("V", self._is_letter_v, landmarks, extended),
            ("W", self._is_letter_w, landmarks, extended),
            ("B", self._is_letter_b, extended, landmarks),
            ("C", self._is_letter_c, landmarks),
            ("O", self._is_letter_o, landmarks),
            ("F", self._is_letter_f, landmarks, extended),
            ("D", self._is_letter_d, landmarks, extended),
            ("R", self._is_letter_r, landmarks, extended),
            ("U", self._is_letter_u, landmarks, extended),
            ("K", self._is_letter_k, landmarks, extended),
            ("H", self._is_letter_h, landmarks, extended),
            ("G", self._is_letter_g, landmarks, extended),
            ("A", self._is_letter_a, extended, landmarks),
            ("S", self._is_letter_s, landmarks, extended),
            ("E", self._is_letter_e, landmarks),
            ("M", self._is_letter_m, landmarks, extended),
            ("N", self._is_letter_n, landmarks, extended),
            ("T", self._is_letter_t, landmarks, extended),
            ("P", self._is_letter_p, landmarks, extended),
            ("Q", self._is_letter_q, landmarks, extended),
            ("X", self._is_letter_x, landmarks, extended),
            ("J", self._is_letter_j, landmarks, extended),
            ("Z", self._is_letter_z, landmarks, extended),
        ]
        
        for letter, check_func, *args in checks:
            confidence = check_func(*args)
            if confidence > 0.0:
                return letter, float(confidence)
        
        return "", 0.0
    
    def _get_extended_fingers(self, landmarks: Any) -> List[bool]:
        """Determine which fingers are extended"""
        extended = []
        wrist = landmarks[0]
        
        # Thumb
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        thumb_extended = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x) + 0.02
        extended.append(thumb_extended)
        
        # Other fingers
        for i in range(1, 5):
            tip = landmarks[self.finger_tips[i]]
            mcp = landmarks[self.finger_mcps[i]]
            finger_extended = tip.y < mcp.y - 0.03
            extended.append(finger_extended)
        
        return extended
    
    def _distance(self, p1: Any, p2: Any) -> float:
        """Calculate 2D distance"""
        return float(np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2))
    
    # All letter recognition methods
    def _is_letter_a(self, extended: List[bool], landmarks: Any) -> float:
        if not any(extended[1:5]) and extended[0]:
            thumb_tip = landmarks[4]
            index_mcp = landmarks[5]
            if abs(thumb_tip.y - index_mcp.y) < 0.10:
                return 0.92
            return 0.85
        return 0.0
    
    def _is_letter_b(self, extended: List[bool], landmarks: Any) -> float:
        if all(extended[1:5]) and not extended[0]:
            tips_aligned = all(
                abs(landmarks[self.finger_tips[i]].y - landmarks[self.finger_tips[i+1]].y) < 0.05
                for i in range(1, 4)
            )
            return 0.94 if tips_aligned else 0.86
        return 0.0
    
    def _is_letter_c(self, landmarks: Any) -> float:
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        gap = self._distance(thumb_tip, index_tip)
        
        if 0.08 < gap < 0.28:
            all_curved = all(
                landmarks[self.finger_tips[i]].y < landmarks[self.finger_mcps[i]].y + 0.02
                for i in range(1, 5)
            )
            if all_curved:
                return 0.90
            return 0.82
        return 0.0
    
    def _is_letter_d(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and not any(extended[2:5]):
            thumb_tip = landmarks[4]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            
            if self._distance(thumb_tip, middle_tip) < 0.09 or self._distance(thumb_tip, ring_tip) < 0.09:
                return 0.92
            return 0.83
        return 0.0
    
    def _is_letter_e(self, landmarks: Any) -> float:
        all_curled = all(
            landmarks[self.finger_tips[i]].y > landmarks[self.finger_mcps[i]].y - 0.02
            for i in range(1, 5)
        )
        
        if all_curled:
            thumb_tip = landmarks[4]
            index_pip = landmarks[6]
            if self._distance(thumb_tip, index_pip) < 0.10:
                return 0.89
            return 0.81
        return 0.0
    
    def _is_letter_f(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[2] and extended[3] and extended[4]:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            
            if self._distance(thumb_tip, index_tip) < 0.07:
                return 0.92
            return 0.83
        return 0.0
    
    def _is_letter_g(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[0] and extended[1] and not any(extended[2:5]):
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            wrist = landmarks[0]
            
            if abs(thumb_tip.y - index_tip.y) < 0.08:
                if abs(thumb_tip.y - wrist.y) < 0.18:
                    return 0.90
                return 0.82
        return 0.0
    
    def _is_letter_h(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            wrist = landmarks[0]
            
            if (abs(index_tip.x - middle_tip.x) < 0.04 and
                abs(index_tip.y - middle_tip.y) < 0.05 and
                abs(index_tip.y - wrist.y) < 0.15):
                return 0.90
        return 0.0
    
    def _is_letter_i(self, extended: List[bool], landmarks: Any) -> float:
        if extended[4] and not any(extended[0:4]):
            fist_formed = all(
                landmarks[self.finger_tips[i]].y > landmarks[self.finger_mcps[i]].y - 0.05
                for i in range(1, 4)
            )
            if fist_formed:
                return 0.94
            return 0.85
        return 0.0
    
    def _is_letter_j(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[4] and not any(extended[0:4]):
            return 0.86
        return 0.0
    
    def _is_letter_k(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            thumb_tip = landmarks[4]
            index_base = landmarks[5]
            middle_base = landmarks[9]
            
            if (thumb_tip.y < index_base.y and 
                abs(thumb_tip.x - (index_base.x + middle_base.x) / 2) < 0.05):
                return 0.90
            return 0.82
        return 0.0
    
    def _is_letter_l(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[0] and extended[1] and not any(extended[2:5]):
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            wrist = landmarks[0]
            
            thumb_vec = np.array([thumb_tip.x - wrist.x, thumb_tip.y - wrist.y])
            index_vec = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y])
            
            cos_angle = np.dot(thumb_vec, index_vec) / (
                np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec) + 1e-6
            )
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            
            if 65 < angle < 115:
                return 0.93
            return 0.82
        return 0.0
    
    def _is_letter_m(self, landmarks: Any, extended: List[bool]) -> float:
        if not any(extended[3:5]):
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            
            if (self._distance(thumb_tip, index_tip) < 0.10 and
                self._distance(thumb_tip, middle_tip) < 0.12 and
                self._distance(thumb_tip, ring_tip) < 0.14):
                return 0.87
        return 0.0
    
    def _is_letter_n(self, landmarks: Any, extended: List[bool]) -> float:
        if not any(extended[2:5]):
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            
            if (self._distance(thumb_tip, index_tip) < 0.10 and
                self._distance(thumb_tip, middle_tip) < 0.12):
                return 0.87
        return 0.0
    
    def _is_letter_o(self, landmarks: Any) -> float:
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        if self._distance(thumb_tip, index_tip) < 0.08:
            all_curved = all(
                landmarks[self.finger_tips[i]].y < landmarks[self.finger_mcps[i]].y + 0.04
                for i in range(1, 5)
            )
            if all_curved:
                return 0.92
            return 0.83
        return 0.0
    
    def _is_letter_p(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            index_tip = landmarks[8]
            wrist = landmarks[0]
            
            if index_tip.y > wrist.y + 0.05:
                return 0.87
        return 0.0
    
    def _is_letter_q(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[0] and extended[1] and not any(extended[2:5]):
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            wrist = landmarks[0]
            
            if thumb_tip.y > wrist.y + 0.05 and index_tip.y > wrist.y + 0.05:
                return 0.87
        return 0.0
    
    def _is_letter_r(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            index_pip = landmarks[6]
            middle_pip = landmarks[10]
            
            if self._distance(index_tip, middle_tip) < 0.04:
                pip_distance = self._distance(index_pip, middle_pip)
                if pip_distance > 0.02:
                    return 0.91
                return 0.83
        return 0.0
    
    def _is_letter_s(self, landmarks: Any, extended: List[bool]) -> float:
        if not any(extended[1:5]):
            thumb_tip = landmarks[4]
            index_mcp = landmarks[5]
            middle_mcp = landmarks[9]
            
            if (self._distance(thumb_tip, index_mcp) < 0.07 or
                self._distance(thumb_tip, middle_mcp) < 0.08):
                return 0.90
        return 0.0
    
    def _is_letter_t(self, landmarks: Any, extended: List[bool]) -> float:
        if not any(extended[1:5]):
            thumb_tip = landmarks[4]
            index_pip = landmarks[6]
            middle_pip = landmarks[10]
            
            if (self._distance(thumb_tip, index_pip) < 0.08 and
                self._distance(thumb_tip, middle_pip) < 0.10):
                return 0.89
        return 0.0
    
    def _is_letter_u(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            
            if abs(index_tip.x - middle_tip.x) < 0.04:
                if abs(index_tip.y - middle_tip.y) < 0.05:
                    return 0.93
                return 0.85
        return 0.0
    
    def _is_letter_v(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            
            separation = abs(index_tip.x - middle_tip.x)
            
            if separation > 0.06:
                if abs(index_tip.y - middle_tip.y) < 0.06:
                    return 0.95
                return 0.86
        return 0.0
    
    def _is_letter_w(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and extended[2] and extended[3] and not extended[4]:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            
            sep1 = abs(index_tip.x - middle_tip.x)
            sep2 = abs(middle_tip.x - ring_tip.x)
            
            if sep1 > 0.04 and sep2 > 0.04:
                if (abs(index_tip.y - middle_tip.y) < 0.06 and
                    abs(middle_tip.y - ring_tip.y) < 0.06):
                    return 0.93
                return 0.84
        return 0.0
    
    def _is_letter_x(self, landmarks: Any, extended: List[bool]) -> float:
        if not any(extended[2:5]) and not extended[0]:
            index_tip = landmarks[8]
            index_mcp = landmarks[5]
            wrist = landmarks[0]
            
            if (index_tip.y < wrist.y and 
                index_tip.y > index_mcp.y - 0.10):
                return 0.87
        return 0.0
    
    def _is_letter_y(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[0] and extended[4] and not any(extended[1:4]):
            thumb_tip = landmarks[4]
            pinky_tip = landmarks[20]
            
            if self._distance(thumb_tip, pinky_tip) > 0.16:
                return 0.94
            return 0.84
        return 0.0
    
    def _is_letter_z(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and not any(extended[2:5]):
            return 0.84
        return 0.0


class PredictionSmoother:
    """Advanced smoothing"""
    
    def __init__(self, window_size: int = 12, min_confidence: float = 0.75, consensus_threshold: float = 0.65):
        self.predictions: deque = deque(maxlen=window_size)
        self.confidences: deque = deque(maxlen=window_size)
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.consensus_threshold = consensus_threshold
        
    def add_prediction(self, label: str, confidence: float):
        self.predictions.append(label)
        self.confidences.append(confidence)
        
    def get_stable_prediction(self) -> Tuple[str, float]:
        if len(self.predictions) < 6:
            return "", 0.0
        
        valid = [
            (p, c) for p, c in zip(self.predictions, self.confidences)
            if c >= self.min_confidence and p != ""
        ]
        
        if len(valid) < 4:
            return "", 0.0
        
        labels = [p for p, _ in valid]
        counter = Counter(labels)
        
        if not counter:
            return "", 0.0
        
        most_common_label, count = counter.most_common(1)[0]
        
        consensus = count / len(self.predictions)
        if consensus < self.consensus_threshold:
            return "", 0.0
        
        label_confs = [c for p, c in valid if p == most_common_label]
        avg_conf = float(np.mean(label_confs))
        final_conf = min(avg_conf + 0.03, 0.98)
        
        return most_common_label, final_conf


class WordBuilder:
    """Build words from detected letters"""
    
    def __init__(self, letter_hold_time: float = 1.2, space_delay: float = 2.5, min_confidence: float = 0.80):
        self.current_word = ""
        self.last_letter = ""
        self.last_letter_time = 0.0
        self.letter_seen_start = 0.0
        self.letter_confirmed = False
        self.letter_hold_time = letter_hold_time
        self.space_delay = space_delay
        self.min_confidence = min_confidence
        self.words: List[str] = []
        self.lock = threading.Lock()
        
    def add_letter(self, letter: str, confidence: float):
        current_time = time.time()
        
        with self.lock:
            if not letter or confidence < self.min_confidence:
                if (self.current_word and 
                    self.last_letter and
                    current_time - self.last_letter_time > self.space_delay):
                    self._complete_word()
                
                self.letter_seen_start = 0
                self.letter_confirmed = False
                return
            
            if letter == self.last_letter:
                if self.letter_confirmed:
                    return
                
                if current_time - self.letter_seen_start >= self.letter_hold_time:
                    if not self.letter_confirmed:
                        self.current_word += letter
                        self.letter_confirmed = True
                        self.last_letter_time = current_time
                        
                        global _current_word
                        with _state_lock:
                            _current_word = self.current_word
                        
                        logger.info(f"✅ Added letter '{letter}' to word: {self.current_word}")
            else:
                self.last_letter = letter
                self.letter_seen_start = current_time
                self.letter_confirmed = False
                self.last_letter_time = current_time
    
    def _complete_word(self):
        with self.lock:
            if self.current_word:
                self.words.append(self.current_word)
                
                global _word_history
                with _state_lock:
                    _word_history.append(self.current_word)
                    if len(_word_history) > 20:
                        _word_history.pop(0)
                
                logger.info(f"📝 Word completed: '{self.current_word}'")
                
                self.current_word = ""
                self.last_letter = ""
                self.letter_confirmed = False
                
                global _current_word
                with _state_lock:
                    _current_word = ""
    
    def clear_current_word(self):
        with self.lock:
            self.current_word = ""
            self.last_letter = ""
            self.letter_confirmed = False
            
            global _current_word
            with _state_lock:
                _current_word = ""
    
    def delete_last_letter(self):
        with self.lock:
            if self.current_word:
                self.current_word = self.current_word[:-1]
                self.last_letter = ""
                self.letter_confirmed = False
                
                global _current_word
                with _state_lock:
                    _current_word = self.current_word
    
    def get_current_word(self) -> str:
        with self.lock:
            return self.current_word
    
    def get_words(self) -> List[str]:
        with self.lock:
            return self.words.copy()


class VideoCamera:
    """Enhanced video camera with ASL detection - SIMPLIFIED"""
    
    def __init__(self):
        logger.info("🎥 Initializing enhanced VideoCamera...")
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.hands: Optional[Any] = None
        self.is_running = False
        self.frame_count = 0
        self.process_every_n_frames = 3  # Process every 3rd frame
        
        self._initialize_camera()
        self._initialize_mediapipe()
        
        self.recognizer = ASLRecognizer()
        self.smoother = PredictionSmoother(
            window_size=10, 
            min_confidence=0.75,
            consensus_threshold=0.60
        )
        self.word_builder = WordBuilder(
            letter_hold_time=1.2,
            space_delay=2.5,
            min_confidence=0.80
        )
        
        self.last_label = ""
        self.last_conf = 0.0
        
        logger.info("✅ Enhanced VideoCamera initialized")

    def _initialize_camera(self):
        """Initialize camera"""
        for idx in [0, 1, -1]:
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # DirectShow for Windows
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Flush initial frames
                    for _ in range(5):
                        cap.read()
                    
                    success, _ = cap.read()
                    if success:
                        self.cap = cap
                        logger.info(f"✅ Camera opened on index {idx}")
                        return
                    cap.release()
            except Exception as e:
                logger.warning(f"Camera {idx} failed: {e}")
        
        raise RuntimeError("❌ No camera available")

    def _initialize_mediapipe(self):
        """Initialize MediaPipe"""
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=1
        )

    def get_frame(self):
        """Generate frames with sign detection"""
        global _latest_label
        
        if not self.cap:
            return
        
        self.is_running = True
        
        while self.is_running:
            try:
                success, frame = self.cap.read()
                if not success:
                    logger.warning("⚠️ Failed to read frame")
                    time.sleep(0.05)
                    continue
                
                self.frame_count += 1
                frame = cv2.flip(frame, 1)
                
                # Process every Nth frame
                should_process = (self.frame_count % self.process_every_n_frames == 0)
                
                if should_process:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    if self.hands is not None:
                        results = self.hands.process(rgb)
                    else:
                        results = None
                    
                    # Process detection results
                    if results is not None and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
                        # Draw landmarks
                        for handLms in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, handLms, 
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                            )
                        
                        landmarks = results.multi_hand_landmarks[0].landmark
                        current_label, current_conf = self.recognizer.recognize(landmarks)
                        
                        if current_label and current_conf > 0.70:
                            self.smoother.add_prediction(current_label, current_conf)
                            self.last_label = current_label
                            self.last_conf = current_conf
                    else:
                        self.smoother.add_prediction("", 0.0)
                        self.last_label = ""
                        self.last_conf = 0.0
                    
                    label, conf = self.smoother.get_stable_prediction()
                    self.word_builder.add_letter(label, conf)
                    
                    # Update global state
                    with _state_lock:
                        _latest_label = (label, conf)
                else:
                    # Reuse last detection
                    label, conf = self.last_label, self.last_conf
                
                # Always draw UI
                current_word = self.word_builder.get_current_word()
                word_history = self.word_builder.get_words()
                self._draw_ui(frame, label, conf, current_word, word_history)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                       
            except GeneratorExit:
                logger.info("Generator exit - stopping")
                break
            except Exception as e:
                logger.error(f"Frame error: {e}")
                time.sleep(0.1)
        
        self.is_running = False
        logger.info("Frame generation stopped")

    def _draw_ui(self, frame: np.ndarray, label: str, conf: float, current_word: str, word_history: List[str]):
        """Draw UI overlay"""
        h, w = frame.shape[:2]
        
        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        
        if label and conf > 0.70:
            cv2.putText(frame, f"Letter: {label}", (15, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
            
            bar_width = int(350 * conf)
            cv2.rectangle(frame, (15, 65), (365, 85), (50, 50, 50), -1)
            cv2.rectangle(frame, (15, 65), (15 + bar_width, 85), (0, 255, 0), -1)
            cv2.putText(frame, f"{conf:.0%}", (375, 78), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "Show ASL letter...", (15, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        
        # Middle - Current word
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 110), (w, 200), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "Current Word:", (15, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        
        if current_word:
            word_display = current_word
            if len(word_display) > 25:
                word_display = word_display[-25:]
            cv2.putText(frame, word_display, (15, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "(hold letter 1.2s)", (15, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        
        # Bottom - Word history
        if word_history:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 120), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            
            cv2.putText(frame, "Completed Words:", (15, h - 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            
            recent_words = word_history[-3:]
            words_text = " | ".join(recent_words)
            
            if len(words_text) > 45:
                words_text = "..." + words_text[-42:]
            
            cv2.putText(frame, words_text, (15, h - 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Total: {len(word_history)} words", (15, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Instructions
        instructions = [
            "Hold letter 1.2s to add",
            "Pause 2.5s for word end",
            "Use buttons to control"
        ]
        
        y_offset = 15
        for instruction in instructions:
            cv2.putText(frame, instruction, (w - 270, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 18

    def clear_current_word(self):
        """Clear current word"""
        self.word_builder.clear_current_word()
        logger.info("🗑️ Current word cleared")
    
    def delete_last_letter(self):
        """Delete last letter"""
        self.word_builder.delete_last_letter()
        logger.info("⌫ Last letter deleted")

    def stop(self):
        """Stop camera"""
        logger.info("🛑 Stopping camera...")
        self.is_running = False
        time.sleep(0.2)

    def __del__(self):
        """Cleanup"""
        self.is_running = False
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
        if self.hands:
            try:
                self.hands.close()
            except:
                pass
        logger.info("✅ Cleanup complete")