import threading
from typing import Tuple, Optional, Any, List
import cv2
import numpy as np
import logging
from collections import deque
import pyttsx3
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

_latest_label_lock = threading.Lock()
_latest_label: Tuple[str, float] = ("", 0.0)


def get_latest_label() -> Tuple[str, float]:
    """Thread-safe getter for the latest detected label"""
    with _latest_label_lock:
        return _latest_label


class SignLanguageRecognizer:
    """Enhanced rule-based ASL letter recognizer with improved accuracy"""
    
    def __init__(self):
        # Define finger tip and base landmark indices
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_pips = [3, 6, 10, 14, 18]
        self.finger_mcps = [2, 5, 9, 13, 17]
        self.wrist = 0
        
    def recognize_letter(self, landmarks) -> str:
        """Enhanced recognition with better distinction between similar signs"""
        if landmarks is None or len(landmarks) != 21:
            return ""
        
        extended = self._get_extended_fingers(landmarks)
        curled = self._get_curled_fingers(landmarks)
        
        # Check letters in priority order (most specific first)
        if self._is_letter_f(landmarks, extended):
            return "F"
        elif self._is_letter_e(landmarks, extended, curled):
            return "E"
        elif self._is_letter_o(landmarks):
            return "O"
        elif self._is_letter_c(landmarks):
            return "C"
        elif self._is_letter_v(landmarks, extended):
            return "V"
        elif self._is_letter_d(landmarks, extended):
            return "D"
        elif self._is_letter_l(landmarks, extended):
            return "L"
        elif self._is_letter_y(landmarks, extended):
            return "Y"
        elif self._is_letter_i(landmarks, extended):
            return "I"
        elif self._is_letter_b(landmarks, extended):
            return "B"
        elif self._is_fist(landmarks, extended):
            return "A"
        
        return ""
    
    def _get_extended_fingers(self, landmarks) -> List[bool]:
        """Check which fingers are extended (straightened)"""
        extended = []
        
        # Thumb (special case - horizontal extension)
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        
        thumb_extended = (thumb_tip.x < thumb_mcp.x - 0.04) or (thumb_tip.x < wrist.x - 0.05)
        extended.append(thumb_extended)
        
        # Other fingers (vertical extension)
        for i in range(1, 5):
            tip = landmarks[self.finger_tips[i]]
            pip = landmarks[self.finger_pips[i]]
            mcp = landmarks[self.finger_mcps[i]]
            
            finger_extended = tip.y < mcp.y - 0.08
            extended.append(finger_extended)
        
        return extended
    
    def _get_curled_fingers(self, landmarks) -> List[bool]:
        """Check which fingers are curled (bent into palm)"""
        curled = []
        
        wrist = landmarks[0]
        
        # Thumb curling
        thumb_tip = landmarks[4]
        thumb_curled = abs(thumb_tip.x - wrist.x) < 0.08
        curled.append(thumb_curled)
        
        # Other fingers - check if tip is below or close to MCP
        for i in range(1, 5):
            tip = landmarks[self.finger_tips[i]]
            mcp = landmarks[self.finger_mcps[i]]
            
            finger_curled = tip.y >= mcp.y - 0.02
            curled.append(finger_curled)
        
        return curled
    
    def _get_distance(self, p1, p2) -> float:
        """Calculate Euclidean distance between two landmarks"""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def _is_fist(self, landmarks, extended) -> bool:
        """Letter A: All fingers closed into fist"""
        fingers_closed = not any(extended[1:])
        palm_center = landmarks[0]
        tips_close = all(
            self._get_distance(landmarks[self.finger_tips[i]], palm_center) < 0.15
            for i in range(1, 5)
        )
        return fingers_closed and tips_close
    
    def _is_letter_b(self, landmarks, extended) -> bool:
        """Letter B: All four fingers extended together"""
        four_fingers = all(extended[1:])
        thumb_tucked = not extended[0]
        
        if not (four_fingers and thumb_tucked):
            return False
        
        fingers_together = True
        for i in range(1, 4):
            tip1 = landmarks[self.finger_tips[i]]
            tip2 = landmarks[self.finger_tips[i + 1]]
            distance = abs(tip1.x - tip2.x)
            if distance > 0.06:
                fingers_together = False
                break
        
        return fingers_together
    
    def _is_letter_c(self, landmarks) -> bool:
        """Letter C: Curved hand forming C shape"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        thumb_index_dist = self._get_distance(thumb_tip, index_tip)
        is_curved = 0.12 < thumb_index_dist < 0.35
        fingers_curved = all(
            landmarks[self.finger_tips[i]].y < landmarks[self.finger_mcps[i]].y
            for i in range(1, 5)
        )
        return is_curved and fingers_curved
    
    def _is_letter_d(self, landmarks, extended) -> bool:
        """Letter D: Index finger up, others form circle"""
        if not extended[1]:
            return False
        if any(extended[2:5]):
            return False
        thumb_tip = landmarks[4]
        middle_tip = landmarks[12]
        circle_dist = self._get_distance(thumb_tip, middle_tip)
        forms_circle = circle_dist < 0.08
        return forms_circle
    
    def _is_letter_e(self, landmarks, extended, curled) -> bool:
        """Letter E: All fingers curled tightly"""
        all_curled = all(curled[1:]) and not any(extended[1:])
        if not all_curled:
            return False
        palm = landmarks[0]
        tips_to_palm = [
            self._get_distance(landmarks[self.finger_tips[i]], palm)
            for i in range(1, 5)
        ]
        tight_curl = all(dist < 0.12 for dist in tips_to_palm)
        tips_low = all(
            landmarks[self.finger_tips[i]].y > landmarks[self.finger_mcps[i]].y - 0.05
            for i in range(1, 5)
        )
        return all_curled and tight_curl and tips_low
    
    def _is_letter_f(self, landmarks, extended) -> bool:
        """Letter F: Thumb and index form circle, other fingers up"""
        three_up = extended[2] and extended[3] and extended[4]
        index_not_extended = not extended[1]
        if not (three_up and index_not_extended):
            return False
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        touching = self._get_distance(thumb_tip, index_tip) < 0.06
        fingers_up = all(
            landmarks[self.finger_tips[i]].y < landmarks[self.finger_mcps[i]].y - 0.08
            for i in [12, 16, 20]
        )
        return touching and fingers_up
    
    def _is_letter_i(self, landmarks, extended) -> bool:
        """Letter I: Only pinky extended"""
        only_pinky = extended[4] and not any(extended[0:4])
        if not only_pinky:
            return False
        fist_formed = all(
            landmarks[self.finger_tips[i]].y > landmarks[self.finger_mcps[i]].y - 0.05
            for i in range(1, 4)
        )
        return fist_formed
    
    def _is_letter_l(self, landmarks, extended) -> bool:
        """Letter L: Thumb and index extended at 90 degrees"""
        if not (extended[0] and extended[1]):
            return False
        if any(extended[2:5]):
            return False
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        wrist = landmarks[0]
        thumb_vec = np.array([thumb_tip.x - wrist.x, thumb_tip.y - wrist.y])
        index_vec = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y])
        cos_angle = np.dot(thumb_vec, index_vec) / (np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_degrees = np.degrees(angle)
        return 60 < angle_degrees < 120
    
    def _is_letter_o(self, landmarks) -> bool:
        """Letter O: All fingertips form circle with thumb"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        main_circle = self._get_distance(thumb_tip, index_tip) < 0.07
        if not main_circle:
            return False
        all_curved = all(
            landmarks[self.finger_tips[i]].y < landmarks[self.finger_mcps[i]].y + 0.02
            and landmarks[self.finger_tips[i]].y > landmarks[self.finger_mcps[i]].y - 0.1
            for i in range(1, 5)
        )
        return all_curved
    
    def _is_letter_v(self, landmarks, extended) -> bool:
        """Letter V: Index and middle extended and separated"""
        if not (extended[1] and extended[2]):
            return False
        if extended[3] or extended[4]:
            return False
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        separation = abs(index_tip.x - middle_tip.x)
        both_up = (index_tip.y < landmarks[5].y) and (middle_tip.y < landmarks[9].y)
        return separation > 0.06 and both_up
    
    def _is_letter_y(self, landmarks, extended) -> bool:
        """Letter Y: Thumb and pinky extended"""
        if not (extended[0] and extended[4]):
            return False
        if any(extended[1:4]):
            return False
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]
        separation = self._get_distance(thumb_tip, pinky_tip)
        return separation > 0.15


class TextToSpeechEngine:
    """Thread-safe text-to-speech engine with proper cooldown"""
    
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            self.enabled = True
            logger.info("✅ TTS engine initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ TTS initialization failed: {e}. TTS disabled.")
            self.engine = None
            self.enabled = False
            
        self.lock = threading.Lock()
        self.last_spoken = ""
        self.last_spoken_time = 0
        self.speaking = False
        self.cooldown_seconds = 3.0  # Prevent speaking same letter within 3 seconds
        
    def speak(self, text: str):
        """Speak text in a separate thread with cooldown"""
        if not self.enabled or not text:
            return
        
        current_time = time.time()
        
        # Check if already speaking
        if self.speaking:
            return
        
        # Check cooldown - prevent speaking same letter too soon
        if text == self.last_spoken:
            time_since_last = current_time - self.last_spoken_time
            if time_since_last < self.cooldown_seconds:
                return  # Still in cooldown period
        
        def _speak():
            with self.lock:
                self.speaking = True
                try:
                    logger.info(f"🔊 Speaking: {text}")
                    self.engine.say(text)
                    self.engine.runAndWait()
                    self.last_spoken = text
                    self.last_spoken_time = time.time()
                except Exception as e:
                    logger.error(f"❌ TTS error: {e}")
                finally:
                    self.speaking = False
        
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()


class VideoCamera:
    def __init__(self):
        logger.info("🎥 Initializing VideoCamera...")
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.hands = None
        self.is_running = False
        self.frame_timeout = 5.0
        self.last_frame_time = time.time()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10
        
        # Initialize camera
        self._initialize_camera()
        
        # Initialize MediaPipe
        self._initialize_mediapipe()
        
        # Initialize recognizer and TTS
        self.recognizer = SignLanguageRecognizer()
        self.tts = TextToSpeechEngine()
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=10)
        self.last_spoken_letter = ""
        self.letter_stable_count = 0
        self.min_stable_frames = 5
        
        logger.info("✅ VideoCamera initialized successfully")

    def _initialize_camera(self):
        """Initialize camera with error handling"""
        for camera_index in [0, 1, -1]:
            try:
                logger.info(f"📹 Trying camera index {camera_index}...")
                cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if cv2.os.name == 'nt' else cv2.CAP_V4L2)
                
                if cap.isOpened():
                    try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception as e:
                        logger.warning(f"⚠️ Could not set camera properties: {e}")
                    
                    success, test_frame = cap.read()
                    if success and test_frame is not None:
                        self.cap = cap
                        logger.info(f"✅ Camera opened successfully on index {camera_index}")
                        return
                    else:
                        cap.release()
                        logger.warning(f"⚠️ Camera {camera_index} opened but couldn't read frame")
                else:
                    cap.release()
                    
            except Exception as e:
                logger.error(f"❌ Failed to open camera {camera_index}: {e}")
        
        raise RuntimeError("❌ Could not open any camera. Please check if camera is available.")

    def _initialize_mediapipe(self):
        """Initialize MediaPipe"""
        try:
            logger.info("🤖 Initializing MediaPipe Hands...")
            self.hands = mp_hands.Hands(
                model_complexity=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
                max_num_hands=1
            )
            self.drawer = mp_drawing
        except Exception as e:
            logger.error(f"❌ Failed to initialize MediaPipe: {e}")
            raise

    def _reconnect_camera(self):
        """Attempt to reconnect camera"""
        logger.warning("🔄 Attempting to reconnect camera...")
        
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        time.sleep(0.5)
        
        try:
            self._initialize_camera()
            self.consecutive_failures = 0
            logger.info("✅ Camera reconnected successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to reconnect camera: {e}")
            return False

    def _get_stable_prediction(self, current_letter: str) -> Tuple[str, float]:
        """Get stable prediction with lower threshold"""
        if current_letter:
            self.prediction_buffer.append(current_letter)
        
        if len(self.prediction_buffer) < 3:
            return "", 0.0
        
        from collections import Counter
        counter = Counter(self.prediction_buffer)
        most_common = counter.most_common(1)[0]
        letter, count = most_common
        confidence = count / len(self.prediction_buffer)
        
        if confidence > 0.5:
            if letter == self.last_spoken_letter:
                self.letter_stable_count += 1
            else:
                self.letter_stable_count = 0
                self.last_spoken_letter = letter
            
            if self.letter_stable_count >= self.min_stable_frames:
                return letter, confidence
            else:
                return letter, confidence * 0.9
        
        return "", 0.0

    def get_frame(self):
        """Generator function that yields frames for streaming"""
        global _latest_label
        
        if self.cap is None:
            logger.error("❌ Camera not initialized")
            return
        
        logger.info("▶️ Starting frame capture...")
        self.is_running = True
        frame_count = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                if current_time - self.last_frame_time > self.frame_timeout:
                    logger.warning("⏱️ Frame timeout detected")
                    if not self._reconnect_camera():
                        break
                    self.last_frame_time = current_time
                    continue
                
                success, frame = None, None
                
                try:
                    success, frame = self.cap.read()
                except Exception as e:
                    logger.error(f"❌ Error reading frame: {e}")
                    success = False
                
                if not success or frame is None:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        logger.error("❌ Too many consecutive failures")
                        if not self._reconnect_camera():
                            break
                    time.sleep(0.1)
                    continue
                
                self.consecutive_failures = 0
                self.last_frame_time = current_time
                frame_count += 1

                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process with MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = None
                
                try:
                    results = self.hands.process(rgb)
                except Exception as e:
                    logger.error(f"❌ MediaPipe processing error: {e}")

                current_letter = ""
                
                if results and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
                    try:
                        for handLms in results.multi_hand_landmarks:
                            self.drawer.draw_landmarks(
                                frame, 
                                handLms, 
                                mp_hands.HAND_CONNECTIONS,
                                self.drawer.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                                self.drawer.DrawingSpec(color=(255, 0, 255), thickness=2)
                            )
                        
                        hand_landmarks = results.multi_hand_landmarks[0]
                        current_letter = self.recognizer.recognize_letter(hand_landmarks.landmark)
                    except Exception as e:
                        logger.error(f"❌ Error in landmark processing: {e}")
                
                # Get stable prediction
                label, conf = self._get_stable_prediction(current_letter)
                
                # Display result
                if label:
                    cv2.rectangle(frame, (0, 0), (450, 80), (0, 0, 0), -1)
                    cv2.putText(
                        frame, 
                        f"Letter: {label}", 
                        (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, 
                        (0, 255, 0), 
                        3
                    )
                    cv2.putText(
                        frame, 
                        f"Confidence: {conf:.0%}", 
                        (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (200, 200, 200), 
                        2
                    )
                    
                    # Speak when stable
                    if self.letter_stable_count >= self.min_stable_frames:
                        self.tts.speak(label)
                else:
                    cv2.rectangle(frame, (0, 0), (350, 60), (0, 0, 0), -1)
                    cv2.putText(
                        frame, 
                        "Show ASL sign...", 
                        (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, 
                        (200, 200, 200), 
                        2
                    )

                # Update latest label (thread-safe)
                with _latest_label_lock:
                    _latest_label = (label, conf)

                # Encode frame
                try:
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ret:
                        continue
                        
                    jpg = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
                except Exception as e:
                    logger.error(f"❌ Error encoding frame: {e}")
                    continue
                       
            except GeneratorExit:
                logger.info("🛑 Generator exit requested")
                break
            except Exception as e:
                logger.error(f"❌ Error in get_frame: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("⏹️ Frame capture stopped")
        self.is_running = False

    def stop(self):
        """Stop the camera gracefully"""
        logger.info("🛑 Stopping camera...")
        self.is_running = False
        time.sleep(0.2)

    def __del__(self):
        """Cleanup when object is destroyed"""
        logger.info("🧹 Cleaning up VideoCamera...")
        self.is_running = False
        
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"❌ Error releasing camera: {e}")
        
        if hasattr(self, 'hands') and self.hands is not None:
            try:
                self.hands.close()
            except Exception as e:
                logger.error(f"❌ Error closing MediaPipe: {e}")
        
        logger.info("✅ VideoCamera cleanup complete")