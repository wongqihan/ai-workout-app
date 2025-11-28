import os

# Disable GPU for MediaPipe/TensorFlow to prevent GL context errors on Cloud
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import sys

# Force Qt to run in offscreen mode to prevent segfaults in headless environments
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

print("DEBUG: Starting app v2.0 (optimized)...", flush=True)

# Core imports
import streamlit as st
import numpy as np
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Global variables for MediaPipe (initialized lazily)
mp_pose = None
mp_drawing = None
mp_drawing_styles = None

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Workout Form Corrector",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS for UI Customization ---
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #0e1117;
        padding: 0rem;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem;
        transition: all 0.3s ease;
    }
    
    /* Start/Pause Button (Dynamic) */
    div[data-testid="stButton"] button:contains("Start") {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
    }
    div[data-testid="stButton"] button:contains("Pause") {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border: none;
    }
    
    /* Reset Button */
    div[data-testid="stButton"] button:contains("Reset") {
        background-color: #374151;
        color: white;
        border: 1px solid #4b5563;
    }
    
    /* Radio Button Styling */
    .stRadio > label {
        font-weight: bold;
        color: white;
    }
    
    /* Video Container */
    div[data-testid="stVerticalBlock"] > div:has(video) {
        width: 100%;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    </style>
""", unsafe_allow_html=True)

# --- MediaPipe Initialization (Eager for cloud deployment) ---
def init_mediapipe():
    global mp_pose, mp_drawing, mp_drawing_styles
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe imports
init_mediapipe()

# --- Helper Functions ---

def calculate_angle(a, b, c):
    """
    Calculate the angle at point b given three points a, b, c.
    Returns angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def draw_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                            font_scale=1, text_color=(255, 255, 255), bg_color=(0, 0, 0), 
                            thickness=2, padding=10, alpha=0.6):
    """Draw text with a semi-transparent background."""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    
    # Background rectangle coordinates
    if x == 'center':
        x = (img.shape[1] - text_width) // 2
    if y == 'center':
        y = (img.shape[0] + text_height) // 2
        
    # Adjust for bottom-right alignment if needed
    if x == 'right':
        x = img.shape[1] - text_width - padding * 2
    if y == 'bottom':
        y = img.shape[0] - padding
        
    # Draw semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, 
                 (x - padding, y - text_height - padding), 
                 (x + text_width + padding, y + padding), 
                 bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

# --- Video Processor Class ---

class WorkoutProcessor:
    def __init__(self):
        # MediaPipe should already be initialized at module level
        # Optimized settings for cloud deployment
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Using 1 (full) because it's bundled. 0 (lite) requires download which fails on cloud.
            min_detection_confidence=0.3,  # Lower for better performance
            min_tracking_confidence=0.3
        )
        self.counter = 0
        self.stage = "UP"
        self.feedback = ""
        self.mode = "Squat"  # Default
        self.running = False
        self.frame_count = 0  # For frame skipping

    def process_squat(self, landmarks):
        # Keypoints: Hip, Knee, Ankle (Right side)
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        angle = calculate_angle(hip, knee, ankle)
        
        # Logic
        if angle > 160:
            self.stage = "UP"
        if angle < 80 and self.stage == "UP":
            self.stage = "DOWN"
            self.counter += 1
            
        # Feedback
        if self.stage == "DOWN" and angle > 100:
            self.feedback = "Go lower! Break parallel."
        elif self.stage == "DOWN":
            self.feedback = "Good depth!"
        else:
            self.feedback = ""
            
        return angle

    def process_pushup(self, landmarks):
        # Keypoints: Shoulder, Elbow, Wrist (Right side)
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        # For back sag check
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        
        # Back Sag Logic (simplified linear check)
        # Vector from shoulder to ankle
        vec_sa = np.array(ankle) - np.array(shoulder)
        # Vector from shoulder to hip
        vec_sh = np.array(hip) - np.array(shoulder)
        # Project hip onto shoulder-ankle line
        t = np.dot(vec_sh, vec_sa) / np.dot(vec_sa, vec_sa)
        closest_point = np.array(shoulder) + t * vec_sa
        # Distance from hip to line
        sag_dist = np.linalg.norm(np.array(hip) - closest_point)
        
        # Logic
        if angle > 160:
            self.stage = "UP"
        if angle < 90 and self.stage == "UP":
            self.stage = "DOWN"
            self.counter += 1
            
        # Feedback
        if sag_dist > 0.05: # Threshold for sag
            self.feedback = "Keep back straight!"
        elif self.stage == "DOWN" and angle > 100:
            self.feedback = "Go lower! Aim for 90 deg."
        elif self.stage == "DOWN":
            self.feedback = "Good depth!"
        else:
            self.feedback = ""
            
        return angle

    def recv(self, frame):
        # Initialize default values in case of error
        pose_detected = False
        img = None
        
        try:
            img = frame.to_ndarray(format="bgr24")
            h, w, _ = img.shape
            
            # Frame skipping for performance - process every 3rd frame (was 2nd)
            self.frame_count += 1
            skip_frame = (self.frame_count % 3 != 0)
            
            # Do pose detection
            if not skip_frame:
                # Resize for much faster processing (keep aspect ratio)
                # Reduced to 320px for maximum memory/CPU efficiency on Cloud
                process_width = 320
                scale = process_width / w
                process_height = int(h * scale)
                img_small = cv2.resize(img, (process_width, process_height))
                img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
                
                results = self.pose.process(img_rgb)
                
                # Explicitly clear large objects to help memory
                del img_small
                del img_rgb
                
                # Periodic Garbage Collection (every 100 frames processed)
                if self.frame_count % 300 == 0:
                    import gc
                    gc.collect()
                
                # Draw skeleton if detected
                if results.pose_landmarks:
                    pose_detected = True
                    
                    # Draw on original image (requires scaling landmarks back up)
                    # But mp_drawing.draw_landmarks works with normalized coordinates (0-1),
                    # so we can pass the original image and the landmarks found on the small image!
                    mp_drawing.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=2)
                    )
                    
                    # Only count reps if running
                    if self.running:
                        landmarks = results.pose_landmarks.landmark
                        if self.mode == "Squat":
                            self.process_squat(landmarks)
                        else:
                            self.process_pushup(landmarks)
        except Exception as e:
            print(f"Error in recv: {e}")
            import traceback
            traceback.print_exc()

        
        # Show status indicator (top left)
        status_text = "🟢 RUNNING" if self.running else "⏸️ PAUSED"
        status_color = (0, 255, 0) if self.running else (255, 165, 0)
        cv2.putText(img, status_text, (20, 40), 
                   cv2.FONT_HERSHEY_BOLD, 1.0, status_color, 3, cv2.LINE_AA)
        
        # Show pose detection status
        detection_text = "✓ Pose Detected" if pose_detected else "✗ No Pose - Stand in frame!"
        detection_color = (0, 255, 0) if pose_detected else (0, 0, 255)
        cv2.putText(img, detection_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, detection_color, 2, cv2.LINE_AA)
        
        # If paused, show instruction
        if not self.running:
            instruction = "Click 'Start Workout' to begin counting"
            cv2.putText(img, instruction, (20, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                
        # 4. Draw UI Overlay (On Video)
        
        # Rep Count Box (Bottom Right) - LARGE and PROMINENT
        # h, w, _ = img.shape # This line is now redundant as h, w are defined at the start
        # Bigger box dimensions
        box_w, box_h = 220, 150
        # Bright green background with border
        cv2.rectangle(img, (w - box_w - 20, h - box_h - 20), (w - 20, h - 20), (0, 255, 0), -1)
        cv2.rectangle(img, (w - box_w - 20, h - box_h - 20), (w - 20, h - 20), (255, 255, 255), 3)
        # HUGE counter number
        cv2.putText(img, str(self.counter), (w - box_w + 40, h - 50), 
                   cv2.FONT_HERSHEY_BOLD, 3.5, (0, 0, 0), 8, cv2.LINE_AA)
        # Label
        cv2.putText(img, "REPS", (w - box_w + 60, h - 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Feedback (Center, Semi-transparent)
        if self.feedback:
            draw_text_with_background(img, self.feedback, ('center', 'center'), 
                                    font_scale=1.2, bg_color=(0, 0, 255) if "!" in self.feedback else (0, 255, 0))

        if img is None:
            return frame
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Main App Layout ---

def main():
    # Initialize session state for controls and shared state
    if 'mode' not in st.session_state:
        st.session_state.mode = "Squat"
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'counter' not in st.session_state:
        st.session_state.counter = 0
    if 'stage' not in st.session_state:
        st.session_state.stage = "UP"
    
    # Layout: Video (Left/Center) + Controls (Right Column)
    col_video, col_controls = st.columns([3, 1])
    
    with col_controls:
        st.markdown("### ⚙️ Controls")
        
        # 1. Mode Selector
        mode = st.radio(
            "Select Mode", 
            ["Squat", "Push-up"],
            index=0 if st.session_state.mode == "Squat" else 1,
            key="mode_select"
        )
        
        # Update mode in session state
        st.session_state.mode = mode
        
        st.markdown("---")
        
        # 2. Start/Pause Button
        # We use a toggle-like behavior
        if st.session_state.running:
            if st.button("⏸️ Pause Workout", use_container_width=True):
                st.session_state.running = False
                st.rerun()
        else:
            if st.button("▶️ Start Workout", use_container_width=True):
                st.session_state.running = True
                st.rerun()
                
        # 3. Reset Button
        if st.button("🔄 Reset Counter", use_container_width=True):
            st.session_state.counter = 0
            st.session_state.stage = "UP"
            st.rerun()
            
        st.markdown("---")
        st.markdown(f"**Current Mode:** {mode}")
        st.markdown(f"**Reps:** {st.session_state.counter}")
        st.markdown("**Instructions:**")
        if mode == "Squat":
            st.info("Keep back straight. Lower hips until knees are < 80°.")
        else:
            st.info("Keep body aligned. Lower chest until elbows are < 90°.")

    with col_video:
        # Enhanced RTC Configuration with multiple ICE servers (STUN + TURN)
        rtc_configuration = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                # OpenRelay free TURN server (often helps with connection issues)
                {"urls": ["turn:openrelay.metered.ca:80"], "username": "openrelayproject", "credential": "openrelayproject"},
                {"urls": ["turn:openrelay.metered.ca:443"], "username": "openrelayproject", "credential": "openrelayproject"},
                {"urls": ["turn:openrelay.metered.ca:443?transport=tcp"], "username": "openrelayproject", "credential": "openrelayproject"},
            ]
        })
        
        webrtc_ctx = webrtc_streamer(
            key="workout-corrector",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_processor_factory=WorkoutProcessor,  # Pass class directly to avoid recreation
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Dynamically update processor state without restarting stream
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.mode = st.session_state.get('mode', "Squat")
            webrtc_ctx.video_processor.running = st.session_state.get('running', False)
            webrtc_ctx.video_processor.counter = st.session_state.get('counter', 0)
            webrtc_ctx.video_processor.stage = st.session_state.get('stage', "UP")
        
        # Display connection status
        if webrtc_ctx.state.playing:
            st.success("🟢 Camera connected!")
        elif webrtc_ctx.state.signalling:
            st.info("🟡 Connecting to camera...")
        else:
            st.warning("⚠️ Click 'START' above to begin")

if __name__ == "__main__":
    main()
