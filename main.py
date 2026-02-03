import mediapipe as mp
import cv2
import numpy as np
import time
import random
import sys
import math
import json
import os

# Fix for importlib.metadata error
try:
    import importlib.metadata
except ImportError:
    import importlib_metadata

# Colors - Define before classes that use them
NEON_GREEN = (57, 255, 20)
NEON_PINK = (255, 20, 147)
NEON_BLUE = (30, 144, 255)
NEON_YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)

# Initialize MediaPipe
try:
    import mediapipe as mp
    
    # Create hand landmarker
    base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task')
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.8,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
    
    print("✅ Using MediaPipe v0.10+ tasks API")
    
except Exception as e:
    print(f"❌ Error: MediaPipe initialization failed: {e}")
    print("🔧 Make sure hand_landmarker.task model file is present")
    sys.exit(1)

# Game variables
score = 0
player_name = ""
game_time_limit = 60  # 60 seconds
game_start_time = 0
target_size = 30
particles = []
game_state = "MENU"  # MENU, PLAYING, GAME_OVER

# Target position - will be set properly when game starts
x_target = 640  # Center initially
y_target = 360  # Center initially

# High scores file
SCORES_FILE = "high_scores.json"

def load_high_scores():
    if os.path.exists(SCORES_FILE):
        try:
            with open(SCORES_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_high_scores(scores):
    with open(SCORES_FILE, 'w') as f:
        json.dump(scores, f, indent=2)

def add_high_score(name, score):
    scores = load_high_scores()
    scores.append({"name": name, "score": score, "date": time.strftime("%Y-%m-%d %H:%M")})
    scores.sort(key=lambda x: x["score"], reverse=True)
    scores = scores[:10]  # Keep top 10
    save_high_scores(scores)
    return scores

def get_best_score():
    scores = load_high_scores()
    if scores:
        return scores[0]
    return {"name": "None", "score": 0}

class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-5, 5)
        self.vy = random.uniform(-5, 5)
        self.life = 30
        colors = [(57, 255, 20), (255, 20, 147), (30, 144, 255), (255, 255, 0)]
        self.color = random.choice(colors)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        return self.life > 0

    def draw(self, image):
        if self.life > 0:
            cv2.circle(image, (int(self.x), int(self.y)), 3, self.color, -1)

def create_explosion(x, y):
    for _ in range(15):
        particles.append(Particle(x, y))

def draw_glowing_circle(image, center, radius, color, thickness=3):
    for i in range(thickness, 0, -1):
        alpha = 0.3 + (i / thickness) * 0.7
        glow_color = tuple(int(c * alpha) for c in color)
        cv2.circle(image, center, radius + i, glow_color, 2)
    cv2.circle(image, center, radius, color, -1)

def draw_target(image):
    global x_target, y_target, target_size
    pulse = int(5 * math.sin(time.time() * 8))
    current_size = target_size + pulse
    draw_glowing_circle(image, (x_target, y_target), current_size, NEON_GREEN)
    cv2.line(image, (x_target - 15, y_target), (x_target + 15, y_target), WHITE, 2)
    cv2.line(image, (x_target, y_target - 15), (x_target, y_target + 15), WHITE, 2)

def draw_menu(image):
    height, width = image.shape[:2]
    
    # Dark overlay
    overlay = np.zeros_like(image)
    cv2.addWeighted(image, 0.3, overlay, 0.7, 0, image)
    
    # Title
    cv2.putText(image, "VISION TAP", (width//2 - 150, 100), cv2.FONT_HERSHEY_DUPLEX, 2.5, NEON_PINK, 3)
    cv2.putText(image, "Hand Tracking Game", (width//2 - 120, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, NEON_BLUE, 2)
    
    # Best score
    best = get_best_score()
    cv2.putText(image, f"Best Score: {best['name']} - {best['score']}", (width//2 - 150, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, NEON_YELLOW, 2)
    
    # Instructions
    cv2.putText(image, "Enter your name and press SPACE to start", (width//2 - 200, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    cv2.putText(image, "Game Duration: 60 seconds", (width//2 - 120, 320), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, NEON_GREEN, 2)
    
    # Name input
    cv2.rectangle(image, (width//2 - 150, 360), (width//2 + 150, 400), WHITE, 2)
    cv2.rectangle(image, (width//2 - 148, 362), (width//2 + 148, 398), BLACK, -1)  # Black background
    cv2.putText(image, f"Name: {player_name}", (width//2 - 140, 385), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    
    # Controls
    cv2.putText(image, "Controls:", (50, height - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, NEON_PINK, 2)
    cv2.putText(image, "- Point finger at targets", (50, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    cv2.putText(image, "- SPACE: Start game", (50, height - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    cv2.putText(image, "- Q: Quit", (50, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

def draw_game_ui(image):
    global score, player_name, game_start_time, game_time_limit
    
    # Calculate remaining time
    elapsed_time = time.time() - game_start_time
    remaining_time = max(0, game_time_limit - elapsed_time)
    
    # UI Panel
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (350, 140), BLACK, -1)
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
    
    # Player info
    cv2.putText(image, f"Player: {player_name}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, NEON_PINK, 2)
    cv2.putText(image, f"Score: {score}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, NEON_GREEN, 2)
    
    # Best score
    best = get_best_score()
    cv2.putText(image, f"Best: {best['name']} - {best['score']}", (20, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, NEON_YELLOW, 2)
    
    # Timer with color coding
    timer_color = RED if remaining_time < 10 else NEON_BLUE
    cv2.putText(image, f"Time: {int(remaining_time)}s", (20, 115), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, timer_color, 2)
    
    # Progress bar
    bar_width = 200
    bar_height = 10
    progress = (game_time_limit - remaining_time) / game_time_limit
    cv2.rectangle(image, (20, 125), (20 + bar_width, 125 + bar_height), WHITE, 1)
    cv2.rectangle(image, (20, 125), (20 + int(bar_width * progress), 125 + bar_height), timer_color, -1)
    
    return remaining_time

def draw_game_over(image):
    global score, player_name
    height, width = image.shape[:2]
    
    # Dark overlay
    overlay = np.zeros_like(image)
    cv2.addWeighted(image, 0.3, overlay, 0.7, 0, image)
    
    # Game Over
    cv2.putText(image, "GAME OVER!", (width//2 - 150, 150), cv2.FONT_HERSHEY_DUPLEX, 2, RED, 3)
    
    # Final score
    cv2.putText(image, f"{player_name}'s Score: {score}", (width//2 - 120, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, NEON_GREEN, 2)
    
    # High scores
    scores = load_high_scores()
    cv2.putText(image, "TOP SCORES:", (width//2 - 80, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, NEON_YELLOW, 2)
    
    for i, score_entry in enumerate(scores[:5]):
        y_pos = 310 + i * 25
        color = NEON_PINK if score_entry["name"] == player_name and score_entry["score"] == score else WHITE
        cv2.putText(image, f"{i+1}. {score_entry['name']}: {score_entry['score']}", 
                    (width//2 - 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    # Instructions
    cv2.putText(image, "Press R to restart or Q to quit", (width//2 - 150, height - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

def spawn_new_target(screen_width=1280, screen_height=720):
    global x_target, y_target
    # Use a margin to keep targets away from edges
    margin = 80
    
    x_target = random.randint(margin, screen_width - margin)
    y_target = random.randint(margin, screen_height - margin)

def reset_game():
    global score, game_start_time, particles
    score = 0
    game_start_time = time.time()
    particles = []
    spawn_new_target()  # Will use default dimensions initially

print("� VISION TAP GAME STARTING...")
print("� Initializing camera...")

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("❌ Error: Could not open camera!")
    print("🔧 Make sure your camera is connected and not being used by another app")
    sys.exit(1)

print("✅ Camera initialized successfully!")

# Set camera to fullscreen resolution
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create window and set to fullscreen immediately
cv2.namedWindow('🎮 Vision Tap - Hand Tracking Game', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('🎮 Vision Tap - Hand Tracking Game', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with hand_landmarker:
    frame_timestamp_ms = 0

    while video.isOpened():
        ret, frame = video.read()
        
        if not ret:
            print("❌ Error: Could not read from camera")
            break

        # Flip and convert frame
        frame = cv2.flip(frame, 1)
        imageHeight, imageWidth, _ = frame.shape
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Use BGR frame for display
        image = frame.copy()

        if game_state == "MENU":
            draw_menu(image)
            
        elif game_state == "PLAYING":
            # Check if time is up
            remaining_time = draw_game_ui(image)
            
            if remaining_time <= 0:
                # Game over - save score
                add_high_score(player_name, score)
                game_state = "GAME_OVER"
            else:
                # Process hand tracking
                frame_timestamp_ms += 33  # Approximate 30 FPS
                results = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                
                # Draw target
                draw_target(image)
                
                # Update particles
                particles[:] = [p for p in particles if p.update()]
                for particle in particles:
                    particle.draw(image)
                
                # Hand tracking
                if results.hand_landmarks:
                    for hand_landmarks in results.hand_landmarks:
                        # Draw hand landmarks
                        for landmark in hand_landmarks:
                            x = int(landmark.x * imageWidth)
                            y = int(landmark.y * imageHeight)
                            cv2.circle(image, (x, y), 3, NEON_PINK, -1)
                        
                        # Check index finger tip (landmark 8)
                        if len(hand_landmarks) > 8:
                            finger_landmark = hand_landmarks[8]  # INDEX_FINGER_TIP
                            finger_x = int(finger_landmark.x * imageWidth)
                            finger_y = int(finger_landmark.y * imageHeight)
                            
                            draw_glowing_circle(image, (finger_x, finger_y), 15, NEON_YELLOW)
                            
                            # Check collision
                            distance = math.sqrt((finger_x - x_target)**2 + (finger_y - y_target)**2)
                            if distance < target_size + 15:
                                score += 1
                                create_explosion(x_target, y_target)
                                spawn_new_target(imageWidth, imageHeight)
                                print(f"🎯 HIT! Score: {score}")
        
        elif game_state == "GAME_OVER":
            draw_game_over(image)

        cv2.imshow('🎮 Vision Tap - Hand Tracking Game', image)

        # Handle input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
        elif game_state == "MENU":
            if key == ord(' ') and player_name.strip():
                game_state = "PLAYING"
                reset_game()
                # Set first target with proper screen dimensions
                spawn_new_target(imageWidth, imageHeight)
            elif key == 8:  # Backspace
                player_name = player_name[:-1]
            elif 32 <= key <= 126:  # Printable characters
                if len(player_name) < 15:
                    player_name += chr(key)
        elif game_state == "GAME_OVER":
            if key == ord('r') or key == ord('R'):
                game_state = "MENU"
                player_name = ""

video.release()
cv2.destroyAllWindows()
print("👋 Thanks for playing Vision Tap!")