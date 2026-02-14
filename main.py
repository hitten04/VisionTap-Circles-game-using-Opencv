import mediapipe as mp
import cv2
import numpy as np
import time
import random
import sys
import math
import json
import os

# Suppress importlib.metadata warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")

# Simple import fix
try:
    import importlib.metadata
except ImportError:
    pass

# Colors - Define before classes that use them
NEON_GREEN = (57, 255, 20)
NEON_PINK = (255, 20, 147)
NEON_BLUE = (30, 144, 255)
NEON_YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Game variables
score = 0
player_name = ""
game_time_limit = 60  # 60 seconds
game_start_time = 0
target_size = 30
particles = []
game_state = "MENU"  # MENU, PLAYING, GAME_OVER

# Target position - Will be set after screen detection
x_target = 500  # Temporary initial position
y_target = 300

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
    cv2.rectangle(image, (width//2 - 148, 362), (width//2 + 148, 398), BLACK, -1)
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

def spawn_new_target():
    global x_target, y_target, screen_width, screen_height
    # Use actual screen dimensions with safe margins
    margin = 80
    x_target = random.randint(margin, max(screen_width - margin, margin + 50))
    y_target = random.randint(margin, max(screen_height - margin, margin + 50))

def reset_game():
    global score, game_start_time, particles
    score = 0
    game_start_time = time.time()
    particles = []
    spawn_new_target()

print("🎮 VISION TAP GAME STARTING...")
print("📷 Initializing camera...")

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("❌ Error: Could not open camera!")
    print("🔧 Make sure your camera is connected and not being used by another app")
    sys.exit(1)

print("✅ Camera initialized successfully!")

# Set camera to fullscreen resolution
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("🤖 Initializing MediaPipe...")

# Create fullscreen window before the game loop
window_name = '🎮 Vision Tap - Hand Tracking Game'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)

# Get actual screen resolution
import tkinter as tk
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

print(f"🖥️  Screen Resolution: {screen_width}x{screen_height}")

# Initialize target position with actual screen dimensions
spawn_new_target()

try:
    # Suppress MediaPipe warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5) as hands:

        print("✅ MediaPipe initialized successfully!")
        print("🎯 Game Controls:")
        print("   - Type your name and press SPACE to start")
        print("   - Point your index finger at targets")
        print("   - Press 'Q' to quit")

        while video.isOpened():
            ret, frame = video.read()
            
            if not ret:
                print("❌ Error: Could not read from camera")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            
            # Resize camera image to fill entire screen
            image = cv2.resize(image, (screen_width, screen_height))
            imageHeight, imageWidth = screen_height, screen_width

            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
                    # Draw target
                    draw_target(image)
                    
                    # Update particles
                    particles[:] = [p for p in particles if p.update()]
                    for particle in particles:
                        particle.draw(image)
                    
                    # Hand tracking
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=NEON_PINK, thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=NEON_BLUE, thickness=2)
                            )
                            
                            # Check finger tip
                            for point in mp_hands.HandLandmark:
                                if str(point) == 'HandLandmark.INDEX_FINGER_TIP':
                                    landmark = hand_landmarks.landmark[point]
                                    pixel_coords = mp_drawing._normalized_to_pixel_coordinates(
                                        landmark.x, landmark.y, imageWidth, imageHeight
                                    )
                                    
                                    if pixel_coords:
                                        finger_x, finger_y = pixel_coords
                                        draw_glowing_circle(image, (finger_x, finger_y), 15, NEON_YELLOW)
                                        
                                        # Check collision
                                        distance = math.sqrt((finger_x - x_target)**2 + (finger_y - y_target)**2)
                                        if distance < target_size + 15:
                                            score += 1
                                            create_explosion(x_target, y_target)
                                            spawn_new_target()
                                            print(f"🎯 HIT! Score: {score}")
            
            elif game_state == "GAME_OVER":
                draw_game_over(image)

            cv2.imshow(window_name, image)

            # Handle input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif game_state == "MENU":
                if key == ord(' ') and player_name.strip():
                    game_state = "PLAYING"
                    reset_game()
                elif key == 8:  # Backspace
                    player_name = player_name[:-1]
                elif 32 <= key <= 126:  # Printable characters
                    if len(player_name) < 15:
                        player_name += chr(key)
            elif game_state == "GAME_OVER":
                if key == ord('r') or key == ord('R'):
                    game_state = "MENU"
                    player_name = ""

except Exception as e:
    print(f"❌ Error: MediaPipe initialization failed: {e}")
    print("🔧 Try reinstalling MediaPipe: pip install --upgrade mediapipe")

video.release()
cv2.destroyAllWindows()
print("👋 Thanks for playing Vision Tap!")