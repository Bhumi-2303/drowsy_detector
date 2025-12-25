import os

# FLAGS
ENABLE_ALARM = True   # Set False in Docker / CI

ALARM_ON = False
pygame_available = False

# PATH SETUP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

BUZZER_PATH = os.path.join(
    ROOT_DIR, "assets", "buzzer.mp3"
)

# SAFE INIT

if ENABLE_ALARM:
    try:
        import pygame
        pygame.mixer.init()
        pygame_available = True
    except Exception as e:
        print(f"⚠️ Alarm disabled (audio unavailable): {e}")
        pygame_available = False

# ALARM FUNCTIONS

def start_alarm(mp3_path=BUZZER_PATH):
    global ALARM_ON
    if not pygame_available:
        return

    if not ALARM_ON:
        pygame.mixer.music.load(mp3_path)
        pygame.mixer.music.play(-1)
        ALARM_ON = True


def stop_alarm():
    global ALARM_ON
    if not pygame_available:
        return

    if ALARM_ON:
        pygame.mixer.music.stop()
        ALARM_ON = False
