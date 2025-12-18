import pygame

pygame.mixer.init()
ALARM_ON = False

def start_alarm(mp3_path="assets/buzzer.mp3"):
    global ALARM_ON
    if not ALARM_ON:
        pygame.mixer.music.load(mp3_path)
        pygame.mixer.music.play(-1)
        ALARM_ON = True

def stop_alarm():
    global ALARM_ON
    if ALARM_ON:
        pygame.mixer.music.stop()
        ALARM_ON = False
