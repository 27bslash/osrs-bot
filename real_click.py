import pyautogui
from random import randint, uniform


def real_click(x):
    if x != 'shift':
        if randint(0, 54) == 53:
            pyautogui.click(clicks=2, interval=uniform(0.1, 0.2))
        else:
            pyautogui.click()
    elif x == 'shift':
        if randint(0, 300) == 53:
            pyautogui.keyDown('shift')
            pyautogui.click(clicks=2, interval=uniform(0.1, 0.2))
            pyautogui.keyUp('shift')
        elif randint(0, 500) == 32:
            pyautogui.click(button='right')
        elif randint(0, 225) == 3:
            pyautogui.click()
        else:
            pyautogui.keyDown('shift')
            pyautogui.click()
