import cv2
import time
from playsound import playsound
cap = cv2.VideoCapture(0)


def detect_motion(frame, frames):
    delta = cv2.absdiff(frames[0], frames[1])
    _, threshold = cv2.threshold(delta,50, 255, cv2.THRESH_BINARY)
    dilated_threshold = cv2.dilate(threshold, None, iterations=2)

    contours, _ = cv2.findContours(dilated_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 2000:
            continue
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    
    return frame

def cam_loop():
    frames = []
    end_time = time.time() + 5
    while time.time() < end_time:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray,(21,21),0)
        
        frames.append(blurred_gray)
        if len(frames) < 2:    
            continue 

        frame = detect_motion(frame, frames)
        cv2.imshow('Viewer', frame)
        frames = frames[-2:]
        
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def game_loop():
    while True:
        playsound('audio.wav')
        cam_loop()
    
if __name__ == "__main__":
    game_loop()
