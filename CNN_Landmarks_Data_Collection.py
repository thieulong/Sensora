import cv2
import os

OUTPUT_DIR = "data/img"
CLASSES = ["anger", "fear", "joy", "love", "neutral", "sadness", "surprise"]
FRAMES_PER_CLASS = 200
IMG_SIZE = (224, 224)  

for cls in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

cap = cv2.VideoCapture(0)
print("[INFO] Starting data collection...")

for cls in CLASSES:
    print(f"[CLASS] Collecting for: {cls.upper()} — Press 's' to start, 'q' to skip")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        display = frame.copy()
        cv2.putText(display, f"Press 's' to start capturing '{cls.upper()}'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Preview", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break
        elif key == ord('q'):
            print(f"[SKIP] Skipping {cls.upper()}")
            break

    if key == ord('q'):
        continue

    saved = 0
    while saved < FRAMES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            continue

        resized = cv2.resize(frame, IMG_SIZE)
        save_path = os.path.join(OUTPUT_DIR, cls, f"{cls}_{saved:04d}.jpg")
        cv2.imwrite(save_path, resized)

        cv2.putText(resized, f"{cls.upper()} {saved+1}/{FRAMES_PER_CLASS}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("Capturing", resized)
        cv2.waitKey(1)
        saved += 1

    print(f"[DONE] Collected {saved} images for '{cls}'")

cap.release()
cv2.destroyAllWindows()
print("All classes processed. Data saved to 'image_data'")
