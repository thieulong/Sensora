import speech_recognition as sr

def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Starting microphone...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    while True:
        try:
            with mic as source:
                print("Listening...")
                audio = recognizer.listen(source, phrase_time_limit=5)

            print("Transcribing...")
            text = recognizer.recognize_google(audio)
            print(f"Result: {text}\n")

        except sr.UnknownValueError:
            print("Could not understand audio.\n")
        except sr.RequestError as e:
            print(f"Could not request results; {e}\n")
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break

if __name__ == "__main__":
    main()
