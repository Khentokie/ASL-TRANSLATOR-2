"""
data_collection.py
Collects training data from your webcam for three model types:

  1. STATIC LETTERS (A-Y except J/Z, plus SPACE) -> trained by
     train_cnn_letters.py, a static-pose classifier.
  2. MOTION LETTERS (J, Z) -> trained by train_lstm_jz.py, a small LSTM
     that tells the two apart by how the hand moves.
  3. WORDS (hello, thankyou, iloveyou, or any you add) -> trained by
     train_lstm_words.py, an LSTM over the whole gesture.

All three are recorded the same way, as motion sequences: pick the
letter/word, a 3-2-1 countdown plays, then SEQUENCE_LENGTH frames are
captured automatically (set in utils.py, along with the requested camera
FPS) while you hold/perform the sign. train_cnn_letters.py unpacks every
frame of a static letter's recordings into its own training sample; J, Z
and words are trained on the whole sequence instead, since the MOTION is
what distinguishes them. After each recording the script prints the frame
rate it actually achieved.

WORDS also record WHERE the hands are relative to your face (forehead,
chin, cheek, chest...), because many words differ only by location -- e.g.
FATHER (thumb at the forehead) vs MOTHER (thumb at the chin) are the same
handshape and movement. Keep your whole face in view while recording words;
each word frame is 143 numbers (126 hand-shape + 17 location, see
utils.WORD_FEATURES). Words recorded before this change are 126 numbers per
frame and can't be used -- re-record them.

Letters mode uses only ONE hand (either left or right -- whichever
MediaPipe sees) so the static/motion letter models are never confused by
an incidental second hand in frame. Words mode still tracks up to two
hands, since some words use both.

Your face is marked with a dashed square and, if tracking is on, saved as a companion
file next to each recording -- see utils.FaceTracker. This is purely for
visualization/future use; nothing in the current training scripts reads it.

Controls (letters mode):
  - Press a letter key (A-Z) to record ONE motion sequence for that
    letter: 3-2-1 countdown, then SEQUENCE_LENGTH frames are captured
    automatically while you hold/perform the handshape.
  - Press the SPACE BAR to record your "space" gesture (used to separate
    words while fingerspelling). Pick a handshape clearly different from
    every letter, e.g. a flat open palm.
  - Press the same key again for another recording (vary your hand's
    angle/distance/speed slightly between recordings).
  - Press TAB to toggle the face tracking overlay on or off.
  - Press 'q' to quit.

Controls (words mode):
  - Press SPACE to start recording one sequence for the currently
    selected word.
  - Press 'n' to cycle to the next word.
  - Press 'a' to type the name of a brand-new word/gesture on the spot --
    it gets its own folder under data/words/<name>/ immediately, no code
    editing required, and stays available in future sessions.
  - (Face tracking is always on in words mode -- it's needed to record
    where your hands are relative to your face.)
  - Press 'q' to quit.

A letter is "recorded and auto-saved" the moment you press its key --
there's no separate typing step for letters, since the key press already
IS the input (this intentionally differs from words, which need typed
names because there's no one key per arbitrary word).

Run:
  python data_collection.py --mode letters
  python data_collection.py --mode words
  python data_collection.py --mode letters --no-body-face
"""

import os
import time
import argparse

import cv2
import numpy as np

from utils import (
    extract_landmarks, draw_landmarks, open_camera, list_cameras, mp_hands,
    FaceTracker, EMASmoother, compose_collection_frame, extract_location,
    HUD_ACCENT_GOOD, HUD_ACCENT_WARN, HUD_ACCENT_BAD, HUD_PRIMARY_BLUE,
    SEQUENCE_LENGTH, TOTAL_FEATURES, WORD_FEATURES,
)

LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]  # A-Z
SPACE_LABEL = "SPACE"                     # gesture used to separate words
LETTER_LABELS = LETTERS + [SPACE_LABEL]   # every class recorded in letters mode
WORDS = ["hello", "thankyou", "iloveyou"]

FACE_EVERY_N_FRAMES = 3   # run Face Mesh every Nth frame (it's relatively heavy)
WORD_FACE_EVERY_N_FRAMES = 3   # words: how often the face position is refreshed.
                              # Keep this the SAME as realtime_app.py so recording and live match.
TOGGLE_KEY = 9                 # TAB switches face tracking on/off
SMOOTHING_ALPHA = 0.5          # EMA smoothing strength for recorded landmarks

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LETTERS_DIR = os.path.join(DATA_DIR, "letters")
WORDS_DIR = os.path.join(DATA_DIR, "words")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def save_face_companion(out_dir, idx, face_sequence):
    face_dir = os.path.join(out_dir, "face")
    os.makedirs(face_dir, exist_ok=True)
    np.save(os.path.join(face_dir, f"{idx}.npy"), np.array(face_sequence, dtype=np.float32))


def report_face_motion(motion_log):
    if motion_log:
        print(f"  Face motion during recording: average {np.mean(motion_log):.4f}, "
              f"peak {np.max(motion_log):.4f}  (0 = your face stayed still)")


def report_fps(num_frames, elapsed):
    if elapsed > 0 and num_frames:
        print(f"  Recorded {num_frames} frames in {elapsed:.2f}s "
              f"(~{num_frames / elapsed:.0f} fps actually achieved)")


def show_frame(window_name, camera_frame, status_text, status_color, info_lines):
    """Builds the themed screen for data_collection.py and displays it."""
    canvas = compose_collection_frame(camera_frame, status_text, status_color, info_lines)
    cv2.imshow(window_name, canvas)


# ---------------------------------------------------------------------------
# Letters
# ---------------------------------------------------------------------------

def collect_letters(body_face=True, camera_index=None):
    for label in LETTER_LABELS:
        os.makedirs(os.path.join(LETTERS_DIR, label), exist_ok=True)

    cap = open_camera(camera_index)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)
    tracker = FaceTracker(enabled=body_face, every_n=FACE_EVERY_N_FRAMES)

    print("Letters mode (single hand only). Press A-Z to record a letter, "
          "SPACE BAR for the 'space' gesture, TAB for tracking, 'q' to quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        tracker.process(rgb)
        tracker.draw(frame)
        draw_landmarks(frame, results)

        counts = {
            letter: len([f for f in os.listdir(os.path.join(LETTERS_DIR, letter)) if f.endswith(".npy")])
            for letter in LETTER_LABELS
        }
        hand_seen = results.multi_hand_landmarks is not None
        info_lines = [
            "Counts: " + ", ".join(f"{k}:{v}" for k, v in list(counts.items())[:14]),
            ", ".join(f"{k}:{v}" for k, v in list(counts.items())[14:]),
            f"Face tracking: {'ON' if tracker.enabled else 'OFF'}",
            "A-Z=record  SPACE=record 'space'  TAB=tracking  q=quit",
        ]
        show_frame("ASL Data Collection - Letters", frame,
                  "HAND DETECTED" if hand_seen else "NO HAND",
                  HUD_ACCENT_GOOD if hand_seen else HUD_ACCENT_BAD, info_lines)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == TOGGLE_KEY:
            tracker.toggle()
            continue
        if key == 32:  # space bar -> record the "space" gesture
            record_letter_sequence(cap, hands, SPACE_LABEL, tracker)
            continue

        pressed = chr(key).upper() if 33 <= key < 127 else None
        if pressed in LETTERS:
            record_letter_sequence(cap, hands, pressed, tracker)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    tracker.close()


def record_letter_sequence(cap, hands, letter, tracker):
    """3-2-1 countdown, then SEQUENCE_LENGTH frames captured and saved as
    one .npy motion sequence (used by the static CNN, the J/Z LSTM, or
    simply recognized as a single class like SPACE -- data_collection.py
    doesn't need to know which; that's decided at training/recognition time)."""
    for countdown in [3, 2, 1]:
        ok, frame = cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        show_frame("ASL Data Collection - Letters", frame,
                  f"GET READY: '{letter}' in {countdown}", HUD_ACCENT_WARN, [])
        cv2.waitKey(500)

    sequence = []
    face_sequence = []
    motion_log = []
    recording_face = tracker.enabled
    smoother = EMASmoother(size=TOTAL_FEATURES, alpha=SMOOTHING_ALPHA)
    t_start = time.perf_counter()
    for _ in range(SEQUENCE_LENGTH):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        face_sequence.append(tracker.process(rgb).copy())
        motion_log.append(tracker.motion)
        tracker.draw(frame)
        draw_landmarks(frame, results)
        raw = extract_landmarks(results, single_hand=True)
        features = smoother.smooth(raw)
        sequence.append(features)

        info_lines = [f"face motion: {tracker.motion:.4f}"] if recording_face else []
        show_frame("ASL Data Collection - Letters", frame,
                  f"RECORDING '{letter}'", HUD_ACCENT_BAD, info_lines)
        cv2.waitKey(1)

    report_fps(len(sequence), time.perf_counter() - t_start)
    if len(sequence) == SEQUENCE_LENGTH:
        out_dir = os.path.join(LETTERS_DIR, letter)
        idx = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])
        np.save(os.path.join(out_dir, f"{idx}.npy"), np.array(sequence, dtype=np.float32))
        print(f"Saved sequence {idx} for letter '{letter}'")
        if recording_face:
            save_face_companion(out_dir, idx, face_sequence)
            report_face_motion(motion_log)
    else:
        print("Sequence too short, discarded.")


# ---------------------------------------------------------------------------
# Words
# ---------------------------------------------------------------------------

def collect_words(body_face=True, camera_index=None):
    os.makedirs(WORDS_DIR, exist_ok=True)
    for w in WORDS:
        os.makedirs(os.path.join(WORDS_DIR, w), exist_ok=True)

    words = list(WORDS)
    for existing_dir in sorted(os.listdir(WORDS_DIR)):
        if os.path.isdir(os.path.join(WORDS_DIR, existing_dir)) and existing_dir not in words:
            words.append(existing_dir)

    cap = open_camera(camera_index)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)
    # Face tracking is REQUIRED for words (not optional like in letters mode):
    # every word frame records where the hands are relative to the face.
    tracker = FaceTracker(enabled=True, every_n=WORD_FACE_EVERY_N_FRAMES)

    word_idx = 0
    print("Words mode (up to two hands). Keep your FACE in view of the camera -- "
          "each frame records where your hands are relative to it. "
          "SPACE to record, 'n' for next word, 'a' to add a new word, 'q' to quit.")

    while cap.isOpened():
        word = words[word_idx]
        existing = len([f for f in os.listdir(os.path.join(WORDS_DIR, word)) if f.endswith(".npy")])

        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        tracker.process(rgb)
        tracker.draw(frame)
        draw_landmarks(frame, results)

        face_ok = tracker.reference() is not None
        info_lines = [
            "Face: found" if face_ok else "Face: NOT FOUND - keep your face in view",
            "Hand location is recorded relative to your face.",
            "SPACE=record  n=next  a=add word  q=quit",
        ]
        show_frame("ASL Data Collection - Words", frame,
                  f"'{word}' - {existing} saved" if face_ok else "FACE NOT FOUND",
                  HUD_ACCENT_GOOD if face_ok else HUD_ACCENT_WARN, info_lines)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('n'):
            word_idx = (word_idx + 1) % len(words)
        elif key == ord(' '):
            record_word_sequence(cap, hands, word, tracker)
        elif key == ord('a'):
            new_name = prompt_new_word_name(cap, hands, tracker)
            if new_name:
                os.makedirs(os.path.join(WORDS_DIR, new_name), exist_ok=True)
                if new_name not in words:
                    words.append(new_name)
                word_idx = words.index(new_name)
                print(f"Added new word '{new_name}'. Press SPACE to start recording it.")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    tracker.close()


def prompt_new_word_name(cap, hands, tracker):
    """Lets you type a name for a brand-new word/gesture directly in the
    webcam window. ENTER confirms, ESC cancels, BACKSPACE deletes."""
    name = ""
    while True:
        ok, frame = cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        tracker.process(rgb)
        tracker.draw(frame)
        draw_landmarks(frame, results)

        show_frame("ASL Data Collection - Words", frame, f"New word: {name}_",
                  HUD_ACCENT_WARN, ["ENTER=confirm  ESC=cancel"])
        key = cv2.waitKey(1) & 0xFF

        if key in (13, 10):
            break
        elif key == 27:
            return None
        elif key in (8, 127):
            name = name[:-1]
        elif 32 <= key < 127:
            ch = chr(key)
            if ch.isalnum() or ch in ("_", "-"):
                name += ch

    name = name.strip()
    return name if name else None


def record_word_sequence(cap, hands, word, tracker):
    sequence = []
    face_sequence = []
    motion_log = []
    recording_face = tracker.enabled
    face_frames = 0
    smoother = EMASmoother(size=TOTAL_FEATURES, alpha=SMOOTHING_ALPHA)

    for countdown in [3, 2, 1]:
        ok, frame = cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        # Keep the face tracker running through the countdown so the face
        # position is already known when recording starts.
        tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        show_frame("ASL Data Collection - Words", frame,
                  f"GET READY: '{word}' in {countdown}", HUD_ACCENT_WARN,
                  ["Face found" if tracker.reference() else "FACE NOT FOUND - face the camera"])
        cv2.waitKey(500)

    t_start = time.perf_counter()
    for _ in range(SEQUENCE_LENGTH):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        face_sequence.append(tracker.process(rgb).copy())
        motion_log.append(tracker.motion)
        tracker.draw(frame)
        draw_landmarks(frame, results)
        raw = extract_landmarks(results, single_hand=False)
        features = smoother.smooth(raw)
        # Hand shape (126) + where the hands are relative to the face (17).
        face_ref = tracker.reference()
        face_frames += face_ref is not None
        sequence.append(np.concatenate([features, extract_location(results, face_ref)]))

        info_lines = ["Face found" if face_ref else "FACE NOT FOUND - face the camera"]
        show_frame("ASL Data Collection - Words", frame, f"RECORDING '{word}'",
                  HUD_ACCENT_BAD, info_lines)
        cv2.waitKey(1)

    report_fps(len(sequence), time.perf_counter() - t_start)
    if sequence:
        face_pct = 100.0 * face_frames / len(sequence)
        print(f"  Face located in {face_pct:.0f}% of frames.")
        if face_pct < 80:
            print("  WARNING: the face was missing for a lot of this recording, so the hand "
                  "LOCATION data is unreliable. Face the camera with your whole face visible "
                  "(and good lighting) and re-record this one.")
    if len(sequence) == SEQUENCE_LENGTH:
        out_dir = os.path.join(WORDS_DIR, word)
        idx = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])
        np.save(os.path.join(out_dir, f"{idx}.npy"), np.array(sequence, dtype=np.float32))
        print(f"Saved sequence {idx} for '{word}'")
        if recording_face:
            save_face_companion(out_dir, idx, face_sequence)
            report_face_motion(motion_log)
    else:
        print("Sequence too short, discarded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["letters", "words"], required=True)
    parser.add_argument("--no-body-face", action="store_true",
                        help="letters mode only: start with the face overlay off (TAB turns it on). Words mode always tracks the face.")
    parser.add_argument("--camera", type=int, default=None,
                        help="camera index to open. If not given, the first "
                             "working camera found is used automatically.")
    parser.add_argument("--list-cameras", action="store_true",
                        help="list camera indices that respond, then exit")
    args = parser.parse_args()

    if args.list_cameras:
        found = list_cameras()
        if not found:
            print("No camera responded on indices 0-7.")
        else:
            print("Working cameras:")
            for idx, w, h in found:
                print(f"  --camera {idx}   ({w}x{h})")
        raise SystemExit(0)

    if args.mode == "letters":
        collect_letters(body_face=not args.no_body_face, camera_index=args.camera)
    else:
        collect_words(body_face=not args.no_body_face, camera_index=args.camera)