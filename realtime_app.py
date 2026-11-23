"""
realtime_app.py
Real-time ASL translator with two modes (press 'm' to switch):

  FINGERSPELLING -- a hybrid static + motion pipeline for letters:
      Hand detected (single hand only)
        -> is it moving?
             NO  -> static CNN classifies the current pose (A-Y, minus J/Z,
                    plus SPACE) once the hand has held still for a moment.
             YES -> buffer frames while the motion continues, then (once it
                    settles or the buffer fills) a small LSTM decides J or
                    Z from the whole motion. Too-short/interrupted motion
                    is discarded rather than forced into a guess.
      This mirrors a simple state machine:
          IDLE -> hand detected -> static or motion? -> (CNN | buffer->LSTM)
                -> display result -> back to IDLE

  WORDS -- an LSTM reads the last SEQUENCE_LENGTH frames (both hands
      tracked) and predicts hello / thankyou / iloveyou / any word you've
      added via data_collection.py. Only the single most recent recognized
      word is shown; a new one replaces it rather than piling up.

In Fingerspelling mode, each recognized letter is added to a small
in-progress buffer shown as plain text ("Typing: ..."). The big "Translated
Text" display only updates when you press ENTER -- this finalizes whatever
you've typed so far (no spaces inserted between letters) and clears the
buffer for the next word, the same way Words mode only swaps its display
once a whole word is recognized. A visual flash confirms each letter
capture and each ENTER-finalized word.

A dashed square marks your tracked face (shown for reference; in words mode the face is
also what hand location is measured from).

The final picture is cropped/resized to a phone-sized portrait window
(utils.PHONE_WIDTH x utils.PHONE_HEIGHT) so it can be measured against a
real phone screen; detection itself always runs on the full camera frame
first, so this never costs recognition accuracy.

Controls:
  m         - toggle Fingerspelling / Words mode
  ENTER     - (Fingerspelling mode) finalize the typed letters into the big display
  TAB       - toggle face tracking overlay
  q         - quit
  c         - clear the in-progress letters AND the finalized display / clear captured word
  BACKSPACE - delete last typed letter (before ENTER) / clear captured word

Run: python realtime_app.py [--camera N]
"""

import os
import json
import time
import argparse
import threading
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

from utils import (
    extract_landmarks, draw_landmarks, open_camera, list_cameras, mp_hands,
    FaceTracker, EMASmoother, compose_app_frame,
    detected_hand_label, MotionSegmenter, hand_anchor, extract_location,
    MOTION_START_THRESHOLD, MOTION_END_THRESHOLD,
    HUD_ACCENT_GOOD, HUD_ACCENT_WARN, HUD_ACCENT_BAD,
    SEQUENCE_LENGTH, TOTAL_FEATURES, WORD_FEATURES,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# --- Tunable thresholds -----------------------------------------------
LETTER_CONF_THRESHOLD = 0.70     # min confidence to accept a static letter
JZ_CONFIDENCE_THRESHOLD = 0.80   # min confidence to accept a J/Z motion result
WORD_CONF_THRESHOLD = 0.90       # min confidence to accept a word
STABLE_FRAMES_FOR_LETTER = 3     # consecutive still frames before trusting a static pose

# Motion detection (fingerspelling mode) -- thresholds live in utils.py
# (MOTION_START_THRESHOLD / MOTION_END_THRESHOLD / MOTION_END_HOLD_FRAMES /
# JZ_MIN_FRAMES). Watch the "motion:" number at the bottom-left of the
# window: it should sit near 0.00x while your hand is still and climb past
# the start threshold when you sign J or Z. If it never gets there, lower
# MOTION_START_THRESHOLD; if it triggers while you're holding still, raise it.

JZ_NEGATIVE_LABEL = "OTHER"       # the J/Z model's "movement, but not J or Z" class

# J and Z each begin with a handshape that looks like a static letter (J
# starts as an "I", Z as a pointing index finger). The static recognizer
# can commit that start-pose a moment before the motion is recognized, so a
# J/Z result REPLACES such a letter if it was just typed (and static
# letters of the same kind are blocked briefly afterwards, since the hand
# also ENDS near that shape). Edit to suit how you sign them.
JZ_START_SHAPES = {"J": ("I",), "Z": ("D", "G")}
JZ_REPLACE_WINDOW = 2.5           # seconds: how recent the start-pose letter must be
JZ_STATIC_BLOCK_SECONDS = 3.0     # seconds static start-shape letters are blocked after a J/Z

# --- Per-letter pause AFTER recognition (seconds, not frames) ---------
LETTER_INTERVALS = {}
DEFAULT_LETTER_INTERVAL = 1.0
WORD_INTERVAL = 2.0


def letter_interval(letter):
    return LETTER_INTERVALS.get(letter, DEFAULT_LETTER_INTERVAL)


# --- Capture effect (visual flash when a letter/word is committed) ----
EFFECT_DURATION = 0.35
LETTER_EFFECT_COLOR = (0, 255, 0)
WORD_EFFECT_COLOR = (255, 140, 0)

# --- Mode toggle ---------------------------------------------------------
MODE_TOGGLE_KEY = ord('m')
TRACKING_TOGGLE_KEY = 9   # TAB -- same key as data_collection.py
DEFAULT_MODE = "fingerspelling"

# --- Speed settings --------------------------------------------------
HAND_MODEL_COMPLEXITY = 0
WORD_PREDICT_EVERY_N_FRAMES = 4
LETTER_PREDICT_EVERY_N_FRAMES = 2
FACE_EVERY_N_FRAMES = 6   # fingerspelling: run Face Mesh every Nth frame (it's relatively heavy)
WORD_FACE_EVERY_N_FRAMES = 3   # words: refresh the face position more often. Keep this the
                               # SAME as data_collection.py so live matches what was recorded.
SHOW_FPS = True
SMOOTHING_ALPHA = 0.5


class LatestFrameReader:
    """Reads the camera on a background thread and keeps only the newest
    frame, so processing lag doesn't make the picture fall behind real
    life (old frames are dropped instead of queued)."""

    def __init__(self, cap):
        self.cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        failures = 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                failures += 1
                if failures > 200:
                    break
                time.sleep(0.005)
                continue
            failures = 0
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def read(self, last_id, timeout=5.0):
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if self._frame is not None and self._frame_id != last_id:
                    return True, self._frame.copy(), self._frame_id
            time.sleep(0.001)
        return False, None, last_id

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)


def make_fast_predictor(model, input_shape):
    """Wraps a Keras model in a compiled tf.function for faster single-input
    calls than model.predict() or a plain model() call would give."""
    try:
        compiled = tf.function(
            lambda x: model(x, training=False),
            input_signature=[tf.TensorSpec(input_shape, tf.float32)],
        )
        compiled(np.zeros(input_shape, dtype=np.float32))
        return lambda x: compiled(tf.convert_to_tensor(x, dtype=tf.float32)).numpy()[0]
    except Exception as err:
        print(f"Note: compiled fast path unavailable ({err}); using regular model calls.")
        return lambda x: model(x, training=False).numpy()[0]


def load_models():
    paths = {
        "cnn": os.path.join(MODEL_DIR, "asl_cnn_letters.h5"),
        "jz": os.path.join(MODEL_DIR, "asl_lstm_jz.h5"),
        "words": os.path.join(MODEL_DIR, "asl_lstm_words.h5"),
        "letters_map": os.path.join(MODEL_DIR, "letters_label_map.json"),
        "jz_map": os.path.join(MODEL_DIR, "jz_label_map.json"),
        "words_map": os.path.join(MODEL_DIR, "words_label_map.json"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing) +
            ". Train all three first: python train_cnn_letters.py, "
            "python train_lstm_jz.py, python train_lstm_words.py"
        )

    cnn_model = tf.keras.models.load_model(paths["cnn"])
    jz_model = tf.keras.models.load_model(paths["jz"])
    words_model = tf.keras.models.load_model(paths["words"])

    cnn_model(np.zeros((1, 21, 3, 2), dtype=np.float32), training=False)
    jz_model(np.zeros((1, SEQUENCE_LENGTH, TOTAL_FEATURES), dtype=np.float32), training=False)
    trained_features = words_model.input_shape[-1]
    if trained_features != WORD_FEATURES:
        raise SystemExit(
            f"The words model in models/ expects {trained_features} numbers per frame, but words now use "
            f"{WORD_FEATURES} (hand shape + where the hands are relative to your face, which is what "
            "lets it tell apart words like FATHER and MOTHER). It was trained before that change.\n"
            "Re-record your words (python data_collection.py --mode words) and retrain "
            "(python train_lstm_words.py).")
    words_model(np.zeros((1, SEQUENCE_LENGTH, WORD_FEATURES), dtype=np.float32), training=False)

    cnn_predict = make_fast_predictor(cnn_model, (1, 21, 3, 2))
    jz_predict = make_fast_predictor(jz_model, (1, SEQUENCE_LENGTH, TOTAL_FEATURES))
    words_predict = make_fast_predictor(words_model, (1, SEQUENCE_LENGTH, WORD_FEATURES))

    with open(paths["letters_map"]) as f:
        letters_map = {int(k): v for k, v in json.load(f).items()}
    with open(paths["jz_map"]) as f:
        jz_map = {int(k): v for k, v in json.load(f).items()}
    with open(paths["words_map"]) as f:
        words_map = {int(k): v for k, v in json.load(f).items()}

    return cnn_predict, jz_predict, words_predict, letters_map, jz_map, words_map


def reshape_for_cnn(features):
    """(126,) -> (1, 21, 3, 2), matching train_cnn_letters.py's layout."""
    left = features[:63].reshape(21, 3)
    right = features[63:].reshape(21, 3)
    stacked = np.stack([left, right], axis=-1)
    return np.expand_dims(stacked, axis=0)


def pad_sequence(frames, length=SEQUENCE_LENGTH):
    """Right-pads a list of (126,) frames with zero rows up to `length`.
    The LSTM's Masking layer is built to ignore all-zero rows, so a motion
    that settled before filling the whole window can still be classified."""
    arr = np.array(frames, dtype=np.float32)
    if len(arr) >= length:
        return arr[:length]
    pad = np.zeros((length - len(arr), arr.shape[1]), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def jz_commit_decision(letter, confidence, sentence, last_letter, last_commit_time,
                       now, required_gap):
    """
    Decides whether a J/Z result should be typed. Returns (commit, replace_previous).

    replace_previous is True when the letter just typed was the static
    recognizer's reading of this gesture's STARTING handshape (e.g. an "I"
    typed an instant before the J motion was recognized) -- that letter is
    swapped for the J/Z instead of leaving "IJ". The normal pause between
    letters is waived in that case, since it's the same sign.
    """
    if confidence < JZ_CONFIDENCE_THRESHOLD:
        return False, False
    start_shapes = JZ_START_SHAPES.get(letter, ())
    replacing = (bool(sentence) and sentence[-1] in start_shapes
                 and last_letter in start_shapes
                 and (now - last_commit_time) <= JZ_REPLACE_WINDOW)
    if replacing:
        return True, True
    return (now - last_commit_time) >= required_gap, False


def confidence_color(confidence, threshold):
    if confidence >= threshold:
        return HUD_ACCENT_GOOD
    if confidence >= threshold * 0.6:
        return HUD_ACCENT_WARN
    return HUD_ACCENT_BAD


def draw_capture_effect(frame, effect, now):
    """Brief flash when a letter/word was just captured: a colored border
    plus the captured text popping up large in the center, fading out."""
    if effect is None:
        return frame
    elapsed = now - effect["time"]
    if elapsed >= EFFECT_DURATION:
        return frame

    progress = elapsed / EFFECT_DURATION
    alpha = 1.0 - progress
    color = effect["color"]
    text = effect["text"]

    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3.5 + progress * 1.5
    thickness = 6
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    border_thickness = max(2, int(24 * alpha))
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1),
                  color, border_thickness)
    return frame


def main(camera_index=0):
    cnn_predict, jz_predict, words_predict, letters_map, jz_map, words_map = load_models()

    hands_one = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                               model_complexity=HAND_MODEL_COMPLEXITY,
                               min_detection_confidence=0.6, min_tracking_confidence=0.5)
    hands_two = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                               model_complexity=HAND_MODEL_COMPLEXITY,
                               min_detection_confidence=0.6, min_tracking_confidence=0.5)
    tracker = FaceTracker(enabled=True, every_n=FACE_EVERY_N_FRAMES)

    cap = open_camera(camera_index)
    reader = LatestFrameReader(cap)
    frame_id = 0
    fps = 0.0
    last_loop_time = time.perf_counter()
    frame_count = 0

    smoother = EMASmoother(size=TOTAL_FEATURES, alpha=SMOOTHING_ALPHA)
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)   # words mode sliding window
    stable_count = 0
    segmenter = MotionSegmenter()   # fingerspelling: finds & collects J/Z motion gestures
    static_block_letters = set()    # static letters briefly blocked right after a J/Z
    static_block_until = 0.0

    last_letter = None
    last_word = None
    last_commit_time = 0.0

    sentence = []           # letters typed so far, NOT yet finalized (Enter commits them)
    fingerspell_result = "" # the finalized fingerspelled word, shown big -- replaced on each Enter
    captured_word = ""
    current_display = ""
    current_confidence = 0.0
    current_threshold = 0.0
    capture_effect = None
    mode = DEFAULT_MODE
    if mode == "words":
        tracker.every_n = WORD_FACE_EVERY_N_FRAMES

    while True:
        ok, frame, frame_id = reader.read(frame_id)
        if not ok:
            break

        loop_time = time.perf_counter()
        dt = loop_time - last_loop_time
        last_loop_time = loop_time
        if dt > 0:
            fps = (1.0 / dt) if fps == 0.0 else (0.9 * fps + 0.1 / dt)

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        active_hands = hands_one if mode == "fingerspelling" else hands_two
        results = active_hands.process(rgb)

        tracker.process(rgb)   # tracker internally throttles how often Face Mesh actually runs
        tracker.draw(frame)
        draw_landmarks(frame, results)
        frame_count += 1

        now = time.time()
        single_hand = (mode == "fingerspelling")
        raw_features = extract_landmarks(results, single_hand=single_hand)
        hand_present = np.any(raw_features)
        features = smoother.smooth(raw_features)

        motion = 0.0
        stability_fraction = 0.0
        if mode == "words":
            # Hand shape (126) + where the hands are relative to the face (17),
            # exactly as recorded by data_collection.py --mode words.
            frame_buffer.append(np.concatenate(
                [features, extract_location(results, tracker.reference())]))
            stability_fraction = len(frame_buffer) / SEQUENCE_LENGTH

        # ------------------------------------------------------------------
        # WORDS mode
        # ------------------------------------------------------------------
        if (mode == "words" and hand_present and len(frame_buffer) == SEQUENCE_LENGTH
                and frame_count % WORD_PREDICT_EVERY_N_FRAMES == 0):
            seq = np.expand_dims(np.array(frame_buffer), axis=0)
            probs = words_predict(seq)
            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])
            word = words_map[top_idx]

            current_display = f"{word} ({confidence:.2f})"
            current_confidence = confidence
            current_threshold = WORD_CONF_THRESHOLD

            if last_word is not None:
                required_gap = WORD_INTERVAL
            elif last_letter is not None:
                required_gap = letter_interval(last_letter)
            else:
                required_gap = 0.0
            if confidence >= WORD_CONF_THRESHOLD and (now - last_commit_time) >= required_gap:
                captured_word = word
                last_word = word
                last_letter = None
                last_commit_time = now
                capture_effect = {"text": word, "color": WORD_EFFECT_COLOR, "time": now}
                frame_buffer.clear()

        # ------------------------------------------------------------------
        # FINGERSPELLING mode: static CNN <-> motion-segmented J/Z LSTM
        # ------------------------------------------------------------------
        elif mode == "fingerspelling":
            anchor = hand_anchor(results) if hand_present else None
            event, jz_seq = segmenter.update(features if hand_present else None, anchor)
            motion = segmenter.motion

            if event == "complete":
                probs = jz_predict(np.expand_dims(pad_sequence(jz_seq), axis=0))
                top_idx = int(np.argmax(probs))
                confidence = float(probs[top_idx])
                letter = jz_map[top_idx]

                if letter == JZ_NEGATIVE_LABEL:
                    current_display = "(movement, but not J or Z)"
                    current_confidence, current_threshold = 0.0, 0.0
                else:
                    current_display = f"{letter} ({confidence:.2f})"
                    current_confidence = confidence
                    current_threshold = JZ_CONFIDENCE_THRESHOLD

                    required_gap = (letter_interval(last_letter) if last_letter
                                   else (WORD_INTERVAL if last_word else 0.0))
                    commit, replace_previous = jz_commit_decision(
                        letter, confidence, sentence, last_letter, last_commit_time,
                        now, required_gap)
                    if commit:
                        if replace_previous:
                            sentence.pop()
                        sentence.append(letter)
                        last_letter = letter
                        last_word = None
                        last_commit_time = now
                        capture_effect = {"text": letter, "color": LETTER_EFFECT_COLOR, "time": now}
                        static_block_letters = set(JZ_START_SHAPES.get(letter, ()))
                        static_block_until = now + JZ_STATIC_BLOCK_SECONDS
            elif event == "too_short":
                current_display = "(movement too short, ignored)"
                current_confidence, current_threshold = 0.0, 0.0

            if not hand_present:
                stable_count = 0
                static_block_letters = set()
            elif segmenter.moving or event in ("started", "complete", "too_short"):
                # A gesture is in progress or just ended: no static guess this frame.
                stable_count = 0
                if segmenter.moving:
                    stability_fraction = min(1.0, len(segmenter.buffer) / SEQUENCE_LENGTH)
            else:
                stable_count += 1
                stability_fraction = min(1.0, stable_count / STABLE_FRAMES_FOR_LETTER)
                if (stable_count >= STABLE_FRAMES_FOR_LETTER
                        and frame_count % LETTER_PREDICT_EVERY_N_FRAMES == 0):
                    cnn_input = reshape_for_cnn(features)
                    probs = cnn_predict(cnn_input)
                    top_idx = int(np.argmax(probs))
                    confidence = float(probs[top_idx])
                    letter = letters_map[top_idx]

                    current_display = f"{letter} ({confidence:.2f})"
                    current_confidence = confidence
                    current_threshold = LETTER_CONF_THRESHOLD

                    blocked = letter in static_block_letters and now < static_block_until
                    required_gap = (letter_interval(last_letter) if last_letter
                                   else (WORD_INTERVAL if last_word else 0.0))
                    if (not blocked and confidence >= LETTER_CONF_THRESHOLD
                            and (now - last_commit_time) >= required_gap):
                        sentence.append(" " if letter == "SPACE" else letter)
                        last_letter = letter
                        last_word = None
                        last_commit_time = now
                        capture_effect = {"text": letter, "color": LETTER_EFFECT_COLOR, "time": now}

        # ---------------------------- HUD ----------------------------------
        if mode == "fingerspelling" and segmenter.moving:
            status_text = f"MOTION ({len(segmenter.buffer)} frames)"
        elif hand_present:
            label = detected_hand_label(results)
            status_text = f"HAND DETECTED ({label})" if label else "HAND DETECTED"
        else:
            status_text = "NO HAND"
        if mode == "words" and tracker.reference() is None:
            # Without the face, hand LOCATION can't be measured, so words that
            # differ only by location (FATHER vs MOTHER) can't be told apart.
            status_text = "FACE NOT FOUND - face the camera"

        # MediaPipe's handedness classification score doubles as a rough
        # hand-detection confidence, for the status badge's "- NN%" part
        # (matching the reference design's "HAND DETECTED - 96%" badge).
        if results.multi_handedness:
            hand_conf = float(results.multi_handedness[0].classification[0].score)
        else:
            hand_conf = 0.0

        if mode == "fingerspelling":
            # The big "Translated Text" display only updates when you press
            # ENTER (mirrors Words mode, where the display only swaps once a
            # whole word is recognized -- fingerspelling has no equivalent
            # automatic "word finished" signal, so ENTER is that signal).
            captured_text = fingerspell_result
            typing_text = f"Typing: {''.join(sentence)}" if sentence else "Typing: (press ENTER when done)"
            count_label = f"{len(sentence)} letter(s) typed"
        else:
            captured_text = captured_word or ""
            typing_text = None
            count_label = "1 word" if captured_word else "0 words"
        conf_label = f"{current_confidence:.0%} confidence" if current_threshold > 0 else "-"

        frame = draw_capture_effect(frame, capture_effect, now)

        canvas = compose_app_frame(
            frame,
            status_text=status_text, status_ok=hand_conf,
            modes=[("fingerspelling", "Finger Spelling"), ("words", "ASL Word")],
            active_mode=mode,
            result_label="Translated Text",
            result_big_text=captured_text,
            result_subtitle=(current_display if current_display else
                            ("Tap to enable camera" if not hand_present else "")),
            footer_left=count_label, footer_right=conf_label,
            typing_text=typing_text,
        )
        if SHOW_FPS:
            motion_text = (f"   motion: {motion:.3f} (start {MOTION_START_THRESHOLD}, end {MOTION_END_THRESHOLD})"
                           if mode == "fingerspelling" else "")
            cv2.putText(canvas, f"FPS: {fps:.0f}{motion_text}", (16, canvas.shape[0] - 16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("ASL Translator", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            if mode == "fingerspelling":
                sentence = []
                fingerspell_result = ""
            else:
                captured_word = ""
        elif key in (8, 127):  # backspace -- only edits the in-progress buffer
            if mode == "fingerspelling":
                if sentence:
                    sentence.pop()
            else:
                captured_word = ""
        elif key in (13, 10) and mode == "fingerspelling":  # ENTER: finalize the typed word
            if sentence:
                fingerspell_result = "".join(sentence)
                capture_effect = {"text": fingerspell_result, "color": WORD_EFFECT_COLOR, "time": now}
                sentence = []
        elif key == TRACKING_TOGGLE_KEY and mode == "fingerspelling":
            tracker.toggle()    # words mode always needs the face, so TAB only applies here
        elif key == MODE_TOGGLE_KEY:
            mode = "words" if mode == "fingerspelling" else "fingerspelling"
            frame_buffer.clear()
            segmenter.reset()
            static_block_letters = set()
            stable_count = 0
            current_display = ""
            current_confidence = 0.0
            smoother.reset()
            if mode == "words":
                # Words need the face (hand location is measured against it),
                # and a fresher face position than fingerspelling does.
                tracker.every_n = WORD_FACE_EVERY_N_FRAMES
                if not tracker.enabled:
                    tracker.toggle()
            else:
                tracker.every_n = FACE_EVERY_N_FRAMES

    reader.stop()
    cap.release()
    cv2.destroyAllWindows()
    hands_one.close()
    hands_two.close()
    tracker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    main(camera_index=args.camera)