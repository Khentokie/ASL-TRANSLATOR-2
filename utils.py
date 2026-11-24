"""Shared helpers: hand/face tracking, features, smoothing, camera, motion segmentation, screen drawing.
Hand features per frame: 126 floats = 2 hands x 21 landmarks x (x, y, z), slot 0 "Left", slot 1 "Right"
(zeros if missing), each hand made relative to its wrist and scaled by its own size."""
import platform
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

NUM_LANDMARKS = 21
COORDS = 3
FEATURES_PER_HAND = NUM_LANDMARKS * COORDS      # 63
TOTAL_FEATURES = FEATURES_PER_HAND * 2          # 126
SEQUENCE_LENGTH = 60                            # frames per recording
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 1920, 720, 30   # requested; open_camera() falls back
PHONE_WIDTH, PHONE_HEIGHT = 480, 640           # size of the on-screen window

# ------------------------------------------------------------------ hands
HAND_TRACK_COLOR = (255, 200, 120)   # BGR light blue
_hand_landmark_spec = mp_drawing.DrawingSpec(color=HAND_TRACK_COLOR, thickness=2, circle_radius=3)
_hand_connection_spec = mp_drawing.DrawingSpec(color=HAND_TRACK_COLOR, thickness=2)

def _normalize_single_hand(landmark_list):
    """(63,) landmarks relative to the wrist, scaled by the hand's own size."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmark_list], dtype=np.float32)
    pts -= pts[0].copy()
    scale = np.max(np.linalg.norm(pts, axis=1))
    return (pts / scale if scale > 1e-6 else pts).flatten()

def extract_landmarks(results, single_hand=False):
    """(126,) float32 hand features. single_hand=True keeps only the first hand."""
    left = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    right = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        pairs = list(zip(results.multi_hand_landmarks, results.multi_handedness))
        for hand_landmarks, handedness in (pairs[:1] if single_hand else pairs):
            normalized = _normalize_single_hand(hand_landmarks.landmark)
            if handedness.classification[0].label == "Left":
                left = normalized
            else:
                right = normalized
    return np.concatenate([left, right])

def draw_landmarks(image, results):
    """Draws the hand skeletons on the frame in place."""
    for hand_landmarks in results.multi_hand_landmarks or []:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                  _hand_landmark_spec, _hand_connection_spec)
    return image

def detected_hand_label(results):
    """'Left', 'Right', or None."""
    return results.multi_handedness[0].classification[0].label if results.multi_handedness else None

# ------------------------------------------------------------------ camera
BLACK_FRAME_MEAN_THRESHOLD = 2.0   # average pixel value at/below this counts as black

def _backend_candidates():
    """OS-appropriate capture backends, best first (one can fail where another works)."""
    system = platform.system()
    if system == "Windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if system == "Darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]

def _read_usable_frame(cap, attempts=30, delay=0.03):
    """True once a frame with a real picture arrives (skips warm-up black frames)."""
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0 and float(frame.mean()) > BLACK_FRAME_MEAN_THRESHOLD:
            return True
        time.sleep(delay)
    return False

def _try_open_index(index):
    """A VideoCapture for this index that really delivers a picture, or None."""
    for backend in _backend_candidates():
        try:
            cap = cv2.VideoCapture(index, backend)
        except Exception:
            continue
        if cap.isOpened() and _read_usable_frame(cap):
            return cap
        cap.release()
    return None

def list_cameras(max_index=8):
    """[(index, width, height)] for every camera index that gives a real picture."""
    working = []
    for i in range(max_index):
        if (cap := _try_open_index(i)):
            working.append((i, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            cap.release()
    return working

def open_camera(index=None, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS):
    """Opens a camera that gives a real picture (requested index first, then the others); MJPG at the
    target size/FPS, falling back to the camera's defaults if that breaks it. RuntimeError if none works."""
    indices_to_try = ([index] if index is not None else []) + [i for i in range(5) if i != index]
    cap = chosen_index = None
    for idx in indices_to_try:
        if (cap := _try_open_index(idx)):
            chosen_index = idx
            break
    if cap is None:
        raise RuntimeError(
            f"Could not get a real picture from any camera (tried indices {indices_to_try}). "
            "Cameras that only return a black image are skipped. Things to check: "
            "(1) a phone/iPad webcam app must be running and already streaming BEFORE you start "
            "this script; (2) close other apps that might be using the camera (Zoom, Teams, browser "
            "tabs, the Windows Camera app); (3) Windows Settings > Privacy & security > Camera must "
            "allow desktop apps; (4) check the lens cover/privacy shutter. Run with --list-cameras "
            "to see which indices respond, then --camera N to pick one.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not _read_usable_frame(cap):
        print(f"Camera {chosen_index}: requested MJPG {width}x{height}@{fps} didn't work, "
              "falling back to its default settings.")
        cap.release()
        cap = _try_open_index(chosen_index)
        if cap is None:
            raise RuntimeError(f"Camera {chosen_index} stopped responding after falling back.")
    print(f"Camera {chosen_index}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps (driver-reported)")
    return cap

def to_phone_frame(frame, width=PHONE_WIDTH, height=PHONE_HEIGHT):
    """Center-crops to the target aspect ratio, then resizes (display only)."""
    h, w = frame.shape[:2]
    ratio = width / height
    crop_w, crop_h = (max(1, int(h * ratio)), h) if w / h > ratio else (w, max(1, int(w / ratio)))
    x0, y0 = (w - crop_w) // 2, (h - crop_h) // 2
    return cv2.resize(frame[y0:y0 + crop_h, x0:x0 + crop_w], (width, height), interpolation=cv2.INTER_AREA)

class EMASmoother:
    """Exponential moving average to steady landmark jitter; an all-zero vector (hand gone) resets it."""
    def __init__(self, size, alpha=0.5):
        self.alpha = alpha
        self._state = np.zeros(size, dtype=np.float32)
        self._have_state = False
    def reset(self):
        self._have_state = False
    def smooth(self, features):
        if not np.any(features):
            self.reset()
            return features
        self._state = (self.alpha * features + (1 - self.alpha) * self._state
                       if self._have_state else features.copy())
        self._have_state = True
        return self._state

# ------------------------------------------------------------------ face
FACE_KEY_LANDMARKS = (33, 263, 1, 61, 291, 152)  # eyes, nose tip, mouth corners, chin
FACE_FEATURES = 4 + len(FACE_KEY_LANDMARKS) * 2  # 16: face box + 6 key points
FACE_MIN_SIZE = 1e-4                             # guards against a ~zero eye distance

def extract_face(face_results):
    """(16,) compact face vector, relative to the eyes and scaled by their distance. Zeros if no face."""
    out = np.zeros(FACE_FEATURES, dtype=np.float32)
    if face_results is None or not face_results.multi_face_landmarks:
        return out
    flm = face_results.multi_face_landmarks[0].landmark
    cx, cy = (flm[33].x + flm[263].x) / 2.0, (flm[33].y + flm[263].y) / 2.0
    scale = float(np.hypot(flm[33].x - flm[263].x, flm[33].y - flm[263].y))
    if scale < FACE_MIN_SIZE:
        return out
    xs, ys = [p.x for p in flm], [p.y for p in flm]
    face = [((min(xs) + max(xs)) / 2 - cx) / scale, ((min(ys) + max(ys)) / 2 - cy) / scale,
            (max(xs) - min(xs)) / scale, (max(ys) - min(ys)) / scale]
    for i in FACE_KEY_LANDMARKS:
        face += [(flm[i].x - cx) / scale, (flm[i].y - cy) / scale]
    out[:FACE_FEATURES] = face[:FACE_FEATURES]
    return out

def _dashed_line(frame, p1, p2, color, thickness, dash, gap):
    (x1, y1), (x2, y2) = p1, p2
    length = int(np.hypot(x2 - x1, y2 - y1))
    for start in range(0, length, dash + gap):
        a, b = start / length, min(start + dash, length) / length
        cv2.line(frame, (int(x1 + (x2 - x1) * a), int(y1 + (y2 - y1) * a)),
                 (int(x1 + (x2 - x1) * b), int(y1 + (y2 - y1) * b)), color, thickness, cv2.LINE_AA)

def draw_face_overlay(frame, face_results):
    """Dashed square around each detected face, in the app's light blue."""
    if face_results is None or not face_results.multi_face_landmarks:
        return frame
    h, w = frame.shape[:2]
    for flm in face_results.multi_face_landmarks:
        xs, ys = [p.x * w for p in flm.landmark], [p.y * h for p in flm.landmark]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        side = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.1       # a square, slightly padded
        x1, y1, x2, y2 = (int(round(v)) for v in (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        dash = max(6, int(side / 12))
        for a, b in zip(corners, corners[1:] + corners[:1]):
            _dashed_line(frame, a, b, HAND_TRACK_COLOR, max(2, int(side / 90)), dash, dash // 2)
    return frame

# Hand LOCATION relative to the face (words). Hand features are wrist-relative, so they lose WHERE the
# hand is -- and some words differ only by place (FATHER: thumb at the forehead, MOTHER: at the chin).
# Each word frame also holds the hands' position from the face (origin = midpoint of the eyes, unit =
# distance between the eyes). Per frame: 4 points x (x, y) per hand + a face-found flag.
LOCATION_POINTS = (0, 4, 8, 9)                        # wrist, thumb tip, index tip, palm
LOCATION_FEATURES = 2 * len(LOCATION_POINTS) * 2 + 1  # 17
LOCATION_SCALE = 0.25                                 # keeps values near the hand features' range
LOCATION_CLIP = 2.0                                   # after scaling (+-8 eye-distances)
WORD_FEATURES = TOTAL_FEATURES + LOCATION_FEATURES    # 143 per word frame
FACE_REF_LANDMARKS = (33, 133, 362, 263)              # eye corners
FACE_REF_HOLD_FRAMES = 30   # keep the last face position this long if it's briefly lost

def face_reference(face_results):
    """(center_x, center_y, eye_distance) in image coordinates, or None."""
    if face_results is None or not face_results.multi_face_landmarks:
        return None
    flm = face_results.multi_face_landmarks[0].landmark
    cx, cy = np.array([[flm[i].x, flm[i].y] for i in FACE_REF_LANDMARKS], dtype=np.float32).mean(axis=0)
    scale = float(np.hypot(flm[33].x - flm[263].x, flm[33].y - flm[263].y))
    return None if scale < FACE_MIN_SIZE else (float(cx), float(cy), scale)

def extract_location(results, face_ref):
    """(17,) hand locations relative to the face, in extract_landmarks()'s slots; all zeros if no face."""
    out = np.zeros(LOCATION_FEATURES, dtype=np.float32)
    if face_ref is None:
        return out
    cx, cy, scale = face_ref
    out[-1] = 1.0
    slot_size = len(LOCATION_POINTS) * 2
    for hand_landmarks, handedness in zip(results.multi_hand_landmarks or [], results.multi_handedness or []):
        slot = 0 if handedness.classification[0].label == "Left" else 1
        vals = []
        for i in LOCATION_POINTS:
            vals += [(hand_landmarks.landmark[i].x - cx) / scale, (hand_landmarks.landmark[i].y - cy) / scale]
        out[slot * slot_size:(slot + 1) * slot_size] = np.clip(
            np.array(vals, dtype=np.float32) * LOCATION_SCALE, -LOCATION_CLIP, LOCATION_CLIP)
    return out

class FaceTracker:
    """Face Mesh on the frames you feed it (every_n: only every Nth, reusing the result in
    between): compact face features, face motion (0 = still) and the face position for hands."""
    def __init__(self, enabled=True, every_n=2):
        self.enabled = enabled
        self.every_n = max(1, int(every_n))
        self.face = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                                          min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.features = np.zeros(FACE_FEATURES, dtype=np.float32)
        self.motion = 0.0
        self._face_results = self._prev_features = self._ref = None   # _ref: last good (cx, cy, scale)
        self._frame_index = self._ref_age = 0                         # _ref_age: frames since refreshed
    def reference(self):
        """Face position (cx, cy, scale) or None; survives brief dropouts."""
        return self._ref if self.enabled else None
    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            self._face_results = self._prev_features = self._ref = None
            self._ref_age = 0
            self.features = np.zeros(FACE_FEATURES, dtype=np.float32)
            self.motion = 0.0
        return self.enabled
    def process(self, rgb):
        """Feed one RGB frame; returns the (16,) face features."""
        if not self.enabled:
            return self.features
        if self._frame_index % self.every_n == 0:
            self._face_results = self.face.process(rgb)
            ref = face_reference(self._face_results)
            if ref is not None:
                self._ref, self._ref_age = ref, 0
            else:
                self._ref_age += self.every_n
                if self._ref_age > FACE_REF_HOLD_FRAMES:
                    self._ref = None
        self._frame_index += 1
        self.features = extract_face(self._face_results)
        both = self._prev_features is not None and np.any(self.features) and np.any(self._prev_features)
        self.motion = float(np.mean(np.abs(self.features - self._prev_features))) if both else 0.0
        self._prev_features = self.features.copy()
        return self.features
    def draw(self, frame):
        if self.enabled:
            draw_face_overlay(frame, self._face_results)
        return frame
    def close(self):
        self.face.close()

# ------------------------------------------------------------------ themed screen
HUD_PRIMARY_BLUE, HUD_NAVY, HUD_LIGHT_BG = (255, 100, 30), (60, 25, 10), (248, 240, 232)   # BGR
HUD_WHITE_TEXT, HUD_DARK_TEXT, HUD_MUTED_TEXT = (255, 255, 255), (40, 28, 20), (170, 140, 120)
HUD_ACCENT_GOOD, HUD_ACCENT_WARN, HUD_ACCENT_BAD = (120, 200, 30), (0, 170, 255), (70, 70, 230)  # green, amber, red

def _put(frame, text, org, scale, color, thickness=1):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def rounded_rect(frame, pt1, pt2, color, radius=18, thickness=-1):
    """Rounded rectangle (filled if thickness < 0, else an outline)."""
    (x1, y1), (x2, y2) = pt1, pt2
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    centers = [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]
    if thickness < 0:
        cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), color, -1)
        for c in centers:
            cv2.circle(frame, c, r, color, -1)
    else:
        for a, b in [((x1 + r, y1), (x2 - r, y1)), ((x1 + r, y2), (x2 - r, y2)),
                     ((x1, y1 + r), (x1, y2 - r)), ((x2, y1 + r), (x2, y2 - r))]:
            cv2.line(frame, a, b, color, thickness)
        for c, angle in zip(centers, (180, 270, 90, 0)):
            cv2.ellipse(frame, c, (r, r), angle, 0, 90, color, thickness)
    return frame

def draw_pill(frame, text, center_x, top_y, bg_color, text_color=HUD_WHITE_TEXT,
              font_scale=0.65, thickness=2, pad_x=22, pad_y=12):
    """Centered pill-shaped badge sized to its text. Returns its height."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    w, h = tw + pad_x * 2, th + pad_y * 2
    x1, x2, y2 = int(center_x - w / 2), int(center_x + w / 2), top_y + h
    rounded_rect(frame, (x1, top_y), (x2, y2), bg_color, radius=h // 2)
    _put(frame, text, (x1 + pad_x, y2 - pad_y + 2), font_scale, text_color, thickness)
    return h

def draw_mode_tabs(frame, pt1, pt2, modes, active_mode, active_color=HUD_PRIMARY_BLUE,
                   bg_color=(40, 20, 10), text_color=HUD_WHITE_TEXT):
    """Row of pill buttons for `modes` [(key, label)], highlighting the active one."""
    (x1, y1), (x2, y2) = pt1, pt2
    rounded_rect(frame, pt1, pt2, bg_color, radius=(y2 - y1) // 2)
    gap = 12
    btn_w = (x2 - x1 - gap * (len(modes) + 1)) // len(modes)
    cx = x1 + gap
    for key, label in modes:
        active = key == active_mode
        rounded_rect(frame, (cx, y1 + 6), (cx + btn_w, y2 - 6), active_color if active else bg_color,
                     radius=(y2 - y1 - 12) // 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        _put(frame, label, (cx + (btn_w - tw) // 2, y1 + (y2 - y1 + th) // 2), 0.6,
             text_color if active else HUD_MUTED_TEXT, 2)
        cx += btn_w + gap
    return frame

def _draw_camera_card(canvas, camera_frame, badge_text, badge_color, badge_scale, top_frac, bottom_frac, gap):
    """Navy card with a status pill and the camera inset. Returns (margin, card_x1, card_x2, card_y2)."""
    height, width = canvas.shape[:2]
    margin = int(width * 0.045)
    card_x1, card_y1 = margin, int(height * top_frac)
    card_x2, card_y2 = width - margin, int(height * bottom_frac)
    rounded_rect(canvas, (card_x1, card_y1), (card_x2, card_y2), HUD_NAVY, radius=28)
    badge_h = draw_pill(canvas, badge_text, width // 2, card_y1 + 18, bg_color=badge_color, font_scale=badge_scale)
    inset = int(width * 0.03)
    x1, x2 = card_x1 + inset, card_x2 - inset
    y1, y2 = card_y1 + badge_h + gap, card_y2 - inset
    if x2 - x1 > 0 and y2 - y1 > 0:
        canvas[y1:y2, x1:x2] = to_phone_frame(camera_frame, width=x2 - x1, height=y2 - y1)
        rounded_rect(canvas, (x1, y1), (x2, y2), (255, 255, 255), radius=20, thickness=2)
    return margin, card_x1, card_x2, card_y2

def compose_app_frame(camera_frame, status_text, status_ok, modes, active_mode, result_label,
                      result_big_text, result_subtitle, footer_left, footer_right, typing_text=None,
                      width=PHONE_WIDTH, height=PHONE_HEIGHT):
    """The full screen for realtime_app.py (camera card, mode tabs, results panel). Use it
    INSTEAD of to_phone_frame(). A float status_ok adds " - NN%" to the badge."""
    canvas = np.full((height, width, 3), HUD_PRIMARY_BLUE, dtype=np.uint8)
    # cv2's built-in fonts are ASCII only (a "·" would show as "??"), hence the hyphen.
    badge = f"{status_text} - {int(status_ok * 100)}%" if isinstance(status_ok, float) else status_text
    margin, card_x1, card_x2, card_y2 = _draw_camera_card(
        canvas, camera_frame, badge, HUD_PRIMARY_BLUE, 0.6, 0.1, 0.62, 36)
    tabs_y1 = card_y2 + 18
    tabs_y2 = tabs_y1 + int(height * 0.055)
    draw_mode_tabs(canvas, (card_x1, tabs_y1), (card_x2, tabs_y2), modes, active_mode)
    panel_y1, panel_y2 = tabs_y2 + 20, height - margin
    rounded_rect(canvas, (card_x1, panel_y1), (card_x2, panel_y2), HUD_LIGHT_BG, radius=28)
    px, py = card_x1 + 28, panel_y1 + 46
    _put(canvas, result_label.upper(), (px, py), 0.55, HUD_PRIMARY_BLUE, 2)
    if typing_text:
        py += 30
        _put(canvas, typing_text, (px, py), 0.48, HUD_MUTED_TEXT)
    py += 56
    big = result_big_text or "-"
    _put(canvas, big, (px, py), 1.1 if len(big) < 12 else max(0.4, 1.1 * 12 / len(big)), HUD_DARK_TEXT, 3)
    if result_subtitle:
        _put(canvas, result_subtitle, (px, py + 46), 0.65, HUD_DARK_TEXT, 2)
    foot_y = panel_y2 - 24
    _put(canvas, footer_left, (px, foot_y), 0.55, HUD_MUTED_TEXT)
    (fw, _), _ = cv2.getTextSize(footer_right, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    _put(canvas, footer_right, (card_x2 - 28 - fw, foot_y), 0.55, HUD_MUTED_TEXT)
    return canvas

def compose_collection_frame(camera_frame, status_text, status_color, info_lines,
                             width=PHONE_WIDTH, height=PHONE_HEIGHT):
    """The simpler screen for data_collection.py: camera card plus an info panel."""
    canvas = np.full((height, width, 3), HUD_PRIMARY_BLUE, dtype=np.uint8)
    margin, card_x1, card_x2, card_y2 = _draw_camera_card(
        canvas, camera_frame, status_text, status_color, 0.55, 0.08, 0.78, 30)
    panel_y1, panel_y2 = card_y2 + 18, height - margin
    rounded_rect(canvas, (card_x1, panel_y1), (card_x2, panel_y2), HUD_LIGHT_BG, radius=24)
    for n, line in enumerate(info_lines):
        _put(canvas, line, (card_x1 + 24, panel_y1 + 36 + 28 * n), 0.5, HUD_DARK_TEXT)
    return canvas

# ------------------------------------------------------------------ motion (J and Z)
# J and Z can't be told apart from one frame: the live app waits for the hand to START moving, collects
# frames until it SETTLES, then asks the J/Z model. Movement score = hand SHAPE change + how far the hand
# TRAVELED (wrist, in hand-sizes) -- needed since landmarks are wrist-relative and tracing a Z changes none.
MOTION_START_THRESHOLD = 0.03    # score that counts as "started moving"
MOTION_END_THRESHOLD = 0.015     # score below this counts as "holding still"
MOTION_END_HOLD_FRAMES = 4       # ...for this many frames in a row = gesture over
JZ_MIN_FRAMES = 8                # a gesture needs this many frames of real movement
MOTION_WINDOW = 5                # frames of history the score looks at
TRIM_FEATURE_THRESHOLD = 0.008   # shape-only threshold for trimming recordings

def hand_anchor(results):
    """(wrist_x, wrist_y, hand_size) of the first hand in image coordinates, or None."""
    if not results.multi_hand_landmarks:
        return None
    pts = np.array([[lm.x, lm.y] for lm in results.multi_hand_landmarks[0].landmark], dtype=np.float32)
    return (float(pts[0][0]), float(pts[0][1]), float(np.max(np.linalg.norm(pts - pts[0], axis=1))))

class MotionSegmenter:
    """STILL <-> MOVING state machine that collects each motion gesture. update(features, anchor) ->
    (event, sequence); event is None, "started", "moving", "complete" (sequence = the gesture's frames)
    or "too_short"/"lost" (discarded). Start/end thresholds plus an end-hold stop a pause splitting it."""
    def __init__(self, start_threshold=MOTION_START_THRESHOLD, end_threshold=MOTION_END_THRESHOLD,
                 end_hold_frames=MOTION_END_HOLD_FRAMES, min_frames=JZ_MIN_FRAMES,
                 max_frames=SEQUENCE_LENGTH, window=MOTION_WINDOW):
        self.start_threshold, self.end_threshold = start_threshold, end_threshold
        self.end_hold_frames, self.min_frames = end_hold_frames, min_frames
        self.max_frames, self._window = max_frames, window
        self.reset()
    def reset(self):
        self._features = deque(maxlen=self._window)
        self._anchors = deque(maxlen=self._window)
        self.moving = False
        self.buffer = []
        self.motion = 0.0
        self._still_count = 0
        self._active_count = 0     # frames of real movement in this gesture
    def _score(self):
        if len(self._features) < 2:
            return 0.0
        arr = np.array(list(self._features))
        shape_change = float(np.mean(np.abs(arr[1:] - arr[:-1])))
        travel = 0.0
        anchors = [a for a in self._anchors if a is not None]
        if len(anchors) >= 2:
            (x0, y0, s0), (x1, y1, s1) = anchors[0], anchors[-1]
            # Net displacement, not summed steps, so tracking jitter cancels out.
            travel = float(np.hypot(x1 - x0, y1 - y0) / max((s0 + s1) / 2.0, 1e-6) / (len(anchors) - 1))
        return shape_change + travel
    def update(self, features, anchor):
        if features is None or not np.any(features):
            was_moving = self.moving
            self.reset()
            return ("lost", None) if was_moving else (None, None)
        self._features.append(features)
        self._anchors.append(anchor)
        self.motion = self._score()
        if not self.moving:
            if self.motion >= self.start_threshold:
                self.moving = True
                self.buffer = list(self._features)     # includes the lead-in frames
                self._still_count, self._active_count = 0, 1
                return ("started", None)
            return (None, None)
        self.buffer.append(features)
        if self.motion < self.end_threshold:
            self._still_count += 1
        else:
            self._still_count = 0
            self._active_count += 1
        if self._still_count >= self.end_hold_frames or len(self.buffer) >= self.max_frames:
            seq = self.buffer[:len(self.buffer) - max(0, self._still_count - 1)]   # drop the still tail
            active = self._active_count
            self.reset()    # fresh history so this motion can't instantly re-trigger
            # Count frames of real MOVEMENT, not length (lead-in + score lag already pad any trigger to ~10).
            return ("complete", seq) if active >= self.min_frames else ("too_short", None)
        return ("moving", None)

def trim_to_motion(seq, threshold=TRIM_FEATURE_THRESHOLD):
    """Trims near-still frames off both ends of a (T, 126) recording, like MotionSegmenter does live."""
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) < 2:
        return None
    present = np.any(seq != 0, axis=1)
    diffs = np.zeros(len(seq), dtype=np.float32)
    diffs[1:] = np.mean(np.abs(seq[1:] - seq[:-1]), axis=1)
    diffs[1:][~(present[1:] & present[:-1])] = 0.0   # ignore frames where the hand dropped out
    active = np.where(diffs > threshold)[0]
    if len(active) == 0:
        return None
    return seq[max(0, int(active[0]) - 1):int(active[-1]) + 1]