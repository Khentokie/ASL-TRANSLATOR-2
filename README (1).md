# ASL Translator

Real-time American Sign Language recognizer: **MediaPipe** for hand/body/face
tracking, **TensorFlow/Keras** for recognition, OpenCV for the live app.

## Architecture

Two modes, toggled with `m`:

### Fingerspelling mode (hybrid static + motion)

```
Camera -> single-hand detection -> landmarks -> smoothing
                                        |
                              moving or still?
                 still                              moving
                   |                                   |
          static CNN classifier              buffer frames while moving
        (A-Y except J/Z, + SPACE)                      |
                   |                         motion settles or buffer fills
             commit letter                             |
                                            long enough? ---- no --> discard
                                                  | yes
                                        small LSTM: J or Z?
                                                  |
                                            commit letter
```

- **Static letters** (A-Y minus J/Z, plus a SPACE gesture) are recognized
  the instant your hand holds still for a few frames -- a small CNN
  classifies the single pose. Trained by `train_cnn_letters.py`.
- **J and Z** are motion letters a single frame can't capture. While your
  hand is moving, frames are buffered; once the motion settles (or the
  buffer fills), a separate LSTM looks at the whole sequence and decides
  J or Z. Too-short or interrupted motion is discarded rather than forced
  into a guess. Trained by `train_lstm_jz.py`.
- Only **one hand** is tracked in this mode (either hand), so an incidental
  second hand in frame never confuses recognition.

### Words mode

An LSTM reads the last ~1-2 seconds (both hands tracked) and predicts
`hello` / `thankyou` / `father` / `mother`, or any word you've added via
`data_collection.py`. Only the single most recent word is shown -- a new
recognition replaces the old one rather than piling up. Trained by
`train_lstm_words.py`.

**Location matters for words.** Many signs differ only by *where* the hand
is: FATHER (thumb at the forehead) and MOTHER (thumb at the chin) are the
same handshape making the same movement. Hand shape features are measured
relative to the wrist, which discards position, so on their own they can't
separate those two. Every word frame therefore also records where the hands
are relative to your **face** (wrist, thumb tip, index fingertip and palm,
measured from the midpoint of your eyes in units of the distance between
your eyes, so it doesn't depend on how far you sit from the camera). Chest,
shoulder etc. positions sit at predictable distances below the face in those
same units, so no separate body tracker is needed. Each word frame is 143
numbers: 126 hand-shape + 17 location.

What this means in practice:
- **Keep your whole face in view** while recording and while signing words.
  The window says `FACE NOT FOUND` if it's lost, and a recording where the
  face was missing prints a warning. If a hand briefly covers the face the
  last known face position is kept for about a second.
- Face tracking is always on in Words mode (`TAB` only applies to
  Fingerspelling).
- **Words recorded before this change must be re-recorded** (they have no
  location data and can't be converted). `train_lstm_words.py` skips them and
  says so, and `realtime_app.py` refuses an old words model with an
  explanation.
- Record with the same hand you'll sign with (left and right hands use
  separate slots), at a normal sitting distance, and vary your position a
  little between takes.

### Shared pipeline

- **Hand tracking**: MediaPipe Hands, 21 landmarks/hand, normalized
  relative to the wrist so recognition doesn't depend on hand size or
  distance from the camera.
- **Body tracking**: MediaPipe Pose (shoulders, elbows, wrists), shown live
  for reference; doesn't affect recognition.
- **Face tracking**: MediaPipe Face Mesh (468 points), shown live with a
  bounding box; also doesn't affect recognition.
- **Landmark smoothing**: an exponential moving average reduces frame-to-
  frame jitter before features reach any model, for steadier tracking.
- **HUD**: a plain light-cyan panel with thin black outlines on every
  letter for readability, showing mode, hand status, live FPS, the current
  prediction, its confidence, and a "stability" bar (how close the current
  hold/motion is to actually being accepted).
- **Phone-sized window**: the final picture is cropped/resized to a
  portrait window sized like a typical phone screen, so the app can be
  measured against one even though it runs on a desktop webcam. Detection
  always runs on the full camera frame first -- this crop never costs
  accuracy, only changes what's displayed.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Collect training data

Both commands record **motion sequences** (3-2-1 countdown, then
automatic capture), saved as `.npy` files:

```bash
python data_collection.py --mode letters
```
Press `A`-`Z` to record that letter (single hand only), `SPACE` for the
word-separator gesture, `TAB` to toggle the body/face overlay, `q` to quit.
Record J and Z like any other letter -- the *recording* step is identical
for static and motion letters; which model uses that data is decided at
training time.

```bash
python data_collection.py --mode words
```
`SPACE` records a sequence for the current word, `n` cycles words, `a`
lets you type a brand-new word's name on the spot (no code editing
needed), `TAB` toggles tracking, `q` quits.

Aim for 60-100+ recordings per static letter, 30+ per motion letter
(J/Z), and 30-40+ per word (more for words that differ only by location).

## 2. Train the three models

```bash
python train_cnn_letters.py     # static letters A-Y (minus J/Z) + SPACE
python train_lstm_jz.py         # motion letters J and Z
python train_lstm_words.py      # words
```

Each prints validation accuracy and saves its model + label map to
`models/`.

## 3. Run the real-time translator

```bash
python realtime_app.py
# or, to pick a different camera:
python realtime_app.py --camera 1
```

| Key | Action |
|---|---|
| `m` | Toggle Fingerspelling / Words mode |
| `TAB` | Toggle body/face tracking overlay |
| `c` | Clear the sentence / captured word |
| `BACKSPACE` | Delete last letter / clear captured word |
| `q` | Quit |

## J and Z (motion letters)

J and Z can't be told from a single frame, so while Fingerspelling the app
watches for your hand to **start moving**, collects frames until it
**settles**, and only then asks the J/Z model what it saw. "Movement" counts
both the hand changing shape *and* the whole hand travelling across the
picture (tracing a Z moves the hand without changing its shape, which the
wrist-relative landmarks alone can't see).

**Recording J and Z:** `python data_collection.py --mode letters`, press `J`
or `Z`, and perform the full motion while it says RECORDING. Record plenty
of takes (30+ each) at different speeds, positions and distances.

**The J/Z model has three answers: J, Z and OTHER.** OTHER ("some movement
that isn't J or Z") is what stops ordinary hand movement -- like moving to
the next letter -- from being forced into a J or Z. You don't record it:
`train_lstm_jz.py` builds it automatically from your static-letter
recordings, so record some static letters before training.

**What you'll see live:** the bottom-left shows `motion: 0.0xx`. It should
stay near zero while your hand is still and climb past the start threshold
when you sign J or Z; while a gesture is being collected the status badge
reads `MOTION (n frames)`. If it never triggers, lower
`MOTION_START_THRESHOLD` in `utils.py`; if it triggers while you're holding
still, raise it.

**Start/end poses:** J begins as an "I" and Z as a pointing finger, so the
static recognizer may type that starting letter an instant before the
motion is recognized. A J/Z result replaces it automatically, and those
static letters are blocked for a few seconds afterwards (the hand also
*ends* near that shape). Both behaviours are configured by `JZ_START_SHAPES`
in `realtime_app.py`.

## Tuning

Most knobs live near the top of `realtime_app.py`:

| Constant | Effect |
|---|---|
| `LETTER_CONF_THRESHOLD` / `JZ_CONFIDENCE_THRESHOLD` / `WORD_CONF_THRESHOLD` | Higher = fewer false positives, more misses |
| `MOTION_START_THRESHOLD` / `MOTION_END_THRESHOLD` / `MOTION_END_HOLD_FRAMES` (in `utils.py`) | How much movement counts as starting/ending a J/Z gesture -- tune using the on-screen `motion:` number |
| `JZ_MIN_FRAMES` (in `utils.py`) | A gesture needs this many frames of real movement or it's ignored as a twitch |
| `STABLE_FRAMES_FOR_LETTER` | How long a static pose must hold before it's evaluated |
| `LETTER_INTERVALS` / `DEFAULT_LETTER_INTERVAL` / `WORD_INTERVAL` | Pause after a recognition before the next one is accepted (per-letter overridable) |
| `HAND_MODEL_COMPLEXITY`, `*_PREDICT_EVERY_N_FRAMES`, `BODY_FACE_*_EVERY_N` | Speed/accuracy trade-offs |

`data_collection.py` shares `SEQUENCE_LENGTH`, camera settings, and the
phone-sized window constants from `utils.py`.

## Notes / limitations

- This is a capable starter system, not a production-grade recognizer --
  accuracy depends heavily on how much and how varied your collected data
  is (lighting, background, hand size, distance from camera, signer).
- Data from one signer in one setup will overfit to that person; for a
  generalizable model, collect from multiple people/sessions.
- `migrate_legacy_letters.py` can convert very old single-frame CSV
  captures into the current folder format if you have any left over, but
  that data won't work for J/Z (which need real motion sequences, not a
  pile of static frames).