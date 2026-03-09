# Custom Dataset Guide

Complete guide for recording and integrating custom sound classes into your household sound monitor.

**Authors:** Joshua Kirby & Alan Nur
**Course:** TECHIN 513A — Managing Data And Signal Processing

---

## Table of Contents

1. [Overview](#overview)
2. [Naming Conventions](#naming-conventions)
3. [Step-by-Step Recording Guide](#step-by-step-recording-guide)
4. [Recording Best Practices](#recording-best-practices)
5. [Troubleshooting](#troubleshooting)
6. [Integration with Training Pipeline](#integration-with-training-pipeline)

---

## Overview

### What Are Custom Datasets?

Custom datasets allow you to train the classifier on sounds **not included in the ESC-50 dataset**. The base ESC-50 dataset covers 50 environmental sound classes (dog barking, sirens, water drops, etc.), but household-specific sounds like:

- `chopping` (knife on cutting board)
- `boiling_water` (pot on stove)
- `frying` (food in pan)
- `running_faucet` (water tap)
- `electric_fan` (oscillating fan)

...may not be represented or may need more specialized training data.

### Why Naming Conventions Matter

The training pipeline **automatically parses directory names as class labels**. Incorrect naming will cause:
- ❌ Training errors (invalid characters in class names)
- ❌ Inference failures (mismatched labels)
- ❌ Poor organization (hard to find/manage classes)

### Expected Outcome

After following this guide, you'll have:
- ✅ High-quality custom audio dataset
- ✅ Properly structured directories
- ✅ Dataset ready for Colab training
- ✅ Integrated classifier with ESC-50 + custom classes

---

## Naming Conventions

### 🔑 CRITICAL: Follow These Rules Exactly

#### Class Names (Directory Names)

**Format:** `lowercase_with_underscores`

**✅ GOOD Examples:**
```
chopping
boiling_water
frying_food
running_faucet
electric_fan
door_closing
microwave_beep
coffee_grinder
blender_running
dishwasher_running
```

**❌ BAD Examples:**
```
Chopping              # Capital letters not allowed
boiling water         # Spaces not allowed (use underscores)
frying-food           # Hyphens not allowed (use underscores)
sound1                # Not descriptive enough
test_audio            # Generic names make organization hard
my_custom_sound.wav   # .wav extension in directory name
```

**Rules:**
1. **All lowercase** (a-z)
2. **Numbers allowed** (0-9)
3. **Only underscores** for word separation (no spaces, hyphens, or other special characters)
4. **Descriptive and specific** (not "sound1", "audio2", etc.)
5. **No file extensions** in directory names
6. **Filesystem-safe** (no `/`, `\`, `:`, `*`, `?`, `"`, `<`, `>`, `|`)

#### File Names (Audio Files)

**Auto-Generated Format** (when using `record_samples.py`):
```
<class_name>_YYYYMMDD_HHMMSS.wav
```

**Examples:**
```
chopping_20260308_143522.wav
chopping_20260308_143531.wav
boiling_water_20260308_144102.wav
frying_food_20260308_145633.wav
```

**Manual Recording Format** (if not using the script):
- **Must be:** `.wav` format (uncompressed PCM audio)
- **Can be:** Any filename (e.g., `clip001.wav`, `sample.wav`, etc.)
- **Will work:** As long as they're in the correct class directory

**Rules:**
1. **Must be `.wav` format** (not `.mp3`, `.m4a`, `.ogg`, etc.)
2. **Must be exactly 5 seconds** long (44,100 samples/sec × 5 = 220,500 samples)
3. **Must be mono** (single channel)
4. **Must be 44.1 kHz** sample rate
5. **Should be 16-bit** PCM encoding

### Directory Structure

**Required Structure:**
```
menial_ai/
└── data/
    └── custom/
        ├── chopping/
        │   ├── chopping_20260308_143522.wav
        │   ├── chopping_20260308_143531.wav
        │   ├── chopping_20260308_143540.wav
        │   └── ... (20+ total files)
        │
        ├── boiling_water/
        │   ├── boiling_water_20260308_144102.wav
        │   ├── boiling_water_20260308_144111.wav
        │   └── ... (20+ total files)
        │
        ├── frying_food/
        │   └── ... (20+ files)
        │
        ├── running_faucet/
        │   └── ... (20+ files)
        │
        └── electric_fan/
            └── ... (20+ files)
```

**Rules:**
1. **Root directory:** `data/custom/` (fixed location)
2. **One subdirectory per class** (directory name = class label)
3. **All `.wav` files in a directory = training samples for that class**
4. **Minimum 20 clips per class** (recommended: 50-100 for best results)
5. **No nested subdirectories** (flat structure within each class)

---

## Step-by-Step Recording Guide

### Phase 1: Environment Setup

#### Step 1: Create Virtual Environment (if not already done)

```bash
cd menial_ai
python3.12 -m venv .venv312
source .venv312/bin/activate  # On Windows: .venv312\Scripts\activate
```

#### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Expected packages:
- `sounddevice` (microphone recording)
- `numpy` (audio processing)
- `scipy` (signal processing)

#### Step 3: Test Microphone Access

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

**Expected output:** List of available audio devices
**Look for:** Your microphone (input device) - note its index number

**If error:** "No module named sounddevice"
```bash
pip install sounddevice
```

#### Step 4: Create Data Directories

```bash
mkdir -p data/custom
```

**Verify:**
```bash
ls -la data/
```

Should show `custom/` directory.

#### Step 5: Test Recording Script

```bash
python record_samples.py --help
```

**Expected output:**
```
usage: record_samples.py <class_name>
Records 5-second audio clips for a given sound class.
```

#### Step 6: Verify Folder Structure

```bash
tree data/ -L 2
# Or if tree not installed:
find data/ -type d
```

---

### Phase 2: Planning Custom Classes

#### Step 7: Choose Sounds Not in ESC-50

**ESC-50 includes these household sounds:**
- `dog`, `cat`, `pig`, `cow`, `frog`, `hen`, `insects`, `sheep`, `crow`, `rooster` (animals)
- `crying_baby`, `sneezing`, `coughing`, `footsteps`, `breathing`, `laughing`, `brushing_teeth`, `snoring` (human)
- `door_wood_knock`, `door_wood_creaks`, `glass_breaking`, `clock_tick`, `clock_alarm` (objects)
- `vacuum_cleaner`, `washing_machine`, `keyboard_typing`, `mouse_click`, `can_opening` (appliances)
- `water_drops`, `pouring_water`, `toilet_flush`, `crackling_fire` (liquids/fire)

**Good candidates for custom classes:**
- Kitchen: `chopping`, `boiling_water`, `frying_food`, `microwave_beep`, `coffee_grinder`, `blender_running`, `toaster_pop`
- Water: `running_faucet`, `shower_running`, `bathtub_filling`
- HVAC: `electric_fan`, `air_conditioner`, `heater_running`
- Doors: `door_closing`, `door_opening`, `doorbell_ring`
- Appliances: `dishwasher_running`, `dryer_running`, `refrigerator_hum`

#### Step 8: Define Clear Class Boundaries

**Avoid overlap between classes:**
- ✅ GOOD: `boiling_water` (vigorous bubbling) vs `simmering_water` (gentle bubbles)
- ❌ BAD: `water_sound` (too vague - could be faucet, boiling, pouring, etc.)

**Be specific about the sound source:**
- ✅ GOOD: `knife_chopping` (knife on cutting board)
- ❌ BAD: `kitchen_sounds` (could be anything)

**Keep classes distinct:**
- ✅ GOOD: `frying_food` (sizzling) vs `boiling_water` (bubbling) - acoustically different
- ❌ BAD: `water_running_fast` vs `water_running_slow` - hard to distinguish

#### Step 9: Create Target Class List

**Example list:**
```
1. chopping (knife on cutting board)
2. boiling_water (pot on stove, vigorous bubbles)
3. frying_food (food in hot oil, sizzling)
4. running_faucet (water tap, steady flow)
5. electric_fan (oscillating or stationary)
```

**Decision:** Aim for 3-10 custom classes to start (more can be added later)

---

### Phase 3: Recording Your First Class

#### Step 10: Run Recording Script

```bash
python record_samples.py chopping
```

**Expected output:**
```
Recording 5-second clips for class: chopping
Clips will be saved to: data/custom/chopping/
Press Enter to start recording (Ctrl+C to stop)...
```

#### Step 11: Recording Best Practices - Setup

**Before pressing Enter:**

1. **Position microphone:** 30-100cm from sound source
   - Too close: Clipping, distortion
   - Too far: Low volume, too much room noise

2. **Minimize background noise:**
   - Close windows (traffic, wind)
   - Turn off fans, AC (if recording fan/AC sounds, ignore this)
   - Quiet room (no TV, music, conversations)
   - Turn off phone notifications

3. **Prepare sound source:**
   - For chopping: Knife + cutting board + vegetable/fruit
   - Have everything ready before pressing Enter

4. **Test levels:**
   - Open System Preferences → Sound → Input (macOS)
   - Or Settings → Sound → Input (Windows)
   - Speak/make sound, watch input level
   - Should be in green zone (not red = clipping)

#### Step 12: Record First Clip

1. **Press Enter**
2. **Wait for:** "Recording... (5 seconds)"
3. **Make the sound** continuously for 5 seconds:
   - For chopping: Chop continuously (not just one chop)
   - For boiling: Let it bubble naturally
   - For faucet: Let water run steadily
4. **Wait for:** "Saved to: data/custom/chopping/chopping_20260308_143522.wav"

#### Step 13: Record Variations (CRITICAL for Accuracy!)

**Record 20+ clips with variations:**

**Variation 1: Angle** (5 clips)
- Front of cutting board
- Side of cutting board
- Top view (mic above)
- 45-degree angle left
- 45-degree angle right

**Variation 2: Intensity** (5 clips)
- Soft chopping (gentle cuts)
- Medium chopping (normal)
- Hard chopping (forceful cuts)
- Fast chopping (rapid cuts)
- Slow chopping (deliberate cuts)

**Variation 3: Material Being Chopped** (5 clips)
- Carrot (hard vegetable)
- Onion (medium)
- Lettuce (soft)
- Apple (crunchy)
- Meat (soft)

**Variation 4: Environmental** (5 clips)
- Silence in background
- Slight background noise (normal room)
- Different surface (different cutting board material)
- Different knife (if available)
- Different time of day

**Total: 20 clips minimum, 50-100 recommended**

#### Step 14: Quality Validation

After recording each clip, **listen to it:**

```bash
# macOS:
afplay data/custom/chopping/chopping_20260308_143522.wav

# Linux:
aplay data/custom/chopping/chopping_20260308_143522.wav

# Windows:
# Use Windows Media Player or VLC
```

**Check for:**
- ✅ **Full 5 seconds** (not cut off)
- ✅ **Clear sound** (not muffled)
- ✅ **No clipping** (no distortion/crackling)
- ✅ **Target sound dominant** (not drowned by background noise)
- ✅ **Consistent volume** (not too quiet)

**If bad quality:** Press Enter again to record a replacement clip

#### Step 15: Verify File Count

```bash
ls data/custom/chopping/ | wc -l
```

**Expected:** 20 or more

**If fewer than 20:** Record more clips until you reach 20

---

### Phase 4: Recording Additional Classes

#### Step 16: Record Second Class

```bash
python record_samples.py boiling_water
```

**Repeat Steps 10-15** for this class:
- 20+ clips
- Variations in angle, intensity, etc.
- Quality validation

#### Step 17: Record Third Class

```bash
python record_samples.py frying_food
```

**Repeat Steps 10-15** again.

#### Step 18: Record Remaining Classes

Continue for each class in your target list:

```bash
python record_samples.py running_faucet
python record_samples.py electric_fan
```

#### Step 19: Maintain Consistency Across Classes

**Same microphone:** Use the same recording device for all classes
**Same position:** Keep mic roughly same distance from source
**Same environment:** Record in the same room if possible
**Same settings:** Don't change system volume or input gain between classes

#### Step 20: Avoid Class Overlap

**Bad practice:** Recording running faucet sounds in the `boiling_water` class
**Good practice:** Each clip contains ONLY the target sound (as much as possible)

**Class boundaries:**
- `boiling_water`: Bubbling, steam, pot rattling
- `running_faucet`: Steady water flow, no bubbling
- `pouring_water`: Water hitting container, transient sound

Keep these distinct!

---

### Phase 5: Dataset Validation

#### Step 21: Count Files Per Class

```bash
for class in data/custom/*/; do
    echo "$(basename "$class"): $(ls "$class"/*.wav 2>/dev/null | wc -l) clips"
done
```

**Expected output:**
```
chopping: 25 clips
boiling_water: 30 clips
frying_food: 22 clips
running_faucet: 28 clips
electric_fan: 20 clips
```

**Requirement:** Each class must have ≥20 clips

#### Step 22: Verify All Files Are WAV Format

```bash
find data/custom/ -type f ! -name "*.wav"
```

**Expected:** No output (empty)
**If files shown:** These are non-WAV files, remove or convert them

**Convert MP3/M4A to WAV (if needed):**
```bash
# Install ffmpeg first: brew install ffmpeg (macOS)
ffmpeg -i input.mp3 -ar 44100 -ac 1 output.wav
```

#### Step 23: Check File Sizes

```bash
find data/custom/ -name "*.wav" -exec ls -lh {} \; | awk '{print $5, $9}'
```

**Expected size:** ~850 KB per file (for 5-second, 44.1kHz, 16-bit mono WAV)

**If significantly different:**
- Much smaller (< 100 KB): Likely wrong sample rate or duration
- Much larger (> 5 MB): Likely stereo or higher bit depth
- **Re-record** with the script to ensure correct format

#### Step 24: Listen to Random Samples

```bash
# macOS:
afplay "$(find data/custom/ -name "*.wav" | shuf -n 1)"

# Linux:
aplay "$(find data/custom/ -name "*.wav" | shuf -n 1)"
```

**Run this 5-10 times** to spot-check quality across all classes.

**Check for:**
- Sound quality is consistent
- No silent/near-silent clips
- No clipped/distorted audio
- Correct sound class (file in right directory)

#### Step 25: Test Loading in Python

Create a test script `test_custom_data.py`:

```python
import os
import librosa

custom_dir = "data/custom"
classes = os.listdir(custom_dir)

print(f"Found {len(classes)} custom classes:")

for class_name in classes:
    class_path = os.path.join(custom_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
    print(f"  {class_name}: {len(wav_files)} clips")

    # Test loading first file
    if wav_files:
        test_file = os.path.join(class_path, wav_files[0])
        try:
            audio, sr = librosa.load(test_file, sr=44100, mono=True)
            duration = len(audio) / sr
            print(f"    ✓ Loaded successfully (duration: {duration:.2f}s, sr: {sr}Hz)")
        except Exception as e:
            print(f"    ✗ Error loading: {e}")

print("\nDataset validation complete!")
```

Run it:
```bash
python test_custom_data.py
```

**Expected output:**
```
Found 5 custom classes:
  chopping: 25 clips
    ✓ Loaded successfully (duration: 5.00s, sr: 44100Hz)
  boiling_water: 30 clips
    ✓ Loaded successfully (duration: 5.00s, sr: 44100Hz)
  ...
Dataset validation complete!
```

---

### Phase 6: Upload to Google Colab

#### Step 26: Create ZIP Archive

```bash
cd data
zip -r custom.zip custom/
cd ..
```

**Verify ZIP created:**
```bash
ls -lh data/custom.zip
```

Expected size: 20-100 MB (depends on number of clips)

#### Step 27: Upload to Colab

1. Open your training notebook in Google Colab
2. Click **📁 Files** icon in left sidebar
3. Click **📤 Upload** icon
4. Select `data/custom.zip`
5. Wait for upload to complete (2-10 minutes depending on size)

#### Step 28: Unzip in Colab

In the **first code cell** of your notebook (before any imports), add:

```python
# Unzip custom dataset
!unzip -q custom.zip -d /content/
!mkdir -p /content/data
!mv /content/custom /content/data/
```

Run this cell (Shift+Enter).

#### Step 29: Verify Upload

Add a verification cell:

```python
import os

custom_dir = "/content/data/custom"
if os.path.exists(custom_dir):
    classes = [d for d in os.listdir(custom_dir) if os.path.isdir(os.path.join(custom_dir, d))]
    print(f"✓ Custom dataset found: {len(classes)} classes")
    for class_name in classes:
        class_path = os.path.join(custom_dir, class_name)
        wav_count = len([f for f in os.listdir(class_path) if f.endswith('.wav')])
        print(f"  - {class_name}: {wav_count} clips")
else:
    print("✗ Custom dataset not found - check upload")
```

**Expected output:**
```
✓ Custom dataset found: 5 classes
  - chopping: 25 clips
  - boiling_water: 30 clips
  - frying_food: 22 clips
  - running_faucet: 28 clips
  - electric_fan: 20 clips
```

#### Step 30: Proceed to Training

Now run the rest of your training notebook. The custom classes will be automatically merged with ESC-50.

---

## Recording Best Practices

### Audio Quality Requirements

| Parameter | Value | Why |
|-----------|-------|-----|
| **Sample Rate** | 44,100 Hz | CD-quality, industry standard for audio ML |
| **Bit Depth** | 16-bit | Good dynamic range without excessive file size |
| **Channels** | Mono (1 channel) | Reduces complexity, faster processing |
| **Duration** | Exactly 5 seconds | Matches ESC-50 format, consistent feature extraction |
| **Format** | WAV (uncompressed) | No lossy compression artifacts (unlike MP3) |

### Microphone Settings

#### Recommended Setup
- **Built-in laptop mic:** Adequate for most sounds
- **USB condenser mic:** Better quality, more consistent
- **Headset mic:** Avoid (too close to mouth, not omnidirectional)
- **Phone mic:** Avoid (different characteristics, hard to keep consistent)

#### System Settings (macOS)
1. Open **System Preferences → Sound → Input**
2. Select your microphone
3. **Input volume:** 50-70% (adjust based on sound source)
4. **Use ambient noise reduction:** OFF (can distort target sounds)

#### System Settings (Windows)
1. Open **Settings → System → Sound → Input**
2. Select your microphone
3. **Input volume:** 50-70%
4. **Microphone enhancements:** OFF

#### System Settings (Linux)
```bash
# List devices
arecord -l

# Set input volume (50%)
amixer sset 'Capture' 50%
```

### Recording Environment

#### Ideal Recording Space
- ✅ Small to medium room (less reverb than large spaces)
- ✅ Carpeted or furnished (absorbs echoes)
- ✅ Windows closed (blocks traffic, wind)
- ✅ Quiet time of day (early morning, late evening)
- ✅ Consistent background noise level (same AC/fan state)

#### What to Avoid
- ❌ Large empty rooms (too much echo)
- ❌ Bathrooms (extreme reverb)
- ❌ Near windows facing busy streets
- ❌ During construction hours
- ❌ Changing background conditions between clips

### Sound Variation Strategies

#### Angle Variation (Critical!)
Record from **multiple positions** around the sound source:
- Front (0°)
- Left side (90°)
- Right side (270°)
- Back (180°)
- Top (mic above source)

**Why:** Real-world monitoring won't always be from the same angle

#### Intensity Variation
- **Soft:** Minimal intensity (gentle chopping, low faucet flow)
- **Medium:** Normal usage
- **Loud:** Maximum intensity (hard chopping, full faucet flow)

**Why:** Same sound can vary in volume in real scenarios

#### Duration Pattern Variation
- **Continuous:** Sound throughout entire 5 seconds
- **Intermittent:** Sound starts/stops within clip
- **Crescendo:** Sound builds up over time
- **Decrescendo:** Sound fades out

**Why:** Teaches model temporal patterns

#### Instance Variation (if applicable)
- **Different materials:** Chop carrots, onions, meat, etc.
- **Different speeds:** Fast frying vs slow simmering
- **Different flows:** Trickle vs full blast faucet

**Why:** Captures intra-class diversity

### What to Avoid

#### Don't Record
- ❌ **Silence or near-silence** (< -40 dB): Teaches model nothing
- ❌ **Multiple overlapping sounds**: Keep it to ONE sound class per clip
- ❌ **Clipped audio**: Red levels = distortion = bad training data
- ❌ **Compressed formats**: MP3/AAC introduce artifacts
- ❌ **Inconsistent recording devices**: Don't switch between mics

#### Don't Mix Classes
- ❌ Recording faucet in `boiling_water` class
- ❌ Recording speech in `chopping` class
- ❌ Recording door slam in `footsteps` class

**Keep each class pure!**

#### Don't Skimp on Quantity
- ❌ 5 clips per class: Model will overfit, won't generalize
- ✅ 20 clips per class: Minimum for reasonable performance
- ✅✅ 50-100 clips per class: Much better accuracy and robustness

---

## Troubleshooting

### Common Errors and Solutions

#### Error: `FileNotFoundError: data/custom`

**Cause:** Directory doesn't exist
**Solution:**
```bash
mkdir -p data/custom
```

#### Error: `No module named 'sounddevice'`

**Cause:** Missing dependency
**Solution:**
```bash
pip install sounddevice
```

#### Error: `PortAudio error: invalid number of channels`

**Cause:** Microphone not available or wrong channel count
**Solution:**
1. Check available devices:
   ```bash
   python -c "import sounddevice as sd; print(sd.query_devices())"
   ```
2. Find your input device index (e.g., index 2)
3. Edit `record_samples.py` and set `device=2` in `sd.InputStream()`

#### Error: `Permission denied: /dev/snd` (Linux)

**Cause:** User not in audio group
**Solution:**
```bash
sudo usermod -a -G audio $USER
# Log out and back in
```

#### Issue: Files are wrong size (not ~850KB)

**Possible causes:**
1. **Wrong sample rate:** Should be 44,100 Hz
2. **Wrong duration:** Should be exactly 5 seconds
3. **Stereo instead of mono:** Should be 1 channel
4. **Wrong bit depth:** Should be 16-bit

**Solution:** Re-record using `record_samples.py` (it sets correct parameters)

#### Issue: Training accuracy doesn't improve with custom data

**Possible causes:**
1. **Not enough clips:** Need 50-100 per class for best results
2. **Poor variation:** All clips sound identical (too consistent)
3. **Class overlap:** Sounds bleeding between classes
4. **Low quality:** Clipping, distortion, too much noise

**Solutions:**
- Record more clips (aim for 50-100 per class)
- Increase variation (angles, intensities, instances)
- Ensure class boundaries are clear
- Re-record poor quality clips

#### Issue: Model predicts wrong class for custom sounds

**Possible causes:**
1. **Sounds too similar to ESC-50 class:** e.g., `running_faucet` confused with `water_drops`
2. **Insufficient training data:** Only 20 clips not enough
3. **Inconsistent recordings:** Different mic positions, noise levels

**Solutions:**
- Make your custom class more specific (e.g., `kitchen_faucet_running`)
- Record 50+ clips per class
- Re-record with consistent setup

#### Issue: `labels.json` doesn't include custom classes

**Cause:** Custom data not loaded during training
**Solution:**
1. Verify `data/custom/` exists in Colab: `!ls /content/data/custom`
2. Check notebook cell that loads custom data ran successfully
3. Re-run training notebook from scratch

---

## Integration with Training Pipeline

### How Custom Data is Merged with ESC-50

The training notebook automatically:

1. **Loads ESC-50 first:**
   - 50 classes (dog, cat, water_drops, etc.)
   - 40 clips per class from training folds (1600 total)
   - 10 clips per class from validation fold (400 total)
   - 10 clips per class from test fold (400 total)

2. **Loads custom data second:**
   - N custom classes (e.g., 5 classes)
   - Variable clips per class (20-100 each)
   - Split 70% train / 15% val / 15% test

3. **Merges datasets:**
   - Combined class count: 50 + N (e.g., 55 classes)
   - Combined training samples: 1600 + (custom train)
   - Labels updated: `[...ESC-50 labels..., "chopping", "frying_food", ...]`

4. **Updates model:**
   - Output layer size: `num_classes = 50 + N`
   - Saves updated `labels.json` with all classes

### Which Notebook Cells Handle Custom Data

#### Cell 3: Load Custom Data
```python
# Load custom classes from data/custom/
custom_dir = "/content/data/custom"
if os.path.exists(custom_dir):
    for class_name in os.listdir(custom_dir):
        class_path = os.path.join(custom_dir, class_name)
        # ... load WAV files ...
```

#### Cell 5: Merge with ESC-50
```python
# Combine ESC-50 + custom
all_classes = esc50_classes + custom_classes  # e.g., 50 + 5 = 55
num_classes = len(all_classes)
```

#### Cell 10: Save Updated labels.json
```python
labels = {
    "num_classes": 55,
    "labels": ["dog", "cat", ..., "chopping", "frying_food", ...]
}
json.dump(labels, open("labels.json", "w"))
```

### Verification After Training

#### 1. Check labels.json

```bash
cat models/trained_models/labels.json
```

**Expected:**
```json
{
  "num_classes": 55,
  "labels": [
    "dog", "cat", "pig", ..., "chopping", "boiling_water", "frying_food", ...
  ]
}
```

**Verify:**
- `num_classes` = 50 + your custom count
- Last items in `labels` array are your custom class names

#### 2. Check Model Output Shape

In a Python shell:
```python
import torch
import json

# Load model
model = torch.load("models/trained_models/model.pt", map_location="cpu")

# Check output layer
print(model.classifier[-1].out_features)  # Should equal num_classes
```

**Expected:** 55 (or 50 + your custom count)

#### 3. Test Inference with Custom Classes

```bash
python classifier.py --model-dir models/trained_models
```

**Then make one of your custom sounds** (e.g., start chopping)

**Expected output:**
```
Top predictions:
  chopping: 78.3%
  can_opening: 12.1%
  mouse_click: 5.2%
```

Your custom class should appear in top predictions!

#### 4. Test Monitor with Custom Context Rules

If you added context rules for your custom classes (in `monitor.py`):

```bash
python monitor.py --model-dir models/trained_models
```

**Make your custom sound** and verify:
- ✅ Class appears in terminal display
- ✅ Voice alert triggers (if rule defined)
- ✅ Duration tracking works

---

## Summary Checklist

Before training, verify:

- [ ] All class directories use `lowercase_with_underscores` naming
- [ ] Each class has **≥20 clips** (ideally 50-100)
- [ ] All files are `.wav` format (not MP3, M4A, etc.)
- [ ] Files are ~850 KB each (5 seconds, 44.1kHz, mono, 16-bit)
- [ ] Recordings include variation (angles, intensities, instances)
- [ ] No silent/near-silent clips (< -40 dB)
- [ ] No clipped/distorted audio
- [ ] Classes are distinct (no overlap between similar sounds)
- [ ] Recorded with same microphone throughout
- [ ] Directory structure is `data/custom/<class_name>/*.wav`
- [ ] ZIP archive created: `data/custom.zip`
- [ ] ZIP uploaded to Google Colab successfully
- [ ] Verification cell confirms all classes loaded

After training, verify:

- [ ] `labels.json` includes all custom classes
- [ ] `num_classes` = 50 + custom count
- [ ] Model output shape matches `num_classes`
- [ ] `python classifier.py` shows custom classes in predictions
- [ ] `python monitor.py` detects custom sounds correctly

---

**You're ready to train! Open `training/train_classifier.ipynb` in Google Colab and run all cells.**

Good luck! 🎯
