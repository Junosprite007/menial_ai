from flask import Flask, request, jsonify, send_from_directory
import os
import librosa
import numpy as np

app = Flask(__name__)

# Basic configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return send_from_directory('.', 'water_test.html')

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio.wav')
    file.save(filepath)
    
    try:
        # Load audio file trying different libraries if needed
        try:
            y, sr = librosa.load(filepath, duration=3.0)  # Analyze first 3 seconds
        except Exception as load_err:
             print(f"Librosa load error: {load_err}")
             return jsonify({'error': f"Could not load audio file: {str(load_err)}"}), 400

        # Extract features
        # 1. Zero Crossing Rate (Water sounds have high ZCR due to noise nature)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # 2. Spectral Centroid (Water sounds often have high frequency content)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # 3. Spectral Rolloff (Measure of the shape of the signal. High for water.)
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
        
        # Heuristic for water detection (This is a simplified example!)
        # Adjust these thresholds based on your specific "water" sound (tap, rain, splash)
        # Typical water sounds: High ZCR (> 0.05), High Centroid (> 1500Hz), High Rolloff
        
        is_water = False
        confidence = 0.0
        details = []

        # Check ZCR
        if zcr > 0.03:
            details.append("High Zero Crossing Rate (Noise-like)")
            confidence += 0.3
        
        # Check Spectral Centroid
        if spectral_centroid > 2500: # Hz
            details.append("High Frequency Content (Centroid > 2500Hz)")
            confidence += 0.4
        elif spectral_centroid > 1500:
            confidence += 0.2
            
        # Check Spectral Rolloff
        if spectral_rolloff > 4000: # Hz
            details.append("Significant High Frequency Energy (Rolloff > 4000Hz)")
            confidence += 0.3

        if confidence > 0.6:
            result = "Water Detected 💧"
            is_water = True
        else:
            result = "Not Water / Unsure"
            is_water = False

        return jsonify({
            'result': result,
            'is_water': is_water,
            'confidence': round(confidence * 100, 2),
            'features': {
                'zcr': float(zcr),
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff)
            },
            'details': details
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    print("Starting Water Detection Server...")
    print("Go to http://localhost:5001 to test")
    app.run(debug=True, port=5001)
