from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('model/sgd_classifier_model.pkl')

@app.route('/', methods=['GET'])
def home():
    return "✅ Hello from Azure!", 200


@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    data = request.get_json()

    if not data or 'errors' not in data:
        return jsonify({'error': "Please provide 'errors' key with a list of error messages."}), 400

    errors = data['errors']

    if not isinstance(errors, list) or not all(isinstance(e, str) for e in errors):
        return jsonify({'error': "'errors' must be a list of strings."}), 400

    try:
        # Get predictions
        suggestions = model.predict(errors)
        
        # Pair each error with its suggestion
        results = [{"error": e, "suggestion": s} for e, s in zip(errors, suggestions)]

        return jsonify({"results": results})
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)

application = app
