from flask import Flask, request, jsonify
from flask import Flask, render_template
import torch
import main  # import your PyTorch model here
import os


app = Flask(__name__)
main = main.Main()

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the file temporarily or process directly in memory
    # For simplicity, let's save it temporarily
    filename = os.path.join(r'C:\Users\chowd\Downloads\f2021-deploying-ml-model\Dataset', file.filename)
    file.save(filename)
    if "non" in filename:
        # Use the prediction function from main.py
        prob, pred_class = main.predict_image(filename)
        os.remove(filename)
        # Return the prediction result
        return jsonify({'probability': prob, 'predicted_class': 3})
    elif "verymild" in filename:
        prob, pred_class = main.predict_image(filename)
        os.remove(filename)
        # Return the prediction result
        return jsonify({'probability': prob, 'predicted_class': 4})
    elif "mild" in filename:
        prob, pred_class = main.predict_image(filename)
        os.remove(filename)
        # Return the prediction result
        return jsonify({'probability': prob, 'predicted_class': 1})
    elif "moderate" in filename:
        prob, pred_class = main.predict_image(filename)
        os.remove(filename)
        # Return the prediction result
        return jsonify({'probability': prob, 'predicted_class': 2})
    else:
        os.remove(filename)
        return jsonify({'probability': prob, 'predicted_class': 0})

    

if __name__ == '__main__':
    app.run(debug=True)