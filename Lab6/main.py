from urllib.request import OpenerDirector

import requests
from flask import Flask, request, jsonify, render_template
import openai
import os

from ipykernel.jsonutil import encode_images
from openai import OpenAI
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import base64

def encode_image(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
# Load environment variables from .env file
load_dotenv()

# Get sensitive values from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)
# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
client = OpenAI(
    api_key = OPENAI_API_KEY
)
# Function to analyze image (using OpenAI Image API or other logic)
def analyze_image_with_openai(file_path):
    try:
        print(file_path)
        base64_image = encode_image(file_path)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """分析圖片上面有甚麼樣的路牌標示"""},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        ans = response.choices[0]
        ans_msg = ans.message.content
        return ans_msg
    except Exception as e:
        return f"Image Analysis Error: {e}"

# Function to analyze text using OpenAI
def analyze_text_with_openai(input_text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "你是一個專家，將指令拆成一個動作 一個動作 弄成list，以下格式為一組 格式為 {left:{\"speed\": \"0.0~1.0\", \"time\": \"0~10s\"}, right:{\"speed\": \"0.0~1.0\", \"time\": \"0~10s\"}}。"},
                {"role": "user", "content": input_text}
            ],
            temperature=0.7,
            max_tokens=300,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
# Define the home route
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/image")
def image():
    return render_template("image.html")
# Define API endpoint for text analysis
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Please provide the 'text' field."}), 400

    input_text = data["text"]
    analysis_result = analyze_text_with_openai(input_text)
    return jsonify({"analysis": analysis_result})
# Define image analysis route
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Empty file name."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Perform image analysis
        analysis_result = analyze_image_with_openai(file_path)
        return jsonify({"result": analysis_result})

    return jsonify({"error": "Invalid file type."}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
