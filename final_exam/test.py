import requests
import json

# Path to the image file
image_path = "test.jpg"

# Open the image file in binary mode
with open(image_path, "rb") as image_file:
    # Send the POST request with the file
    response = requests.post(
        "https://openai.chinococo.tw/analyze-image",
        files={"file": image_file}  # Key "file" is commonly expected by servers
    )

# Decode the server response
response_content = response.content.decode('utf-8')  # Decode the raw bytes
try:
    # Parse JSON response
    response_json = json.loads(response_content)
    # Extract and print the "result" field
    result_text = response_json.get("result", "No result found")
    print("Response status code:", response.status_code)
    print("Decoded response content:", result_text)
except json.JSONDecodeError:
    print("Failed to parse JSON response")
    print("Raw response content:", response_content)

