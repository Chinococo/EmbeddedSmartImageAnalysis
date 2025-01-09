import requests
import json
import time  # 引入time模組來計算請求的時間

# Path to the image file
image_path = "test.jpg"

# 計算請求的開始時間
start_time = time.time()

# Open the image file in binary mode
with open(image_path, "rb") as image_file:
    # Send the POST request with the file
    response = requests.post(
        "http://0.0.0.0:5000/analyze-image",
        files={"file": image_file}  # Key "file" is commonly expected by servers
    )

# 計算請求結束時間
end_time = time.time()

# 計算請求的耗時（秒）
request_duration = end_time - start_time

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

# 打印請求的時間
print(f"Request duration: {request_duration:.4f} seconds")
