import base64
import json

signs = ["Maximum Speed Limit",
         "Minimum Speed Limit",
         "Pedestrian Crossing",
         "Railroad Crossing Warning",
         "Right Turn",
         "Stop",
         "Stop and Proceed"]

# List to hold all the JSON data for fine-tuning
training_data = []

# Open the annotations file and process each line
with open("train/_annotations.txt") as f:
    for index, line in enumerate(f):
        # Stop after processing the first 30 lines
        if index >= 60:
            break

        line_data = line.split()
        dir = f"train/{line_data[0]}"

        # Open the image file in binary mode and encode it in base64
        with open(dir, "rb") as image_file:
            data = base64.b64encode(image_file.read()).decode("utf-8")

        # Get the class labels for each object in the image
        _class = []
        __index = []
        for i in range(1, len(line_data)):
            size = line_data[i].split(",")
            __index.append([signs[int(line_data[i].split(",")[-1])],int(size[2])-int(size[0])])
        __index = sorted(__index,key=lambda x:x[1],reverse=True)
        _class = [data[0] for data in __index]
        print(__index)
        print(_class)

        # Prepare the format as per the example you want
        entry = {
            "messages": [
                {
                    "role": "system",
                    "content": "分析圖片上面有甚麼樣的路牌標示，並回傳一個list給我，照寬度大小順序回傳"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{data}",
                            }
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": str(_class)
                }
            ]
        }

        # Append each formatted entry to the list
        training_data.append(entry)

# Write the training data to a JSONL file in the requested format
with open("training_chat.jsonl", "w", encoding="utf-8") as json_file:
    for entry in training_data:
        json.dump(entry, json_file, ensure_ascii=False)
        json_file.write("\n")

print("Training data JSONL created successfully.")
