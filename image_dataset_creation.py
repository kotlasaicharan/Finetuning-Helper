
import os
import io
from PIL import Image
from datasets import Dataset, DatasetDict


data = []
for i in range(len(df)):
    image_path = "/content/drive/MyDrive/flux_gen/" +  f"{i}.png"
    image_filename = "/content/drive/MyDrive/flux_gen" + f"{i}.png"

    try:
        image = Image.open(os.path.join(image_path, image_filename))
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format)
        image = Image.open(img_byte_arr)

        conversations = [
            {
                "content": [
                    {'index': None, "type": "text", "text": "question"},
                    {'index': 0, "type": "image"},
                ],
                "role": "user",
            },
            {
                "role": "assistant",
                "content": [
                    {'index': None, "type": "text", "text": str(df["Poem"][i])},
                ],
            },
        ]
        
        data.append({
            'messages': conversations,
            'images': [image]
        })
    except FileNotFoundError:
        print(f"Warning: Image file not found: {os.path.join(image_path, image_filename)}")
        continue # Skip to the next iteration if the image is not found

dataset = Dataset.from_list(data)
DatasetDict({
    'train': dataset
}).push_to_hub("Cherran/sample") # Replace with your desired dataset name
