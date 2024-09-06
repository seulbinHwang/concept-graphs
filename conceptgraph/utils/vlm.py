import json
from openai import OpenAI
import os
import base64

from PIL import Image
import numpy as np

import ast
import re

system_prompt_1 = '''
You are an agent specialized in describing the spatial relationships between objects in an annotated image.

You will be provided with an annotated image and a list of labels for the annotations. Your task is to determine the spatial relationships between the annotated objects in the image, and return a list of these relationships in the correct list of tuples format as follows:
[("object1", "spatial relationship", "object2"), ("object3", "spatial relationship", "object4"), ...]

Your options for the spatial relationship are "on top of" and "next to".

For example, you may get an annotated image and a list such as 
["cup 3", "book 4", "clock 5", "table 2", "candle 7", "music stand 6", "lamp 8"]

Your response should be a description of the spatial relationships between the objects in the image. 
An example to illustrate the response format:
[("book 4", "on top of", "table 2"), ("cup 3", "next to", "book 4"), ("lamp 8", "on top of", "music stand 6")]
'''
'''
You are an agent specialized in identifying and describing objects that are placed "on top of" each other in an annotated image. You always output a list of tuples that describe the "on top of" spatial relationships between the objects, and nothing else. When in doubt, output an empty list.

When provided with an annotated image and a corresponding list of labels for the annotations, your primary task is to determine and return the "on top of" spatial relationships between the annotated objects. Your responses should be formatted as a list of tuples, specifically highlighting objects that rest on top of others, as follows:
[("object1", "on top of", "object2"), ...]
'''

# Only deal with the "on top of" relation
system_prompt_only_top = '''
You are an agent specializing in identifying the physical and spatial relationships in annotated images for 3D mapping.

In the images, each object is annotated with a bright numeric id (i.e. a number) and a corresponding colored contour outline. Your task is to analyze the images and output a list of tuples describing the physical relationships between objects. Format your response as follows: [("1", "relation type", "2"), ...]. When uncertain, return an empty list.

Note that you are describing the **physical relationships** between the **objects inside** the image.

You will also be given a text list of the numeric ids of the objects in the image. The list will be in the format: ["1: name1", "2: name2", "3: name3" ...], only output the physical relationships between the objects in the list.

The relation types you must report are:
- phyically placed on top of: ("object x", "on top of", "object y") 
- phyically placed underneath: ("object x", "under", "object y") 

An illustrative example of the expected response format might look like this:
[("object 1", "on top of", "object 2"), ("object 3", "under", "object 2"), ("object 4", "on top of", "object 3")]. Do not put the names of the objects in your response, only the numeric ids.

Do not include any other information in your response. Only output a parsable list of tuples describing the given physical relationships between objects in the image.
'''

# For captions
system_prompt_captions = '''
You are an agent specializing in accurate captioning objects in an image.

In the images, each object is annotated with a bright numeric id (i.e. a number) and a corresponding colored contour outline. Your task is to analyze the images and output in a structured format, the captions for the objects.

You will also be given a text list of the numeric ids and names of the objects in the image. The list will be in the format: ["1: name1", "2: name2", "3: name3" ...]

The names were obtained from a simple object detection system and may be inaacurate.

Your response should be in the format of a list of dictionaries, where each dictionary contains the id, name, and caption of an object. Your response will be evaluated as a python list of dictionaries, so make sure to format it correctly. An example of the expected response format is as follows:
[
    {"id": "1", "name": "object1", "caption": "concise description of the object1 in the image"},
    {"id": "2", "name": "object2", "caption": "concise description of the object2 in the image"},
    {"id": "3", "name": "object3", "caption": "concise description of the object3 in the image"}
    ...
]

And each caption must be a concise description of the object in the image.
'''

system_prompt_consolidate_captions = '''
You are an agent specializing in consolidating multiple captions for the same object into a single, clear, and accurate caption.

You will be provided with several captions describing the same object. Your task is to analyze these captions, identify the common elements, remove any noise or outliers, and consolidate them into a single, coherent caption that accurately describes the object.

Ensure the consolidated caption is clear, concise, and captures the essential details from the provided captions.

Here is an example of the input format:
[
    {"id": "3", "name": "cigar box", "caption": "rectangular cigar box on the side cabinet"},
    {"id": "9", "name": "cigar box", "caption": "A small cigar box placed on the side cabinet."},
    {"id": "7", "name": "cigar box", "caption": "A small cigar box is on the side cabinet."},
    {"id": "8", "name": "cigar box", "caption": "Box on top of the dresser"},
    {"id": "5", "name": "cigar box", "caption": "A cigar box placed on the dresser next to the coffeepot."},
]

Your response should be a JSON object with the format:
{
    "consolidated_caption": "A small rectangular cigar box on the side cabinet."
}

Do not include any additional information in your response.
'''

system_prompt = system_prompt_only_top

# gpt_model = "gpt-4-vision-preview"
gpt_model = "gpt-4o-2024-05-13"


def get_openai_client():
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    return client


# Function to encode the image as base64
def encode_image_for_openai(image_path: str,
                            resize=False,
                            target_size: int = 512):
    print(f"Checking if image exists at path: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not resize:
        # Open the image
        print(f"Opening image from path: {image_path}")
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            print("Image encoded in base64 format.")
        return encoded_image

    print(f"Opening image from path: {image_path}")
    with Image.open(image_path) as img:
        # Determine scaling factor to maintain aspect ratio
        original_width, original_height = img.size
        print(
            f"Original image dimensions: {original_width} x {original_height}")

        if original_width > original_height:
            scale = target_size / original_width
            new_width = target_size
            new_height = int(original_height * scale)
        else:
            scale = target_size / original_height
            new_height = target_size
            new_width = int(original_width * scale)

        print(f"Resized image dimensions: {new_width} x {new_height}")

        # Resizing the image
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        print("Image resized successfully.")

        # Convert the image to bytes and encode it in base64
        with open("temp_resized_image.jpg", "wb") as temp_file:
            img_resized.save(temp_file, format="JPEG")
            print("Resized image saved temporarily for encoding.")

        # Open the temporarily saved image for base64 encoding
        with open("temp_resized_image.jpg", "rb") as temp_file:
            encoded_image = base64.b64encode(temp_file.read()).decode('utf-8')
            print("Image encoded in base64 format.")

        # Clean up the temporary file
        os.remove("temp_resized_image.jpg")
        print("Temporary file removed.")

    return encoded_image


def consolidate_captions(client: OpenAI, captions: list):
    """

[
    {"id": "1", "name": "object1", "caption": "concise description of object1"},
    {"id": "2", "name": "object2", "caption": "concise description of object2"},
    ...
]

### 1. **주요 역할**
- **여러 캡션을 입력**으로 받아, 이들 캡션을 분석하여 **중복 요소를 제거**하고 **하나의 통합된 캡션**을 생성
- GPT 모델을 사용하여 캡션을 자동으로 통합합니다.

### 2. **세부 알고리즘 로직**

1. **캡션 포맷팅**:
   - 입력으로 주어진 **여러 개의 캡션** 리스트에서 각각의 **캡션 텍스트**를 추출해 하나의 문자열로 결합
    - 각 캡션은 줄바꿈(`\n`)을 통해 구분
   - 이때 **None 값**을 가진 캡션은 무시

2. **시스템 프롬프트 설정**:
   - **시스템 프롬프트**는 모델에게 작업의 목적과 방법을 설명하는 역할
   이 프롬프트는 모델이 각 캡션의 공통 요소를 식별하고,
        가장 중요한 정보를 추출해 통합된 설명을 생성하도록 유도

3. **GPT 모델 호출**:
   - OpenAI의 GPT 모델에 **메시지**를 전달하여 통합된 캡션을 요청
   - 모델은 주어진 캡션을 분석하고, 최종적으로 **하나의 통합된 캡션**을 생성하여 JSON 형식으로 반환

4. **결과 처리**:
   - 모델로부터 반환된 결과(JSON 형식)를 **파싱**하여 **통합된 캡션**을 추출
   - 캡션을 추출하는 과정에서 **예외 처리**를 통해 오류 발생 시 빈 문자열을 반환

### 3. **결론**
이 함수는 여러 개의 설명(캡션)을 하나의 명확하고 일관된 설명으로 통합하는 기능을 제공하며, GPT 모델을 통해 자동으로 이 작업을 수행합니다. 이를 통해 중복되거나 불필요한 정보를 제거하고, **객체를 가장 잘 설명하는 최종 문구**를 생성할 수 있습니다.
    """
    # Formatting the captions into a single string prompt
    captions_text = "\n".join(
        [f"{cap['caption']}" for cap in captions if cap['caption'] is not None])
    user_query = f"Here are several captions for the same object:\n{captions_text}\n\nPlease consolidate these into a single, clear caption that accurately describes the object."

    messages = [{
        "role": "system",
        "content": system_prompt_consolidate_captions
    }, {
        "role": "user",
        "content": user_query
    }]

    consolidated_caption = ""
    try:
        response = client.chat.completions.create(
            model=f"{gpt_model}",
            messages=messages,
            response_format={"type": "json_object"})

        consolidated_caption_json = response.choices[0].message.content.strip()
        consolidated_caption = json.loads(consolidated_caption_json).get(
            "consolidated_caption", "")
        print(f"Consolidated Caption: {consolidated_caption}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        consolidated_caption = ""

    return consolidated_caption


def extract_list_of_tuples(text: str):
    # Pattern to match a list of tuples, considering a list that starts with '[' and ends with ']'
    # and contains any characters in between, including nested lists/tuples.
    text = text.replace('\n', ' ')
    pattern = r'\[.*?\]'

    # Search for the pattern in the text
    match = re.search(pattern, text)
    if match:
        # Extract the matched string
        list_str = match.group(0)
        try:
            # Convert the string to a list of tuples
            result = ast.literal_eval(list_str)
            if isinstance(result, list):  # Ensure it is a list
                return result
        except (ValueError, SyntaxError):
            # Handle cases where the string cannot be converted
            print("Found string cannot be converted to a list of tuples.")
            return []
    else:
        # No matching pattern found
        print("No list of tuples found in the text.")
        return []


def vlm_extract_object_captions(text: str):
    # Replace newlines with spaces for uniformity
    text = text.replace('\n', ' ')

    # Pattern to match the list of objects
    pattern = r'\[(.*?)\]'

    # Search for the pattern in the text
    match = re.search(pattern, text)
    if match:
        # Extract the matched string
        list_str = match.group(0)
        try:
            # Try to convert the entire string to a list of dictionaries
            result = ast.literal_eval(list_str)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            # If the whole string conversion fails, process each element individually
            elements = re.findall(r'{.*?}', list_str)
            result = []
            for element in elements:
                try:
                    obj = ast.literal_eval(element)
                    if isinstance(obj, dict):
                        result.append(obj)
                except (ValueError, SyntaxError):
                    print(f"Error processing element: {element}")
            return result
    else:
        # No matching pattern found
        print("No list of objects found in the text.")
        return []


def get_obj_rel_from_image_gpt4v(client: OpenAI, image_path: str,
                                 label_list: list):
    """
이 함수는 **GPT-4 비전 모델(GPT-4V)**을 활용하여 이미지에서
    **객체 간의 물리적 관계**를 분석하고 추출하는 기능을 수행
특히, 이미지 내에서 객체들이 서로 **겹쳐져 있는지** 또는 **위아래로 쌓여 있는지**에 관한
    **"on top of"** 또는 **"under"**와 같은 관계를 찾아내는 데 초점

### 2. **세부 알고리즘 로직**

1. **이미지 인코딩**:
   - 함수는 먼저 주어진 이미지를 **Base64** 형식으로 인코딩
   - 이를 통해 GPT-4V가 이미지를 처리할 수 있음

2. **시스템 프롬프트 설정**:
   - 함수는 **GPT-4V 모델**에 전달될 시스템 프롬프트를 미리 정의
     - 이 프롬프트는 모델이 수행해야 할 작업을 명확히 설명해 줍니다.
   - 이 경우, 프롬프트는 **객체 간의 물리적 관계**(특히 "on top of"와 "under")만 분석하도록 명령

3. **사용자 쿼리 생성**:
   - 함수는 이미지와 함께 객체의 **라벨 목록**을 제공
        라벨 목록에는 이미지 내에서 탐지된 각 객체의 **고유 숫자 ID**와 그 **이름**이 포함
   - 예를 들어, "1: 의자", "2: 책상"과 같은 형식의 목록이 주어지며,
        - 이 정보가 모델이 객체 간 관계를 분석할 때 사용

4. **GPT-4V API 호출 및 응답 처리**:
   - OpenAI의 GPT-4V API가 호출되어, 이미지와 라벨 목록을 입력으로 받아 **객체 간의 관계**를 추론
   - 추론된 관계는 주어진 형식대로 **숫자 ID로 표현된 튜플**의 리스트로 반환
        [("1", "on top of", "2"), ("3", "under", "2")]

5. **관계 추출 및 오류 처리**:
   - 모델에서 반환된 결과는 함수가 **텍스트에서 튜플 형식으로 추출**하여 처리
   - 만약 API 호출 중 오류가 발생하면,
        - 빈 리스트가 반환되어 안전하게 작동할 수 있도록 **예외 처리**가 포함되어 있음

6. **결과 반환**:
   - 함수는 최종적으로 추출된 **객체 간의 관계** 리스트를 반환
     - 이 리스트는 **객체 ID**와 **관계 유형("on top of", "under")**으로 구성된 튜플 형태
    """
    # Getting the base64 string
    base64_image = encode_image_for_openai(image_path)

    global system_prompt
    global gpt_model

    user_query = (
        f"Here is the list of labels for the annotations of the objects "
        f"in the image: {label_list}. "
        f"Please describe the spatial relationships between "
        f"the objects in the image.")

    vlm_answer = []
    try:
        response = client.chat.completions.create(
            model=f"{gpt_model}",
            messages=[{
                "role": "system",
                "content": system_prompt_only_top
            }, {
                "role":
                    "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },],
            }, {
                "role": "user",
                "content": user_query
            }])

        vlm_answer_str = response.choices[0].message.content
        print(f"Line 113, vlm_answer_str: {vlm_answer_str}")

        vlm_answer = extract_list_of_tuples(vlm_answer_str)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        vlm_answer = []
    print(f"Line 68, user_query: {user_query}")
    print(f"Line 97, vlm_answer: {vlm_answer}")

    return vlm_answer


def get_obj_captions_from_image_gpt4v(client: OpenAI, image_path: str,
                                      label_list: list):
    """
이 함수는 **GPT-4 비전 모델(GPT-4V)**을 사용하여
    이미지에서 탐지된 객체에 대한 **간결한 설명(캡션)**을 생성
함수는 이미지에 있는 객체의 숫자 ID와 라벨 목록을 입력받아,
    각 객체를 분석한 후 **정확한 캡션**을 반환

### 1. **주요 역할**
- 이미지 속 **객체들에 대한 설명을 자동으로 생성**하는 것이 핵심
- 탐지된 객체에 대해 **객체 ID, 이름, 그리고 해당 객체에 대한 간결한 설명**을 포함한 목록을 반환

### 2. **세부 알고리즘 로직**

2. **시스템 프롬프트 설정**:
   - **GPT-4V**가 해야 할 작업을 정의하는 **프롬프트**를 설정합니다.
   이 프롬프트는 모델이 이미지 속 객체에 대해 간결하고 정확한 설명을 생성하도록 유도

3. **사용자 쿼리 생성**:
   - 탐지된 객체 목록(객체 ID 및 이름)을 포함한 쿼리를 생성하여 모델에 전달
   - 여기서 객체의 이름은 **객체 탐지 시스템**으로부터 얻은 정보이지만,
     - 정확하지 않을 수 있기 때문에 모델이 이를 기반으로 **객체 설명을 생성**

4. **GPT-4V API 호출 및 응답 처리**:
   - OpenAI의 **GPT-4V** 모델을 호출하여,
        - 이미지와 객체 라벨을 기반으로 각 객체에 대한 **설명을 생성**
   - 반환된 결과는
        - **Python 딕셔너리**의 리스트 형태
         - 각 객체에 대해 ID, 이름, 그리고 해당 객체의 설명이 포함된 딕셔너리

5. **오류 처리 및 결과 반환**:
   - API 호출 중 오류가 발생할 경우, 빈 리스트를 반환하도록 예외 처리가 되어 있습니다.
   - 함수는 최종적으로 각 객체에 대한 **ID, 이름, 캡션**을 포함한 리스트를 반환합니다.

[
    {"id": "1", "name": "object1", "caption": "concise description of object1"},
    {"id": "2", "name": "object2", "caption": "concise description of object2"}
]
    """
    # Getting the base64 string
    base64_image = encode_image_for_openai(image_path)

    global system_prompt

    user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please accurately caption the objects in the image."

    messages = [{
        "role": "system",
        "content": system_prompt_captions
    }, {
        "role":
            "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
            },
        },],
    }, {
        "role": "user",
        "content": user_query
    }]

    vlm_answer_captions = []
    try:
        response = client.chat.completions.create(model=f"{gpt_model}",
                                                  messages=messages)

        vlm_answer_str = response.choices[0].message.content
        print(f"Line 113, vlm_answer_str: {vlm_answer_str}")

        vlm_answer_captions = vlm_extract_object_captions(vlm_answer_str)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        vlm_answer_captions = []
    print(f"Line 68, user_query: {user_query}")
    print(f"Line 97, vlm_answer: {vlm_answer_captions}")

    return vlm_answer_captions
