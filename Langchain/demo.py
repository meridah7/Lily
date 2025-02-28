from pydantic import BaseModel
from fastapi import FastAPI, Request  # type: ignore
import uvicorn  # type: ignore
from langchain_community.chat_models import ChatOpenAI  # type: ignore # Updated import as per new LangChain structure
from langchain_core.messages import HumanMessage
import os
import json
import requests
import time


app = FastAPI()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not found. Please set it before running the application.")

leonardo_api_key = os.getenv("LEONARDO_API_KEY")
if not leonardo_api_key:
    raise ValueError("  environment variable not found. Please set it before running the application.")
print("leonardo_api_key: ", leonardo_api_key)


llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="o1-mini",
    temperature=1
)

class InputSchema(BaseModel):
    user_input: str

class PromptSchema(BaseModel):
    prompt: str

class GenerationStatusRequest(BaseModel):
    generation_id: str

async def parse_text_with_langchain(user_input: str):
    prompt = f"""
    Based on the given content, generate classified keywords and a corresponding narrative visual description. 
    Ensure that the output is related to the post content, reasonable, and represents a realistic scenario. 
    The keywords should be classified into categories that best fit the context of the content (categories do not need to be fixed). 
    The narrative visual description should align with the classified keywords.

    If the article includes some medical-related metrics, such as BMI and blood sugar levels, please lean towards a medical scenario involving a doctor. 
    However, if these metrics are not present, this is not necessary.
    If the keyword involves many types of food, include some real food items and display them in the background of a poster. 
    If it's an open book, try to illustrate some blurred food images. Avoid showing raw meats, such as sashimi, but cooked salmon can be included if mentioned.

    Please perform a full analysis of the primary keywords and the secondary keywords in the entire text. 
    Based on the primary keywords, generate a Visual_Description, with the other keywords serving as supplementary information.

    ### Example Output:
    json: ```{{
      \"Keywords\": {{
        \"Most important keywords\": {{
          \"Health Metrics\": [
            \"BMI\",
            \"Weight Gain Curve\"
          ]
        }},
        \"Less important keywords\": {{
          \"Potential Risks\": [
            \"Gestational Diabetes\"
          ],
          \"Lifestyle & Nutrition\": [
            \"Healthy Weight Gain\",
            \"Balanced Diet\"
          ]
        }}
      }},

      \"Visual_Description\": \"The image shows a pregnant woman standing on a scale, smiling at the camera, while a doctor holds a health record chart displaying her BMI and pregnancy weight gain curve. In the background is a cozy clinic with a poster on the wall labeled 'Pregnancy Health Guidelines,' showing recommended weight gain ranges based on BMI (e.g., 25-35 pounds). On the table are models of healthy foods like fruits, vegetables, whole-grain bread, and nuts, symbolizing a nutrient-rich diet. At the bottom right corner of the image, there's a small calendar marking the three trimesters to emphasize changes during each stage. The overall color tone is warm and soft, conveying a theme of health and care.\"
    }}
    ```

    ### Key Guidelines:
    - **Categories**: Flexible and tailored to the specific content provided.
    - **Characters**: Ensure that the narrative visual description includes **no more than two people** in the scene.
    - **Coherence**: The generated keywords and narrative visual description should be coherent and reflect a realistic, health-conscious or lifestyle-related scenario.

    Content: {user_input}
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


async def generate_final_prompt(parsed_result: dict):
    keywords = json.dumps(parsed_result.get("Keywords", {}), indent=2)
    visual_description = parsed_result.get("Visual_Description", "")

    prompt_template = f"""
    Please generate a detailed image generation prompt based on the content provided below. 
    The description should carefully outline the desired image scene, ensuring it is vivid and intricate. 

    Content Details:
    Keywords: {keywords}
    Visual_Description: {visual_description}

    Requirements:
    - The description should be no more than 200 words.
    - Describe the background, lighting, main subjects, items, and overall mood.
    - The target audience is pregnant women; include calming, comforting, and supportive elements.
    - Specify the sex of characters in Main Subjects.
    - For any background posters, generate a clear and meaningful title, blur all other text.
    - If food is mentioned in keywords, include real food items in the background of a poster.
    - Avoid raw meats; cooked salmon can be included if mentioned.
    - Use style elements, aperture effects, and softening techniques to enhance visual appeal.

    Example Output Format:
    **Main Subjects:**
    1. **Pregnant Woman**: ...
    2. **Doctor**: ...

    **Background Elements:**
    - **Poster**: ...
    - **Table**: ...

    Overall, ensure the scene conveys the core message and emotion of the content.
    """

    response = llm.invoke([HumanMessage(content=prompt_template)])
    return response.content

def generate_image_with_leonardo(prompt: str):
    trimmed_prompt = prompt[:1450]
    generation_response = requests.post(
        "https://cloud.leonardo.ai/api/rest/v1/generations",
        headers={
            "accept": "application/json",
            "authorization": f"Bearer {leonardo_api_key}",
            "content-type": "application/json"
        },
        json={
            "modelId": "b2614463-296c-462a-9586-aafdb8f00e36",
            "contrast": 3.5,
            "prompt": trimmed_prompt,
            "num_images": 1,
            "width": 1472,
            "height": 832,
            "styleUUID": "111dc692-d470-4eec-b791-3475abac4c46",
            "enhancePrompt": False
        }
    ).json()

    print("generation_response: ", generation_response)

    generation_id = generation_response.get("sdGenerationJob", {}).get("generationId")
    if not generation_id:
        raise ValueError("Failed to get generation ID from Leonardo API response.")

    # 轮询查询生成状态
    for attempt in range(30):
        print(f"[Attempt {attempt + 1}/10] Checking generation status for ID: {generation_id}...")

        status_response = requests.get(
            f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}",
            headers={
                "accept": "application/json",
                "authorization": f"Bearer {leonardo_api_key}"
            }
        ).json()

        status = status_response.get("status")
        print(f"[Attempt {attempt + 1}] Current status: {status}")

        if status == "completed":
            images = status_response.get("images", [])
            if images:
                image_url = images[0].get("url")
                print(f"[Attempt {attempt + 1}] Image generation completed. Image URL: {image_url}")
                return image_url
            else:
                print(f"[Attempt {attempt + 1}] Generation completed, but no images found.")
                return None
        else:
            print(f"[Attempt {attempt + 1}] Generation not completed yet. Retrying in 10 seconds...")
            time.sleep(10)

    raise TimeoutError("Image generation timed out after multiple attempts.")

import requests

def get_generation_status(generation_id: str, leonardo_api_key: str) -> dict:
    """
    调用 Leonardo AI API 来获取指定 generation_id 的生成状态。

    Args:
        generation_id (str): 图像生成任务的唯一标识符。
        leonardo_api_key (str): 访问 Leonardo AI API 的密钥。

    Returns:
        dict: 包含生成状态和相关信息的响应字典。
    """
    try:
        response = requests.get(
            f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}",
            headers={
                "accept": "application/json",
                "authorization": f"Bearer {leonardo_api_key}"
            }
        )
        response.raise_for_status()  
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

    return {}



@app.post("/parse_text_step2")
async def handle_text_parsing(input_data: InputSchema):
    parsed_result = await parse_text_with_langchain(input_data.user_input)
    return {"parsed_result": parsed_result}


@app.post("/generate_prompt_step3")
async def handle_prompt_generation(input_data: InputSchema):
    parsed_result = json.loads(input_data.user_input)
    final_prompt = await generate_final_prompt(parsed_result)
    return {"final_prompt": final_prompt}


@app.post("/generate_image_step4")
async def handle_image_generation(input_data: PromptSchema):
    image_url = generate_image_with_leonardo(input_data.prompt)
    return {"image_url": image_url}

@app.post("/get_generation_status")
async def generation_status(request: GenerationStatusRequest):
    """
    FastAPI 路由：根据 generation_id 返回 Leonardo AI 生成状态。

    请求示例:
    {
        "generation_id": "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
    }
    """
    status_response = get_generation_status(request.generation_id, leonardo_api_key)
    return {"status_response": status_response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
