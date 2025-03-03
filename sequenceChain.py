from pydantic import BaseModel, Field
from fastapi import FastAPI  # type: ignore
import uvicorn  # type: ignore
import os
import json
import time
import requests
import random  # 用于模拟反馈

from langchain.chat_models import ChatOpenAI  # 使用最新 LangChain 结构
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chains.base import Chain

app = FastAPI()

# 环境变量检查
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not found. Please set it before running the application.")

leonardo_api_key = os.getenv("LEONARDO_API_KEY")
if not leonardo_api_key:
    raise ValueError("LEONARDO_API_KEY environment variable not found. Please set it before running the application.")

print(f"[INFO] OpenAI API Key Loaded: {openai_api_key[:10]}********")
print(f"[INFO] Leonardo API Key Loaded: {leonardo_api_key[:10]}********")

# 初始化 LLM 模型
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="o1-mini",
    temperature=1
)

# 定义 FastAPI 的输入 Schema
class InputSchema(BaseModel):
    user_input: str

class GenerationStatusRequest(BaseModel):
    generation_id: str


class LogWrapperChain(Chain):
    chain: Chain = Field(..., description="Wrapped Chain")
    name: str = Field(..., description="current Chain")

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @property
    def input_keys(self):
        return self.chain.input_keys

    @property
    def output_keys(self):
        return self.chain.output_keys

    def _call(self, inputs: dict) -> dict:
        result = self.chain._call(inputs)
        # 合并当前链的输出到输入字典中，供后续链使用
        merged = {**inputs, **result}
        return merged


##############################################################################
# Step 2: 文本解析链 —— 使用 LLMChain 解析文本并输出解析结果（JSON 格式字符串）
parse_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
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

    Key Guidelines:
    - Categories: Flexible and tailored to the specific content provided.
    - Characters: Ensure that the narrative visual description includes **no more than two people** in the scene.
    - Coherence: The generated keywords and narrative visual description should be coherent and reflect a realistic, health-conscious or lifestyle-related scenario.

    Content: {user_input}
    """
)
parse_chain = LLMChain(llm=llm, prompt=parse_prompt, output_key="parsed_result")
log_parse_chain = LogWrapperChain(chain=parse_chain, name="ParseChain")

class ExtractChain(Chain):
    @property
    def input_keys(self):
        return ["parsed_result"]

    @property
    def output_keys(self):
        return ["keywords", "visual_description"]

    def _call(self, inputs: dict) -> dict:
        parsed_result = inputs["parsed_result"]

        # 调试日志：打印原始输入
        print(f"[DEBUG] ExtractChain Input：\n{parsed_result}")

        # 清洗输入：移除 Markdown 代码块标记（如 ```json）
        if "```json" in parsed_result:
            parsed_result = parsed_result.split("```json")[1].split("```")[0].strip()
        elif "```" in parsed_result:
            parsed_result = parsed_result.split("```")[1].strip()

        # 尝试修复不完整的 JSON
        try:
            # 检查是否缺少闭合的括号
            if not parsed_result.strip().endswith("}"):
                parsed_result += "}"  # 尝试补全 JSON
            if not parsed_result.strip().endswith("}"):
                parsed_result += "}"  # 再次补全，确保完整性

            # 解析 JSON
            parsed_json = json.loads(parsed_result)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Fail to parse JSON：{e}")
            print(f"[ERROR] invalid JSON content\n{parsed_result}")
            raise ValueError(f"JSON error: {e}")

        # 提取所需字段
        keywords = parsed_json.get("Keywords", {})
        visual_description = parsed_json.get("Visual_Description", "")

        return {"keywords": keywords, "visual_description": visual_description}
extract_chain = ExtractChain() 
log_extract_chain = LogWrapperChain(chain=extract_chain, name="ExtractChain")

##############################################################################
# Step 3: 图像提示生成链 —— 使用 LLMChain 根据解析结果生成图像提示
generate_final_prompt = PromptTemplate(
    input_variables=["keywords", "visual_description"],
    template="""
    Please generate a detailed image generation prompt based on the content provided below. 
    The description should carefully outline the desired image scene, ensuring it is vivid and intricate. 

    ## Output Requirements:
    - Must return **pure JSON format**, do not use ```json or other Markdown markers
    - Ensure JSON keys are enclosed in double quotes

    Content Details:
    Keywords: {keywords}
    Visual_Description: {visual_description}

    Requirements:

    1.Poster Text:

    If the background includes posters or other text-rich information, generate a short, clear and meaningful title (e.g., "BMI Chart" or "Health Guidelines") that is as concise and clear as possible.
    For all text and charts on the poster except for the title, blur them.
    The text on any book, report, or clipboard held in hand does not need to be clear.
    
    
    2.Food Elements:
    If the keyword involves many types of food, include some real food items and display them in the background of a poster.
    If it’s an open book, illustrate some blurred food images, and ensure that the book’s face is oriented toward the person.
    Avoid showing raw meats, such as sashimi; however, if cooked salmon is mentioned, it can be included.
   
    3.Pregnancy Depiction:
    The scene must not show a naked, exposed pregnant belly.
    
    4.Diversity:

    The depiction should include diverse ethnicities in a respectful and inclusive manner.
    
    5.Doctor Representation:
    If there is a doctor in the image, the doctor must not be pregnant. Female doctor will be better.

    6.Others:
    - The description should be no more than 200 words.
    - Describe the background, lighting, main subjects, items, and overall mood.
    - The target audience is pregnant women; include calming, comforting, and supportive elements.
    - Specify the sex of characters in Main Subjects.
    - Consider incorporating style elements, aperture effects, and softening techniques to enhance the visual appeal.
    - Ensure the image description is highly related to the provided keywords
    - No more than two people in the image

    Examples:

    Main Subjects: 
    1. Pregnant Woman: A confident pregnant woman stands on a sleek scale, hands resting gently on her baby bump, smiling warmly. She wears stylish maternity attire in neutral tones.  
    2. Doctor: A compassionate female doctor stands beside her, holding a clipboard labeled *"Healthy Progress"* with a simple graph titled *"BMI & Pregnancy,"* showing smooth upward curves to represent healthy trends.  

    Background Elements:  
    - Poster: On the wall, a poster titled *"Nourish Your Body"* features illustrations of fruits like berries, oranges, and apples, with text: *"Fuel Your Journey."*  
    - Table: A wooden table is arranged with a vibrant selection of fresh fruits: a bowl of ripe berries, halved oranges, sliced apples, and a cluster of grapes. A few almonds and a small bunch of bananas are placed casually for added variety.  

    A plush armchair with a cozy throw blanket sits nearby for comfort. The overall design emphasizes calmness, trust, and empowerment, with soft textures and natural tones creating an inviting atmosphere. The focus remains on supporting a healthy pregnancy journey.

    """
)
generate_final_prompt_chain = LLMChain(llm=llm, prompt=generate_final_prompt, output_key="final_prompt")
log_prompt_chain = LogWrapperChain(chain=generate_final_prompt_chain, name="PromptChain")

##############################################################################
# Step 4: 图像生成链 —— 自定义 Chain 调用 Leonardo API 生成图像
# 此处定义为异步函数，通过 asyncio.run 在同步环境下调用
async def call_leonardo(final_prompt: str) -> dict:
    print("[INFO] [LeonardoChain] Connecting Leonardo AI for image generation...")
    trimmed_prompt = final_prompt[:1450]  # 截断以防过长
    print(f"[DEBUG] [LeonardoChain] trimmed Final prompt：\n{trimmed_prompt}\n")

    # 提交生成任务
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
            "prompt": trimmed_prompt+ "with the text FLUX",
            "num_images": 1,
            "width": 1472,
            "height": 832,
            "styleUUID": "111dc692-d470-4eec-b791-3475abac4c46",
            # "enhancePrompt": Auto,
            # "ultra": True,
        }
    ).json()

    print(f"[DEBUG] [LeonardoChain] Leonardo generation_response\n{json.dumps(generation_response, indent=2)}\n")

    # 提取生成任务 ID
    generation_id = generation_response.get("sdGenerationJob", {}).get("generationId")
    if not generation_id:
        error_msg = "[ERROR] [LeonardoChain]Fail to get generationID！"
        print(error_msg)
        raise ValueError(error_msg)
    print(f"[INFO] [LeonardoChain] task has been submitted，generationID！: {generation_id}\n")

    # 轮询检查状态
    for attempt in range(30):  # 最多尝试 30 次
        print(f"[INFO] [LeonardoChain]  {attempt + 1} times...")
        status_response = requests.get(
            f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}",
            headers={
                "accept": "application/json",
                "authorization": f"Bearer {leonardo_api_key}"
            }
        ).json()

        # 调试日志：打印完整的 status_response
        print(f"[DEBUG] [LeonardoChain] response status:\n{json.dumps(status_response, indent=2)}")

        # 检查状态
        generations_by_pk = status_response.get("generations_by_pk", {})  # 注意路径调整
        if not generations_by_pk:
            print("[WARNING] [LeonardoChain] unable to find generations_by_pk, skip...")
            time.sleep(10)
            continue

        status = generations_by_pk.get("status")
        print(f"[INFO] [LeonardoChain] Current Status: {status}")

        if status and status.upper() == "COMPLETE":
            # 提取生成的图像 URL
            generated_images = generations_by_pk.get("generated_images", [])
            if generated_images:
                image_url = generated_images[0].get("url")
                print(f"[SUCCESS] [LeonardoChain] Image URL: {image_url}\n")
                return {"image_url": image_url}
            else:
                print("[WARNING] [LeonardoChain] Image generation completed,but cannont find the image url!\n")
                return {"image_url": ""}

        time.sleep(10)  # 等待 10 秒后重试

    raise TimeoutError("[ERROR] [LeonardoChain] time out")
class LeonardoChain(Chain):
    @property
    def input_keys(self):
        return ["final_prompt"]

    @property
    def output_keys(self):
        return ["image_url"]

    async def _acall(self, inputs):
        final_prompt = inputs["final_prompt"]
        result = await call_leonardo(final_prompt)
        return result

    def _call(self, inputs):
        import asyncio
        return asyncio.run(self._acall(inputs))

log_leonardo_chain = LogWrapperChain(chain=LeonardoChain(), name="LeonardoChain")

##############################################################################
# 构造 SequentialChain，将所有步骤包装后串联起来
overall_chain = SequentialChain(
    chains=[log_parse_chain, log_extract_chain, log_prompt_chain, log_leonardo_chain],
    input_variables=["user_input"],
    output_variables=["keywords", "visual_description", "final_prompt", "image_url"],
    verbose=True,
)





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
        # 7182bbe1-889a-41ea-aed2-a46da1781ee
        response.raise_for_status()  
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

    return {}


##############################################################################
# 模拟反馈机制：随机返回通过或不通过（50% 概率）
def mock_feedback() -> bool:
    result = random.random() < 0.5
    print(f"[INFO] [feedback] Mock feedback: {'passed' if result else 'Failed'}")
    return result

##############################################################################
# FastAPI 路由：调用 SequentialChain 完成整个流程
@app.post("/run_pipeline_sequence")
def run_pipeline_sequence(input_data: InputSchema):
    """
    使用 SequentialChain 完成文本解析、图像提示生成和图像生成，
    每个步骤前后通过 LogWrapperChain 输出日志，最终返回整个流程的结果。
    """
    final_result = None

    try:
        print(f"[INFO] [run_pipeline_sequence] Start SequentialChain")
        result = overall_chain({"user_input": input_data.user_input})
        print(f"[INFO] [run_pipeline_sequence] SequentialChain successfully, result:\n{result}\n")
    except Exception as e:
        print(f"[ERROR] [run_pipeline_sequence] SequentialChain failed：{e}")
        result = None

    if result is None:
        print("[WARNING] [run_pipeline_sequence] No result")
    else:
        if True:
            final_result = result
            print("[INFO] [run_pipeline_sequence] Evaluation passed.")
        else:
            max_attempts = 2
            attempt = 0
            final_result = None
            while attempt < max_attempts:
                final_prompt = result.get("final_prompt") 

                result = log_leonardo_chain({"final_prompt": final_prompt})
                if mock_feedback():
                    final_result = result
                    print("[INFO] [run_pipeline_sequence] Evaluation passed.")
                    break
                attempt += 1
                time.sleep(1)
                    

    if final_result is None:
        print("[ERROR] [run_pipeline_sequence] After many attempts, no satisfactory results were obtained.")
        return {"result": "After many retries, failed to generate satisfactory image"}
    else:
        print("[INFO] [run_pipeline_sequence] ")
        return {"result": final_result}


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
