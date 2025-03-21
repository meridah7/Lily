from pydantic import BaseModel
from fastapi import FastAPI  # type: ignore
import uvicorn  # type: ignore
import os
import json
import time
import requests
from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from PIL import Image
import clip
import torch

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

##############################################################################
# 工具定义：每个工具都用 @tool 装饰，并提供说明以便 Agent 选择调用

def evaluate_clip_similarity(image_url: str, text_description: str) -> dict:
    """
    使用 CLIP 模型评估图像与文本的对齐程度。
    输入: 图像 URL 和文本描述。
    输出: 包含 "clip_score"（语义对齐分数）和 "aesthetic_scores"（美学分数）的字典。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # 下载并预处理图像
    image = Image.open(requests.get(image_url, stream=True).raw)
    image_input = preprocess(image).unsqueeze(0).to(device)

    # 预处理文本
    text = clip.tokenize([text_description]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text)

        # 计算语义对齐分数
        clip_score = (image_features @ text_features.T).item()
        clip_score = max(0, min(1, (clip_score + 1) / 2))  # 归一化到 [0, 1]

        # 计算美学分数
        aesthetic_texts = ["a beautiful image", "an ugly image"]
        aesthetic_inputs = clip.tokenize(aesthetic_texts).to(device)
        aesthetic_features = clip_model.encode_text(aesthetic_inputs)
        aesthetic_similarity = (image_features @ aesthetic_features.T).softmax(dim=-1)
        beauty_score = aesthetic_similarity[0][0].item()
        ugliness_score = aesthetic_similarity[0][1].item()

    print(f"[INFO] [evaluate_clip] CLIP 分数: {clip_score:.4f}, 美学分数: Beautiful={beauty_score:.4f}, Ugly={ugliness_score:.4f}")
    return {
        "clip_score": clip_score,
        "aesthetic_scores": {"beautiful": beauty_score, "ugly": ugliness_score},
    }

def generate_image_with_leonardo(prompt: str) -> str:
    """
    调用 Leonardo AI 生成图像并返回图像 URL。
    输入: 图像生成提示文本。
    输出: 图像的 URL。
    """
    print("[INFO] [generate_image] 开始调用 Leonardo AI 生成图像...")
    trimmed_prompt = prompt[:1450]  # 截断以防过长
    print(f"[DEBUG] [generate_image] 使用的提示文本（截断后）: {trimmed_prompt[:100]}...")
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
    print(f"[DEBUG] [generate_image] Leonardo 返回内容: {generation_response}")

    generation_id = generation_response.get("sdGenerationJob", {}).get("generationId")
    if not generation_id:
        error_msg = "[ERROR] [generate_image] 获取图像生成 ID 失败！"
        print(error_msg)
        raise ValueError(error_msg)

    print(f"[INFO] [generate_image] 生成任务已提交，任务 ID: {generation_id}")

    # 轮询检查状态
    for attempt in range(30):
        print(f"[INFO] [generate_image] 第 {attempt + 1} 次查询生成状态...")
        status_response = requests.get(
            f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}",
            headers={
                "accept": "application/json",
                "authorization": f"Bearer {leonardo_api_key}"
            }
        ).json()
        status = status_response.get("status")
        print(f"[INFO] [generate_image] 当前状态: {status}")

        if status == "completed":
            images = status_response.get("images", [])
            if images:
                image_url = images[0].get("url")
                print(f"[SUCCESS] [generate_image] 图像生成完成，URL: {image_url}")
                return image_url
            else:
                print("[WARNING] [generate_image] 生成完成，但未找到图像！")
                return ""
        time.sleep(10)
    raise TimeoutError("[ERROR] [generate_image] 图像生成超时")

def regenerate_prompt(original_prompt: str) -> str:
    """
    模拟重新生成提示文本。
    """
    print("[INFO] Regenerating prompt...")
    time.sleep(2)  # 模拟处理时间
    new_prompt = original_prompt + " Ensure the prompt is highly related to the input text."
    print(f"[INFO] New prompt generated: {new_prompt}")
    return new_prompt

##############################################################################
# 使用 Agent 的方式将工具整合成一个完整的流水线
tools = [
    Tool(
        name="evaluate_clip_similarity",
        func=lambda inputs: evaluate_clip_similarity(inputs["image_url"], inputs["text_description"]),
        description="使用 CLIP 模型评估图像与文本的对齐程度。输入是一个包含 'image_url' 和 'text_description' 的字典，输出是一个包含 'clip_score'（语义对齐分数）和 'aesthetic_scores'（美学分数）的字典。",
    ),
    Tool(
        name="generate_image_with_leonardo",
        func=generate_image_with_leonardo,
        description="调用 Leonardo AI 生成图像并返回图像 URL。输入是图像生成提示文本，输出是图像的 URL。",
    ),
    Tool(
        name="regenerate_prompt",
        func=regenerate_prompt,
        description="重新生成提示文本以改进语义相关性。输入是原始提示文本，输出是新的提示文本。",
    ),
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

##############################################################################
# FastAPI 路由：使用 Agent 完成完整流程，并加入评价阶段
@app.post("/run_pipeline_agent")
async def run_pipeline_agent(input_data: InputSchema):
    """
    使用 Agent 完成文本解析、图像提示生成和图像生成的流程，
    并在最后加入评价阶段，评估生成的图像与原始输入文本的对齐程度。
    """
    max_attempts = 5
    attempt = 0
    final_result = None

    while attempt < max_attempts:
        attempt += 1
        print(f"\n[INFO] [run_pipeline_agent] === 尝试次数: {attempt} ===")
        try:
            # Agent 根据输入自动调用工具完成整个流程
            result = agent.run({
                "input": input_data.user_input,
                "image_url": "",
                "text_description": input_data.user_input
            })
            print(f"[INFO] [run_pipeline_agent] Agent 调用成功，返回结果: {result}")
        except Exception as e:
            print(f"[ERROR] [run_pipeline_agent] Agent 调用失败: {e}")
            result = None

        if result is None:
            print("[WARNING] [run_pipeline_agent] 本次尝试未获得有效结果。")
        else:
            # 提取生成的图像 URL 和原始输入文本
            image_url = result.get("image_url", "")
            original_text = input_data.user_input

            if not image_url:
                print("[WARNING] [run_pipeline_agent] 未生成有效图像 URL。")
                continue

            # 使用 CLIP 评估图像与原始输入文本的对齐程度
            evaluation = evaluate_clip_similarity(image_url, original_text)
            clip_score = evaluation["clip_score"]
            beauty_score = evaluation["aesthetic_scores"]["beautiful"]

            print(f"[INFO] [run_pipeline_agent] 当前 CLIP 分数: {clip_score:.4f}, 美学分数: {beauty_score:.4f}")

            decision_input = f"""
            The evaluation results are as follows:
            - CLIP Score (Semantic Alignment): {clip_score:.4f}
            - Aesthetic Score (Beautiful): {beauty_score:.4f}

            Based on the following rules, decide the next step:
            1. If the CLIP Score is below 0.7, it indicates weak semantic alignment. In this case, return "[regenerate prompt]" and provide a detailed explanation of how to improve the final prompt.
            - Example: Suggest adding more descriptive keywords or refining the visual description to better align with the input text.
            2. If the Aesthetic Score is below 0.6, it indicates poor visual quality. In this case, return "[regenerate image]" and provide a detailed explanation of how to improve the image generation prompt.
            - Example: Suggest adjusting lighting, composition, or color schemes to enhance visual appeal.
            3. If both scores are satisfactory (CLIP Score >= 0.7 and Aesthetic Score >= 0.6), return "[final output]" to indicate that the process is complete.

            Provide clear instructions for each decision:
            - For "[regenerate prompt]", include a revised version of the final prompt with specific improvements.
            - For "[regenerate image]", include a revised version of the image generation prompt with specific improvements.
            - For "[final output]", confirm that no further action is needed.

            What should be the next step?
            """

            decision = agent.run({"input": decision_input})

            print(f"[INFO] [run_pipeline_agent] Agent Decision: {decision}")

            # Parse the decision result
            if "[final output]" in decision:
                final_result = {"image_url": image_url, "clip_score": clip_score, "beauty_score": beauty_score}
                print("[INFO] [run_pipeline_agent] CLIP score meets requirements, exiting retry.")
                break
            elif "[regenerate prompt]" in decision:
                print("[INFO] [run_pipeline_agent] Semantic alignment score does not meet requirements, preparing to regenerate the prompt.")
                # Extract the revised prompt
                new_prompt_start = decision.find("Revised Prompt:") + len("Revised Prompt:")
                new_prompt_end = decision.find("\n", new_prompt_start)
                revised_prompt = decision[new_prompt_start:new_prompt_end].strip()
                print(f"[INFO] [run_pipeline_agent] New prompt: {revised_prompt}")
                text_description = revised_prompt
            elif "[regenerate image]" in decision:
                print("[INFO] [run_pipeline_agent] Aesthetic score does not meet requirements, preparing to regenerate the image.")
                # Extract the revised image generation prompt
                new_image_prompt_start = decision.find("Revised Image Prompt:") + len("Revised Image Prompt:")
                new_image_prompt_end = decision.find("\n", new_image_prompt_start)
                revised_image_prompt = decision[new_image_prompt_start:new_image_prompt_end].strip()
                print(f"[INFO] [run_pipeline_agent] New image generation prompt: {revised_image_prompt}")
                image_path = regenerate_image()
            else:
                print("[INFO] [run_pipeline_agent] Unable to parse the Agent's decision, continuing to try...")
                time.sleep(1)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)