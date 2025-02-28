from pydantic import BaseModel
from fastapi import FastAPI  # type: ignore
import uvicorn  # type: ignore
import os
import json
import time
import requests
import random  # 用于模拟反馈

from langchain_community.chat_models import ChatOpenAI  # 使用最新 LangChain 结构
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain_core.messages import HumanMessage

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

class PromptSchema(BaseModel):
    prompt: str

class GenerationStatusRequest(BaseModel):
    generation_id: str

##############################################################################
# 工具定义：每个工具都用 @tool 装饰，并提供说明以便 Agent 选择调用

@tool
async def parse_text_with_langchain(user_input: str) -> dict:
    """解析文本并返回关键词和视觉描述"""
    print(f"[INFO] [parse_text] 开始解析文本: {user_input[:50]}...")
    prompt = f"""
Based on the given content, generate classified keywords and a corresponding narrative visual description.
Ensure that the output is related to the post content, reasonable, and represents a realistic scenario.

Content: {user_input}
"""
    print(f"[DEBUG] [parse_text] Prompt 内容:\n{prompt}")
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"[DEBUG] [parse_text] LLM 返回内容: {response.content[:100]}...")
    try:
        parsed_result = json.loads(response.content)
        print(f"[SUCCESS] [parse_text] 解析成功: {parsed_result}")
    except json.JSONDecodeError:
        error_msg = "[ERROR] [parse_text] JSON解析失败，请检查提示语格式。"
        print(error_msg)
        raise ValueError(error_msg)
    return parsed_result

@tool
async def generate_final_prompt(parsed_result: dict) -> str:
    """根据解析结果生成图像描述提示"""
    print("[INFO] [generate_prompt] 开始生成图像提示...")
    keywords = json.dumps(parsed_result.get("Keywords", {}), indent=2)
    visual_description = parsed_result.get("Visual_Description", "")
    prompt_template = f"""
Please generate a detailed image generation prompt based on the following parsed result.
Parsed Result: 
Keywords: {keywords}
Visual_Description: {visual_description}
"""
    print(f"[DEBUG] [generate_prompt] 使用的模板:\n{prompt_template}")
    response = llm.invoke([HumanMessage(content=prompt_template)])
    print(f"[SUCCESS] [generate_prompt] 生成提示成功: {response.content[:100]}...")
    return response.content

@tool
def generate_image_with_leonardo(prompt: str) -> str:
    """调用 Leonardo AI 生成图像并返回图像 URL"""
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

##############################################################################
# 使用 Agent 的方式将三个工具整合成一个完整的流水线
tools = [parse_text_with_langchain, generate_final_prompt, generate_image_with_leonardo]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

##############################################################################
# 模拟反馈机制：随机返回通过或不通过（50% 概率）
def mock_feedback() -> bool:
    result = random.random() < 0.5
    print(f"[INFO] [feedback] 模拟反馈结果: {'通过' if result else '不通过'}")
    return result

##############################################################################
# FastAPI 路由：使用 Agent 完成完整流程，并加入反馈环路
@app.post("/run_pipeline_agent")
async def run_pipeline_agent(input_data: InputSchema):
    """
    使用 Agent 完成文本解析、图像提示生成和图像生成的流程，
    并基于模拟反馈机制自动重试，最多重试 5 次。
    """
    max_attempts = 5
    attempt = 0
    final_result = None

    while attempt < max_attempts:
        attempt += 1
        print(f"\n[INFO] [run_pipeline_agent] === 尝试次数: {attempt} ===")
        try:
            # Agent 根据输入自动调用工具完成整个流程
            result = agent.invoke({"input": input_data.user_input})
            print(f"[INFO] [run_pipeline_agent] Agent 调用成功，返回结果: {result}")
        except Exception as e:
            print(f"[ERROR] [run_pipeline_agent] Agent 调用失败: {e}")
            result = None

        if result is None:
            print("[WARNING] [run_pipeline_agent] 本次尝试未获得有效结果。")
        else:
            # 模拟反馈，如果通过则退出循环
            if mock_feedback():
                final_result = result
                print("[INFO] [run_pipeline_agent] 反馈通过，退出重试。")
                break
            else:
                print("[INFO] [run_pipeline_agent] 反馈不通过，准备重新生成图像。")
                time.sleep(1)  # 重试前等待

    if final_result is None:
        print("[ERROR] [run_pipeline_agent] 多次尝试后仍未获得满意结果。")
        return {"result": "经过多次重试后，未能生成满意的图像。"}
    else:
        print("[INFO] [run_pipeline_agent] 最终结果已生成。")
        return {"result": final_result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
