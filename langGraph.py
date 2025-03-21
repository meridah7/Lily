import os
import json
import time
import random
import requests
from typing import TypedDict, Optional

from fastapi import FastAPI  # type: ignore
import uvicorn  # type: ignore

# LangGraph modules
from langgraph.graph import StateGraph, START, END

# LangChain modules
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# CLIP model for image evaluation
import torch
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from system_prompts import final_system_prompt, parse_text_template


# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Environment variable OPENAI_API_KEY is not set.")

leonardo_api_key = os.getenv("LEONARDO_API_KEY")
if not leonardo_api_key:
    raise ValueError("Environment variable LEONARDO_API_KEY is not set.")

print(f"[INFO] OpenAI API Key Loaded: {openai_api_key[:10]}********")
print(f"[INFO] Leonardo API Key Loaded: {leonardo_api_key[:10]}********")

# Initialize the LLM model
llm = ChatOpenAI(openai_api_key=openai_api_key, model="o1-mini")

# Define the shared workflow state
class State(TypedDict):
    user_input: str
    cleaned_data: Optional[dict]         # structured data after text parsing
    final_prompt: Optional[str]          # final prompt for image generation
    image_url: Optional[str]             # generated image URL
    clip_score: Optional[float]          # CLIP semantic match score
    aesthetic_score: Optional[float]     # CLIP aesthetic score
    passed: Optional[bool]               # whether the evaluation passed
    attempts: int                        # number of attempts
    error_message: Optional[str]         # error message (if any API call fails)
    user_feedback: Optional[str]         # human feedback (if any)
    user_accepted: bool                  # whether user accepted the result
    result: Optional[dict]               # final aggregated output

# ===================== Node Definitions =====================

# 1. parse_input: Use LLM to parse the user input into structured data
def parse_input(state: State) -> dict:
    if not state.get("user_input") or state["user_input"].strip() == "":
        error = "The user input is empty. Please provide valid text."
        print("[parse_input]", error)
        return {"error_message": error}
    


    prompt = (
        # "Please parse the text below, extract key descriptions, themes, and style information, "
        # "and output in JSON format.\n"
        f"{parse_text_template} Content: {state['user_input']}\n"
        # "Output must be JSON, containing the keys: 'description', 'keywords', 'style'."
    )
    messages = [HumanMessage(content=prompt)]
    try:
        response = llm.invoke(messages)
        raw_content = response.content
        print("[parse_input] LLM raw response:", raw_content)
        
        # If the content is a list, concatenate it
        if isinstance(raw_content, list):
            raw_content = "".join(str(item) for item in raw_content)
        
        # Remove Markdown code block markers
        if raw_content.startswith("```json"):
            raw_content = raw_content.split("```json", 1)[1]
        if raw_content.endswith("```"):
            raw_content = raw_content.rsplit("```", 1)[0]
        raw_content = raw_content.strip()
        
        cleaned = json.loads(raw_content)
        print("[parse_input] Successfully parsed:", cleaned)
        return {"cleaned_data": cleaned}
    except Exception as e:
        error = f"parse_input error: {e}"
        print("[parse_input]", error)
        return {"error_message": error}

# 2. data_extraction: Further process the parsed data (here, just pass through)
def data_extraction(state: State) -> dict:
    if state.get("cleaned_data") is None:
        return {"error_message": "data_extraction error: No valid parsed data"}
    return {"cleaned_data": state["cleaned_data"]}

# 3. prompt_generation: Generate the final image prompt based on the extracted data
def prompt_generation(state: State) -> dict:
    prompt = (
        # "Based on the following extracted data, generate a detailed image generation prompt, in plain text:\n"
        f"{final_system_prompt} Parsed data: {json.dumps(state['cleaned_data'], ensure_ascii=False)}\n"
        # "Consider style, subject, background, lighting, etc. Please limit to 200 words."
    )
    messages = [HumanMessage(content=prompt)]
    try:
        response = llm.invoke(messages)
        final_prompt = response.content.strip()
        print("[prompt_generation] Final prompt:", final_prompt)
        return {"final_prompt": final_prompt}
    except Exception as e:
        error = f"prompt_generation error: {e}"
        print("[prompt_generation]", error)
        return {"error_message": error}

# 4. handle_openai_error: Adjust the prompt based on the OpenAI error
def handle_openai_error(state: State) -> dict:
    error_msg = state.get("error_message", "")
    prompt = (
        "OpenAI prompt generation encountered an error. Error message:\n"
        f"{error_msg}\n"
        "Please adjust the original prompt based on the error, and output a new image generation prompt in plain text."
    )
    messages = [HumanMessage(content=prompt)]
    try:
        response = llm.invoke(messages)
        new_prompt = response.content.strip()
        print("[handle_openai_error] New prompt:", new_prompt)
        return {"final_prompt": new_prompt, "error_message": None}
    except Exception as e:
        error = f"handle_openai_error error: {e}"
        print("[handle_openai_error]", error)
        return {"error_message": error}

# 5. image_generation: Call the Leonardo API to generate an image
def image_generation(state: State) -> dict:
    final_prompt = state.get("final_prompt", "")
    if not final_prompt:
        return {"error_message": "image_generation error: No final prompt"}
    # Truncate prompt to avoid excessive length
    trimmed_prompt = final_prompt[:1450]
    try:
        response = requests.post(
            "https://cloud.leonardo.ai/api/rest/v1/generations",
            headers={
                "accept": "application/json",
                "authorization": f"Bearer {leonardo_api_key}",
                "content-type": "application/json"
            },
            json={
                "modelId": "b2614463-296c-462a-9586-aafdb8f00e36",
                "contrast": 3.5,
                "prompt": trimmed_prompt + " with the text FLUX",
                "num_images": 1,
                "width": 1472,
                "height": 832,
                "styleUUID": "111dc692-d470-4eec-b791-3475abac4c46",
            }
        ).json()
        
        # 如果返回结果中有 error 字段，直接处理
        print("response: ", response)
        if "error" in response:
            error_msg = response["error"].get("message", "Unknown error")
            print(f"[image_generation] API error: {error_msg}")
            return {"error_message": error_msg}
    
        # Extract generation ID
        generation_id = response.get("sdGenerationJob", {}).get("generationId")
        if not generation_id:
            error = "[image_generation] Could not get generationId"
            print(error)
            return {"error_message": error}
    
        # Poll for completion (in practice, use async)
        for attempt in range(5):
            print(f"[image_generation] Poll attempt {attempt+1}...")
            status_response = requests.get(
                f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}",
                headers={
                    "accept": "application/json",
                    "authorization": f"Bearer {leonardo_api_key}"
                }
            ).json()
            print(f"[DEBUG] [LeonardoChain] response status:\n{json.dumps(status_response, indent=2)}")
            generations_by_pk = status_response.get("generations_by_pk", {})
            if not generations_by_pk:
                print("[WARNING] [LeonardoChain] unable to find generations_by_pk, skip...")
                time.sleep(10)
                continue
            status = generations_by_pk.get("status")
            print(f"[INFO] [LeonardoChain] Current Status: {status}")
            if status and status.upper() == "COMPLETE":
                generated_images = generations_by_pk.get("generated_images", [])
                if generated_images:
                    image_url = generated_images[0].get("url")
                    print(f"[SUCCESS] [LeonardoChain] Image URL: {image_url}\n")
                    return {"image_url": image_url}
                else:
                    print("[WARNING] [LeonardoChain] Generation completed but no image URL found!\n")
                    return {"image_url": ""}
            time.sleep(10)
        error = "[image_generation] Timed out waiting for generation"
        return {"error_message": error}
    except Exception as e:
        error = f"[image_generation] exception: {e}"
        print(error)
        return {"error_message": error}


# 6. handle_leonardo_error: Adjust the prompt based on the Leonardo API error
def handle_leonardo_error(state: State) -> dict:
    error_msg = state.get("error_message", "")
    prompt = (
        "Leonardo image generation encountered an error. Error message:\n"
        f"{error_msg}\n"
        "Please adjust the image generation prompt accordingly, output a new final prompt in plain text."
    )
    messages = [HumanMessage(content=prompt)]
    try:
        response = llm.invoke(messages)
        new_prompt = response.content.strip()
        print("[handle_leonardo_error] New prompt:", new_prompt)
        return {"final_prompt": new_prompt, "error_message": None}
    except Exception as e:
        error = f"handle_leonardo_error error: {e}"
        print("[handle_leonardo_error]", error)
        return {"error_message": error}

# 7. CLIP-based image evaluation node
class ImageEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print("[ImageEvaluator] Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        print("[ImageEvaluator] CLIP model loaded successfully.")

    def evaluate_clip_from_path(self, image_path: str, text_description: str) -> float:
        image = Image.open(image_path).convert("RGB")  # type: ignore
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize([text_description]).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_features = self.clip_model.encode_text(text)
            similarity = (image_features @ text_features.T).item()
            similarity = (similarity + 1) / 2
        return similarity

    def evaluate_aesthetic_from_path(self, image_path: str) -> tuple[float, float]:
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        texts = ["a beautiful image", "an ugly image"]
        text_inputs = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_features = self.clip_model.encode_text(text_inputs)
            similarity = (image_features @ text_features.T).softmax(dim=-1)
        beauty_score = similarity[0][0].item()
        ugliness_score = similarity[0][1].item()
        return beauty_score, ugliness_score

# Instantiate the global CLIP evaluator
evaluator = ImageEvaluator()

import uuid

def evaluation(state: State) -> dict:
    """
    Evaluate the generated image with final_prompt via CLIP.
    If you prefer user_input for evaluation, switch final_prompt to user_input.
    Truncate the text to 50 chars max.
    """
    image_url = state.get("image_url", "")
    text_for_eval = state.get("user_input", "")
    if not image_url or not text_for_eval:
        return {"error_message": "evaluation error: missing image URL or text"}

    # Truncate text to 50 chars
    text_for_eval = text_for_eval[:50]

    try:
        resp = requests.get(image_url)
        resp.raise_for_status()

        save_dir = "Data/Generated"
        os.makedirs(save_dir, exist_ok=True)
        temp_filename = os.path.join(save_dir, f"temp_{uuid.uuid4().hex}.png")

        with open(temp_filename, "wb") as f:
            f.write(resp.content)

        # Wait 1s to ensure file system completion
        time.sleep(1)

        clip_score = evaluator.evaluate_clip_from_path(temp_filename, text_for_eval)
        beauty_score, _ = evaluator.evaluate_aesthetic_from_path(temp_filename)

        try:
            os.remove(temp_filename)
        except OSError:
            pass

        passed = (clip_score >= 0.8 and beauty_score >= 0.7)
        print(f"[evaluation] CLIP Score: {clip_score:.4f}, Beauty: {beauty_score:.4f}, Passed: {passed}")

        return {
            "clip_score": clip_score,
            "aesthetic_score": beauty_score,
            "passed": passed
        }
    except Exception as e:
        error = f"evaluation exception: {e}"
        print("[evaluation]", error)
        return {"error_message": error}

# feedback_reception (disabled by default)
def feedback_reception(state: State) -> dict:
    print("[feedback_reception] NULL")
    return {"user_feedback": None, "user_accepted": False}

# 9. decision: Decide next action based on evaluation results, feedback, attempts
def decision(state: State) -> dict:
    # If evaluation passed, attempts limit reached, or user accepted => finalize
    if state.get("passed") or state["attempts"] >= 1 or state.get("user_accepted"):
        print("[decision] Accepting the result, ending workflow.")
        return {"next_action": "accept"}
    # If there's an error => adjust prompt
    if state.get("error_message"):
        print("[decision] Error detected => adjust prompt.")
        return {"next_action": "adjust_prompt"}
    # Evaluate score
    if state.get("clip_score", 0) < 0.8:
        print("[decision] Low semantic alignment => adjust prompt.")
        return {"next_action": "adjust_prompt"}
    else:
        print("[decision] Image not aesthetically pleasing => regenerate image.")
        return {"next_action": "regenerate_image"}

# 10. adjust_prompt: Use LLM to adjust final_prompt
def adjust_prompt(state: State) -> dict:
    base_prompt = state.get("final_prompt", "")
    feedback = state.get("user_feedback", "")
    extra = feedback if feedback else state.get("error_message", "")
    prompt = (
        "Based on the prompt and feedback below, adjust the image generation prompt. Output plain text.\n"
        f"Original prompt: {base_prompt}\n"
        f"Feedback/Error: {extra}\n"
        "Please generate a new prompt."
    )
    messages = [HumanMessage(content=prompt)]
    try:
        response = llm.invoke(messages)
        new_prompt = response.content.strip()
        print("[adjust_prompt] New prompt:", new_prompt)
        return {"final_prompt": new_prompt, "error_message": None}
    except Exception as e:
        error = f"adjust_prompt error: {e}"
        print("[adjust_prompt]", error)
        return {"error_message": error}

# 11. regenerate_image: Retry image generation (minor modifications)
def regenerate_image(state: State) -> dict:
    # Increase attempts
    state["attempts"] += 1
    
    base = state.get("final_prompt", "")
    new_prompt = f"{base} retry-{random.randint(1,100)}"
    print("[regenerate_image] Adjusted prompt for retry:", new_prompt)
    state["final_prompt"] = new_prompt
    
    # Directly call image_generation with the updated final_prompt
    return image_generation(state)

# 12. finalize_output: Summarize the result
def finalize_output(state: State) -> dict:
    output = {
        "final_prompt": state.get("final_prompt"),
        "image_url": state.get("image_url"),
        "clip_score": state.get("clip_score"),
        "aesthetic_score": state.get("aesthetic_score"),
        "passed": state.get("passed"),
        "attempts": state.get("attempts"),
        "user_feedback": state.get("user_feedback")
    }
    print("[finalize_output] Final result:", output)

    # Save to Data/FinalOutput
    save_dir = "Data/FinalOutput"
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"final_output_{uuid.uuid4().hex}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[finalize_output] Result saved to: {filename}")
    
    return {"result": output}

# ===================== Build LangGraph Workflow =====================

graph = StateGraph(State)

graph.add_node("parse_input", parse_input)
graph.add_node("data_extraction", data_extraction)
graph.add_node("prompt_generation", prompt_generation)
graph.add_node("handle_openai_error", handle_openai_error)
graph.add_node("image_generation", image_generation)
graph.add_node("handle_leonardo_error", handle_leonardo_error)
graph.add_node("evaluation", evaluation)
graph.add_node("decision", decision)
graph.add_node("adjust_prompt", adjust_prompt)
graph.add_node("regenerate_image", regenerate_image)
graph.add_node("finalize_output", finalize_output)

# Start
graph.add_edge(START, "parse_input")

# parse_input -> data_extraction -> prompt_generation
graph.add_edge("parse_input", "data_extraction")
graph.add_edge("data_extraction", "prompt_generation")

# If prompt_generation errors => handle_openai_error, else image_generation
def check_prompt_error(state: State) -> str:
    return "handle_openai_error" if state.get("error_message") else "image_generation"

graph.add_conditional_edges("prompt_generation", check_prompt_error, {
    "handle_openai_error": "handle_openai_error",
    "image_generation": "image_generation"
})

# After handle_openai_error, go back to prompt_generation
graph.add_edge("handle_openai_error", "prompt_generation")

# If image_generation errors => handle_leonardo_error, else evaluation
def check_image_error(state: State) -> str:
    return "handle_leonardo_error" if state.get("error_message") else "evaluation"

graph.add_conditional_edges("image_generation", check_image_error, {
    "handle_leonardo_error": "handle_leonardo_error",
    "evaluation": "evaluation"
})

# After handle_leonardo_error => adjust_prompt
graph.add_edge("handle_leonardo_error", "adjust_prompt")

# After evaluation => either finalize_output if pass or attempts limit, else decision
def check_evaluation(state: State) -> str:
    if state.get("passed"):
        return "finalize_output"
    if state["attempts"] >= 1:
        return "finalize_output"
    return "feedback_reception"

graph.add_conditional_edges("evaluation", check_evaluation, {
    "finalize_output": "finalize_output",
    "decision": "decision"
})

# Decision -> next action
def route_decision(state: State) -> str:
    return state.get("next_action", "adjust_prompt")

graph.add_conditional_edges("decision", route_decision, {
    "adjust_prompt": "adjust_prompt",
    "regenerate_image": "regenerate_image",
    "accept": "finalize_output"
})

# After adjust_prompt => post_adjust => image_generation
def post_adjust(state: State) -> dict:
    state["attempts"] += 1
    return {}
graph.add_node("post_adjust", post_adjust)
graph.add_edge("adjust_prompt", "post_adjust")
graph.add_edge("post_adjust", "image_generation")

# regenerate_image => image_generation
graph.add_edge("regenerate_image", "image_generation")

# End
graph.add_edge("finalize_output", END)

# Compile the graph
compiled_graph = graph.compile()

# Generate Mermaid PNG
png_data = compiled_graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)

app = FastAPI()

from pydantic import BaseModel
class InputSchema(BaseModel):
    user_input: str

@app.post("/run_pipeline")
def run_pipeline(input_data: InputSchema):
    print("user_input:", input_data.user_input)
    initial_state: State = {
        "user_input": input_data.user_input,
        "cleaned_data": None,
        "final_prompt": None,
        "image_url": None,
        "clip_score": None,
        "aesthetic_score": None,
        "passed": False,
        "attempts": 0,
        "error_message": None,
        "user_feedback": None,
        "user_accepted": False,
        "result": None,
    }
    try:
        result_state = compiled_graph.invoke(initial_state)
        return {"result": result_state.get("result")}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
