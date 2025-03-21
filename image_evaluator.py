# import torch
# import clip
# from PIL import Image
# import numpy as np
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


# class ImageEvaluator:
#     def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
#         """
#         Initialize CLIP model.
#         """
#         self.device = device

#         try:
#             # Load CLIP model
#             print("Loading CLIP model...")
#             self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
#             print("CLIP model loaded successfully!")
#         except Exception as e:
#             print(f"Error loading CLIP model: {e}")
#             raise

#         # Define aesthetic-related text descriptions
#         self.aesthetic_texts = ["a beautiful image", "an ugly image"]

#     def evaluate_clip(self, image_path, text_description):
#         """
#         Evaluate image-text alignment using CLIP.
#         """
#         image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
#         text = clip.tokenize([text_description]).to(self.device)

#         with torch.no_grad():
#             image_features = self.clip_model.encode_image(image)
#             text_features = self.clip_model.encode_text(text)

#             # Calculate cosine similarity
#             similarity = (image_features @ text_features.T).item()
#             similarity = (similarity + 1) / 2  # Normalize to [0, 1]

#         return similarity

#     def evaluate_aesthetic(self, image_path):
#         """
#         Evaluate image aesthetics using CLIP-based predictor.
#         """
#         image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
#         text_inputs = clip.tokenize(self.aesthetic_texts).to(self.device)

#         with torch.no_grad():
#             image_features = self.clip_model.encode_image(image)
#             text_features = self.clip_model.encode_text(text_inputs)

#             # Calculate aesthetic scores
#             similarity = (image_features @ text_features.T).softmax(dim=-1)

#         beauty_score = similarity[0][0].item()  # Probability of "a beautiful image"
#         ugliness_score = similarity[0][1].item()  # Probability of "an ugly image"

#         return beauty_score, ugliness_score

#     def evaluate(self, image_path, text_description):
#         """
#         Comprehensive evaluation of the image: semantic alignment and aesthetic quality.
#         """
#         # Evaluate semantic alignment
#         clip_score = self.evaluate_clip(image_path, text_description)

#         # Evaluate aesthetic quality
#         beauty_score, ugliness_score = self.evaluate_aesthetic(image_path)

#         print(f"CLIP Score (Semantic Alignment): {clip_score:.4f}")
#         print(f"Aesthetic Scores: Beautiful={beauty_score:.4f}, Ugly={ugliness_score:.4f}")

#         return {
#             "clip_score": clip_score,
#             "beauty_score": beauty_score,
#             "ugliness_score": ugliness_score,
#         }


# # Example usage
# if __name__ == "__main__":
#     try:
#         # Initialize evaluator
#         evaluator = ImageEvaluator()

#         # Input image path and text description
#         image_path = "Data/image1.png"  # Replace with your image path
#         text_description = "New moms, your body needs fuel! Postpartum nutrition is key for recovery and energy. Focus on protein, calcium, and iron-rich foods. Stay hydrated, avoid harmful fish and excess caffeine. Keep healthy snacks handy. Talk to your doctor for personalized advice. Share your tips or tag a friend expecting! 💕 #PostpartumNutrition #NewMomTips"

#          # Perform comprehensive evaluation
#         results = evaluator.evaluate(image_path, text_description)
#         print("Comprehensive Evaluation Results:", results)

#         image_path = "Data/image2.png"  # Replace with your image path
#         text_description = "New moms, your body needs fuel! Postpartum nutrition is key for recovery and energy. Focus on protein, calcium, and iron-rich foods. Stay hydrated, avoid harmful fish and excess caffeine. Keep healthy snacks handy. Talk to your doctor for personalized advice. Share your tips or tag a friend expecting! 💕 #PostpartumNutrition #NewMomTips"

#         # Perform comprehensive evaluation
#         results = evaluator.evaluate(image_path, text_description)
#         print("Comprehensive Evaluation Results:", results)
#     except Exception as e:
#         print(f"Error during execution: {e}")

import torch
import clip
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import random  # 用于模拟重新生成图片或提示
import time

class ImageEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize CLIP model.
        """
        self.device = device

        try:
            # Load CLIP model
            print("Loading CLIP model...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            print("CLIP model loaded successfully!")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise

        # Define aesthetic-related text descriptions
        self.aesthetic_texts = ["a beautiful image", "an ugly image"]

    def evaluate_clip(self, image_path, text_description):
        """
        Evaluate image-text alignment using CLIP.
        """
        image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        text = clip.tokenize([text_description]).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)

            # Calculate cosine similarity
            similarity = (image_features @ text_features.T).item()
            similarity = (similarity + 1) / 2  # Normalize to [0, 1]

        return similarity

    def evaluate_aesthetic(self, image_path):
        """
        Evaluate image aesthetics using CLIP-based predictor.
        """
        image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        text_inputs = clip.tokenize(self.aesthetic_texts).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text_inputs)

            # Calculate aesthetic scores
            similarity = (image_features @ text_features.T).softmax(dim=-1)

        beauty_score = similarity[0][0].item()  # Probability of "a beautiful image"
        ugliness_score = similarity[0][1].item()  # Probability of "an ugly image"

        return beauty_score, ugliness_score

    def evaluate(self, image_path, text_description):
        """
        Comprehensive evaluation of the image: semantic alignment and aesthetic quality.
        """
        # Evaluate semantic alignment
        clip_score = self.evaluate_clip(image_path, text_description)

        # Evaluate aesthetic quality
        beauty_score, ugliness_score = self.evaluate_aesthetic(image_path)

        print(f"CLIP Score (Semantic Alignment): {clip_score:.4f}")
        print(f"Aesthetic Scores: Beautiful={beauty_score:.4f}, Ugly={ugliness_score:.4f}")

        return {
            "clip_score": clip_score,
            "beauty_score": beauty_score,
            "ugliness_score": ugliness_score,
        }


# 模拟重新生成图像或提示文本
def regenerate_image():
    """
    Simulate regenerating an image.
    """
    print("[INFO] Regenerating image...")
    time.sleep(2)  # Simulate processing time
    new_image_path = f"Data/image{random.randint(1, 5)}.png"  # Randomly pick a new image
    print(f"[INFO] New image generated: {new_image_path}")
    return new_image_path


def regenerate_prompt_appealing(original_prompt):
    """
    Simulate regenerating a prompt based on the original input.
    """
    print("[INFO] Regenerating prompt...")
    time.sleep(2)  # Simulate processing time
    new_prompt = original_prompt + " Ensure the image is visually appealing."
    print(f"[INFO] New prompt: {new_prompt}")
    return new_prompt

def regenerate_prompt(original_prompt):
    """
    Simulate regenerating a prompt based on the original input.
    """
    print("[INFO] Regenerating prompt...")
    time.sleep(2)  # Simulate processing time
    new_prompt = original_prompt + " Ensure the new final prompt is highly related to the input text."
    print(f"[INFO] call generate_final_prompt tool...{new_prompt}")
    time.sleep(2)  # Simulate processing time
    new_prompt = "New moms, your body needs fuel! Postpartum nutrition is key for recovery and energy. Focus on protein, calcium, and iron-rich foods to support your health and your baby's growth. Stay hydrated, avoid harmful fish like raw sushi, and limit excess caffeine. Keep healthy snacks like fruits, nuts, and yogurt handy for quick energy boosts. Consult your doctor for personalized advice tailored to your needs. Share your tips or tag a friend who’s expecting! 💕 #PostpartumNutrition #NewMomTips"
    print(f"[INFO] New prompt: {new_prompt}")

    return new_prompt


# 主流程
def main_demo(image_path, text_description):
    """
    Main demo function to evaluate and refine image generation based on feedback.
    """
    evaluator = ImageEvaluator()

    max_attempts = 5
    attempt = 0
    final_result = None

    while attempt < max_attempts:
        attempt += 1
        print(f"\n[INFO] === Attempt {attempt} ===")

        # Evaluate current image and text
        results = evaluator.evaluate(image_path, text_description)
        clip_score = results["clip_score"]
        beauty_score = results["beauty_score"]
        ugliness_score = results["ugliness_score"]

        # Check if the image meets the criteria
        if clip_score >= 14 and beauty_score >= 1:
            print("[SUCCESS] Image is semantically aligned and aesthetically pleasing.")
            final_result = {"image_path": image_path, "text_description": text_description, "evaluation": results}
            break

        # If not, decide whether to regenerate the image or prompt
        if clip_score < 14:
            print("[INFO] Semantic alignment is weak. Regenerating prompt...")
            text_description = regenerate_prompt(text_description)
            regenerate_image()
        elif beauty_score < 1:
            print("[INFO] Image is not aesthetically pleasing. Regenerating image...")
            text_description = regenerate_prompt_appealing(text_description)
            image_path = regenerate_image()
            regenerate_image()

        # Wait before retrying
        time.sleep(1)

    if final_result is None:
        print("[ERROR] Failed to generate a satisfactory result after multiple attempts.")
        return {"result": "Failed to generate a satisfactory image."}
    else:
        print("[INFO] Final result:")
        return {"result": final_result}


# Example usage
if __name__ == "__main__":
    try:
        # Input image path and text description
        image_path = "Data/image1.png"  # Replace with your image path
        text_description = (
            "New moms, your body needs fuel! Postpartum nutrition is key for recovery and energy. "
            "Focus on protein, calcium, and iron-rich foods. Stay hydrated, avoid harmful fish and excess caffeine. "
            "Keep healthy snacks handy. Talk to your doctor for personalized advice. Share your tips or tag a friend expecting! 💕 "
            "#PostpartumNutrition #NewMomTips"
        )

        # Run the demo
        results = main_demo(image_path, text_description)
        print("Final Results:", results)

    except Exception as e:
        print(f"Error during execution: {e}")