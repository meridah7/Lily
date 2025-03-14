import torch
import clip
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


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


# Example usage
if __name__ == "__main__":
    try:
        # Initialize evaluator
        evaluator = ImageEvaluator()

        # Input image path and text description
        image_path = "Data/image1.png"  # Replace with your image path
        text_description = "New moms, your body needs fuel! Postpartum nutrition is key for recovery and energy. Focus on protein, calcium, and iron-rich foods. Stay hydrated, avoid harmful fish and excess caffeine. Keep healthy snacks handy. Talk to your doctor for personalized advice. Share your tips or tag a friend expecting! 💕 #PostpartumNutrition #NewMomTips"

         # Perform comprehensive evaluation
        results = evaluator.evaluate(image_path, text_description)
        print("Comprehensive Evaluation Results:", results)
        
        image_path = "Data/image2.png"  # Replace with your image path
        text_description = "New moms, your body needs fuel! Postpartum nutrition is key for recovery and energy. Focus on protein, calcium, and iron-rich foods. Stay hydrated, avoid harmful fish and excess caffeine. Keep healthy snacks handy. Talk to your doctor for personalized advice. Share your tips or tag a friend expecting! 💕 #PostpartumNutrition #NewMomTips"

        # Perform comprehensive evaluation
        results = evaluator.evaluate(image_path, text_description)
        print("Comprehensive Evaluation Results:", results)
    except Exception as e:
        print(f"Error during execution: {e}")