parse_text_template="""
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

final_system_prompt="""
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

    The depiction should include diverse ethnicities in a respectful and inclusive manner. also try to add white people
    
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