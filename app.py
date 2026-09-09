import streamlit as st
import torch
import torch.nn as nn
from datetime import datetime
from torchvision import models, transforms
from PIL import Image
import json
import html
import re
import time
from google import genai
from google.genai import types


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FoodLens AI",
    page_icon="🍽️",
    layout="wide"
)


# ============================================================
# GEMINI CONNECTION
# ============================================================

@st.cache_resource
def init_gemini():

    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
        http_options=types.HttpOptions(
            timeout=20000,
            retry_options=types.HttpRetryOptions(
                attempts=1
            )
        )
    )


gemini_client = init_gemini()


# ============================================================
# SESSION STATE
# ============================================================

if "ai_chef_messages" not in st.session_state:
    st.session_state.ai_chef_messages = []

user_name = "Food Explorer"


# ============================================================
# MODEL + RECIPE CONFIGURATION
# ============================================================

MODEL_PATH = "./models/food101_resnet50.pth"
RECIPE_PATH = "./recipes/recipes.json"


# ============================================================
# CONFIDENCE THRESHOLDS
# ============================================================

HIGH_CONFIDENCE = 70
MEDIUM_CONFIDENCE = 40


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .stApp {
        background: #FFF9F2;
    }

    .block-container {
        padding-top: 3.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero-title {
        font-size: 58px;
        font-weight: 800;
        color: #2D241F;
        margin-bottom: 5px;
        letter-spacing: -2px;
    }

    .hero-title span {
        color: #E86A33;
    }

    .hero-subtitle {
        font-size: 22px;
        color: #6F625A;
        margin-bottom: 30px;
    }

    .hero-card {
        background: #FFF0E3;
        border-radius: 28px;
        padding: 45px;
        margin-bottom: 30px;
        border: 1px solid #F5DCC8;
    }

    .section-title {
        font-size: 30px;
        font-weight: 750;
        color: #2D241F;
        margin-bottom: 8px;
    }

    .section-text {
        color: #766A62;
        font-size: 16px;
        margin-bottom: 20px;
    }

    .feature-card {
        background: white;
        border-radius: 20px;
        padding: 24px;
        min-height: 165px;
        box-sizing: border-box;
        border: 1px solid #F0E2D5;
        box-shadow: 0 5px 20px rgba(80, 50, 30, 0.06);
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        overflow: hidden;
        transition:
            box-shadow 0.3s ease,
            border-color 0.3s ease;
        animation: floatingCard 4s ease-in-out infinite;
    }

    .feature-card-1 {
        animation-delay: 0s;
    }

    .feature-card-2 {
        animation-delay: 0.8s;
    }

    .feature-card-3 {
        animation-delay: 1.6s;
    }

    @keyframes floatingCard {

        0% {
            transform: translateY(0px);
        }

        50% {
            transform: translateY(-7px);
        }

        100% {
            transform: translateY(0px);
        }

    }

    .feature-card:hover {
        box-shadow:
            0 14px 32px rgba(80, 50, 30, 0.12);

        border-color: #E8CDB8;
    }

    .feature-icon {
        font-size: 30px;
        line-height: 1;
        height: 34px;
        display: flex;
        align-items: center;
        margin-bottom: 13px;
    }

    .feature-title {
        font-size: 17px;
        font-weight: 700;
        color: #2D241F;
        line-height: 1.3;
        margin-bottom: 7px;
    }

    .feature-text {
        font-size: 13px;
        color: #766A62;
        line-height: 1.5;
        max-width: 260px;
    }

    .stButton > button {
        background: #E86A33;
        color: white;
        border: none;
        border-radius: 14px;
        padding: 12px 28px;
        font-weight: 700;
        font-size: 16px;
        width: 100%;
    }

    .stButton > button:hover {
        background: #D95724;
        color: white;
    }

    .result-card {
        background: white;
        padding: 30px;
        border-radius: 22px;
        border: 1px solid #F0E2D5;
        box-shadow: 0 8px 30px rgba(80, 50, 30, 0.08);
        margin-top: 10px;
    }

    .result-label {
        font-size: 17px;
        color: #8A7D74;
        margin-bottom: 8px;
    }

    .result-food {
        font-size: 34px;
        font-weight: 800;
        color: #2D241F;
        margin-bottom: 12px;
        text-transform: capitalize;
    }

    .confidence-label {
        font-size: 15px;
        color: #766A62;
        margin-bottom: 6px;
    }

    .confidence-value {
        font-size: 24px;
        font-weight: 800;
        color: #E86A33;
    }

    .recipe-card {
        background: white;
        border-radius: 24px;
        padding: 30px;
        border: 1px solid #F0E2D5;
        box-shadow: 0 8px 30px rgba(80, 50, 30, 0.07);
        margin-bottom: 24px;
    }

    .recipe-title {
        font-size: 25px;
        font-weight: 800;
        color: #2D241F;
        margin-bottom: 24px;
    }

    .nutrition-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin-bottom: 26px;
    }

    .nutrition-box {
        background: #FFF0E3;
        border-radius: 16px;
        padding: 17px 10px;
        text-align: center;
        border: 1px solid #F5DCC8;
    }

    .nutrition-number {
        font-size: 20px;
        font-weight: 800;
        color: #E86A33;
    }

    .nutrition-label {
        font-size: 12px;
        color: #766A62;
        margin-top: 4px;
    }

    .recipe-heading {
        font-size: 18px;
        font-weight: 750;
        color: #2D241F;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .ingredient-item {
        color: #5F5149;
        padding: 5px 0;
        line-height: 1.5;
        font-size: 15px;
    }

    .instruction-item {
        color: #5F5149;
        padding: 6px 0;
        line-height: 1.55;
        font-size: 15px;
    }

    .instruction-number {
        font-weight: 800;
        color: #E86A33;
    }

    .low-confidence-card {
        background: #FFF3E8;
        border: 1px solid #F3C9A8;
        border-radius: 18px;
        padding: 18px 22px;
        margin-top: 18px;
        margin-bottom: 0px;
    }

    .low-confidence-title {
        font-size: 16px;
        font-weight: 750;
        color: #9A4F20;
        margin-bottom: 6px;
    }

    .low-confidence-text {
        font-size: 14px;
        color: #765F52;
        line-height: 1.6;
    }

    .possible-recipe-note {
        background: #FFF8F2;
        border: 1px solid #F0E2D5;
        border-radius: 14px;
        padding: 13px 16px;
        margin-bottom: 15px;
        color: #766A62;
        font-size: 14px;
        line-height: 1.5;
    }

    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 2px solid #E8CDB8 !important;
        border-radius: 18px !important;
        box-shadow: 0 6px 20px rgba(80, 50, 30, 0.08) !important;
        margin-bottom: 18px !important;
        overflow: hidden !important;
    }

    div[data-testid="stExpander"] details {
        background: #FFFFFF !important;
        border-radius: 18px !important;
    }

    div[data-testid="stExpander"] summary {
        background: #FFF0E3 !important;
        color: #2D241F !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        font-size: 16px !important;
        font-weight: 750 !important;
        min-height: 55px !important;
    }

    div[data-testid="stExpander"] summary:hover {
        background: #FFE5D3 !important;
    }

    div[data-testid="stExpander"] summary span {
        color: #2D241F !important;
        font-weight: 750 !important;
    }

    div[data-testid="stExpander"] summary svg {
        color: #E86A33 !important;
        fill: #E86A33 !important;
    }

    div[data-testid="stExpander"] > div {
        background: #FFFFFF !important;
        border-radius: 0 0 18px 18px !important;
    }


    /* ========================================================
       AI CHEF — ONE COMPLETE FLOATING BOX
       ======================================================== */

    .st-key-ai_chef_chat_box {

        background: #FFFFFF !important;

        border: 1px solid #E8D8CB !important;

        border-radius: 28px !important;

        padding: 32px 34px 28px 34px !important;

        box-shadow:
            0 12px 38px rgba(80, 50, 30, 0.10) !important;

        animation:
            floatingAIChat 4.5s ease-in-out infinite;

        margin-top: 8px !important;

        margin-bottom: 28px !important;

    }


    @keyframes floatingAIChat {

        0% {
            transform: translateY(0px);
        }

        50% {
            transform: translateY(-7px);
        }

        100% {
            transform: translateY(0px);
        }

    }


    /* --------------------------------------------------------
       AI CHEF HEADER
       -------------------------------------------------------- */

    .ai-chef-header {
        margin-bottom: 20px;
    }

    .ai-chef-title {
        font-size: 30px;
        font-weight: 800;
        color: #2D241F !important;
        margin-bottom: 5px;
        line-height: 1.25;
    }

    .ai-chef-subtitle {
        font-size: 15px;
        color: #6F625A !important;
        line-height: 1.55;
    }


    /* --------------------------------------------------------
       READY / INFO BOX
       -------------------------------------------------------- */

    .ai-chef-ready {

        background: #FFF0E3 !important;

        border: 1px solid #F5DCC8 !important;

        border-radius: 17px !important;

        padding: 15px 18px !important;

        margin-bottom: 22px !important;

        color: #5F5149 !important;

        line-height: 1.55 !important;

    }

    .ai-chef-ready strong {
        color: #2D241F !important;
    }


    .ai-chef-info {

        background: #FFF8F2 !important;

        border: 1px solid #F0E2D5 !important;

        border-radius: 17px !important;

        padding: 15px 18px !important;

        margin-bottom: 22px !important;

        color: #766A62 !important;

        line-height: 1.55 !important;

    }


    /* --------------------------------------------------------
       CHAT HISTORY
       -------------------------------------------------------- */

    .ai-chef-history {

        width: 100%;

        margin-bottom: 18px;

    }


    /* --------------------------------------------------------
       USER MESSAGE
       -------------------------------------------------------- */

    .ai-chef-user-message {

        width: 100%;

        box-sizing: border-box;

        padding: 13px 16px;

        margin: 8px 0 10px 0;

        background: transparent;

        border: none;

        color: #2D241F !important;

        font-size: 15px;

        line-height: 1.55;

        font-weight: 600;

    }


    .ai-chef-user-label {

        color: #E86A33 !important;

        font-weight: 800;

        margin-right: 5px;

    }


    /* --------------------------------------------------------
       AI MESSAGE CARD
       -------------------------------------------------------- */

    .ai-chef-ai-message {

        width: 100%;

        box-sizing: border-box;

        padding: 14px 17px 16px 17px;

        margin: 0 0 12px 0;

        background: #FFF9F2;

        border-radius: 16px;

        color: #2D241F !important;

        font-size: 15px;

        line-height: 1.65;

        border: 1px solid #F0E2D5;

    }


    .ai-chef-ai-label {

        color: #E86A33 !important;

        font-weight: 800;

        display: block;

        margin-bottom: 7px;

    }


    .ai-chef-ai-content {

        color: #2D241F !important;

        font-size: 15px;

        line-height: 1.65;

    }


    .ai-chef-ai-content p {

        color: #2D241F !important;

        margin: 5px 0 9px 0;

    }


    .ai-chef-ai-content strong {

        color: #2D241F !important;

        font-weight: 800 !important;

    }


    .ai-chef-ai-content ul,

    .ai-chef-ai-content ol {

        color: #2D241F !important;

        margin-top: 6px;

        margin-bottom: 9px;

        padding-left: 25px;

    }


    .ai-chef-ai-content li {

        color: #2D241F !important;

        margin-bottom: 5px;

    }


    /* --------------------------------------------------------
       CHAT FORM
       -------------------------------------------------------- */

    .ai-chef-form {

        margin-top: 8px;

    }


    .st-key-ai_chef_form {

        border: none !important;

        padding: 0 !important;

        margin: 0 !important;

    }


    .ai-chef-input-label {

        color: #5F5149 !important;

        font-size: 15px !important;

        font-weight: 700 !important;

        margin-top: 8px !important;

        margin-bottom: 8px !important;

        opacity: 1 !important;

    }


    .ai-chef-input-hint {
        color: #8A7D74 !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
        margin-top: -2px !important;
        margin-bottom: 10px !important;
        opacity: 1 !important;
    }


    .st-key-ai_chef_input input {

        background: #FFFFFF !important;

        color: #2D241F !important;

        border: 1px solid #E8D8CB !important;

        border-radius: 15px !important;

        min-height: 48px !important;

        font-size: 15px !important;

    }


    .st-key-ai_chef_input input::placeholder {

        color: #766A62 !important;

        opacity: 1 !important;

        font-weight: 500 !important;

    }


    /* Visible Streamlit warning when the question is empty */

    div[data-testid="stAlert"] {

        background: #FFFBD6 !important;

        border: 1px solid #F1E7A5 !important;

        border-radius: 12px !important;

        color: #5F5149 !important;

    }


    div[data-testid="stAlert"] * {

        color: #5F5149 !important;

        opacity: 1 !important;

    }


    .st-key-ai_chef_input input:focus {

        border-color: #E86A33 !important;

        box-shadow:
            0 0 0 1px #E86A33 !important;

    }


    /* --------------------------------------------------------
       SEND BUTTON
       -------------------------------------------------------- */

    .st-key-ai_chef_send button {

        min-height: 48px !important;

        border-radius: 15px !important;

        margin-top: 0px !important;

        font-size: 15px !important;

        font-weight: 700 !important;

    }


    /* --------------------------------------------------------
       CLEAR BUTTON
       -------------------------------------------------------- */

    .st-key-clear_ai_chef {

        margin-top: 10px !important;

    }


    .st-key-clear_ai_chef button {

        background: #FFF8F2 !important;

        color: #766A62 !important;

        border: 1px solid #E8D8CB !important;

        font-size: 14px !important;

        font-weight: 700 !important;

        padding: 9px 18px !important;

    }


    .st-key-clear_ai_chef button:hover {

        background: #FFF0E3 !important;

        color: #D95724 !important;

        border-color: #E8CDB8 !important;

    }


    /* --------------------------------------------------------
       MOBILE
       -------------------------------------------------------- */

    @media (max-width: 768px) {

        .st-key-ai_chef_chat_box {

            padding: 25px 20px 22px 20px !important;

            border-radius: 23px !important;

        }

        .ai-chef-title {

            font-size: 26px;

        }

        .ai-chef-user-message,

        .ai-chef-ai-message {

            font-size: 14px;

        }

        .ai-chef-ai-content {

            font-size: 14px;

        }

    }


    .footer {
        text-align: center;
        color: #8A7D74;
        margin-top: 50px;
        font-size: 14px;
    }


    @media (max-width: 768px) {

        .block-container {
            padding-top: 2rem;
        }

        .hero-title {
            font-size: 42px;
        }

        .hero-subtitle {
            font-size: 19px;
        }

        .section-title {
            font-size: 25px;
        }

        .feature-card {
            min-height: 145px;
        }

        .nutrition-grid {
            grid-template-columns: repeat(2, 1fr);
        }

    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD FOOD MODEL
# ============================================================

@st.cache_resource
def load_food_model():

    device = torch.device("cpu")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    classes = checkpoint["classes"]

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, classes, device


# ============================================================
# LOAD RECIPE DATABASE
# ============================================================

@st.cache_data
def load_recipes():

    with open(
        RECIPE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        recipes = json.load(file)

    return recipes


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ============================================================
# FORMAT FOOD NAME
# ============================================================

def format_food_name(food_name):

    return (
        str(food_name)
        .replace("_", " ")
        .title()
    )


# ============================================================
# FOOD PREDICTION
# ============================================================

def predict_food(
    image,
    model,
    classes,
    device
):

    image = image.convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    predicted_food = classes[predicted.item()]

    confidence_percentage = (
        confidence.item() * 100
    )

    return (
        predicted_food,
        confidence_percentage
    )


# ============================================================
# FIND RECIPES
# ============================================================

def get_matching_recipes(
    predicted_food,
    recipes
):

    predicted_food = (
        predicted_food
        .lower()
        .strip()
    )

    matching_recipes = []

    for recipe in recipes:

        recipe_food = (
            recipe["food"]
            .lower()
            .strip()
        )

        if recipe_food == predicted_food:

            matching_recipes.append(recipe)

    return matching_recipes


# ============================================================
# DISPLAY RECIPE
# ============================================================

def display_recipe(recipe):

    recipe_name = html.escape(
        str(recipe["recipe_name"])
    )

    servings = html.escape(
        str(recipe["servings"])
    )

    calories = html.escape(
        str(recipe["calories"])
    )

    protein = html.escape(
        str(recipe["protein"])
    )

    carbs = html.escape(
        str(recipe["carbs"])
    )

    fat = html.escape(
        str(recipe["fat"])
    )


    ingredients_html = ""

    for ingredient in recipe["ingredients"]:

        safe_ingredient = html.escape(
            str(ingredient)
        )

        ingredients_html += f"""
        <div class="ingredient-item">
            • {safe_ingredient}
        </div>
        """


    instructions_html = ""

    for index, instruction in enumerate(
        recipe["instructions"],
        start=1
    ):

        safe_instruction = html.escape(
            str(instruction)
        )

        instructions_html += f"""
        <div class="instruction-item">

            <span class="instruction-number">
                {index}.
            </span>

            {safe_instruction}

        </div>
        """


    recipe_html = f"""
    <div class="recipe-card">

        <div class="recipe-title">
            🍳 {recipe_name}
        </div>


        <div class="nutrition-grid">

            <div class="nutrition-box">

                <div class="nutrition-number">
                    {servings}
                </div>

                <div class="nutrition-label">
                    Servings
                </div>

            </div>


            <div class="nutrition-box">

                <div class="nutrition-number">
                    {calories}
                </div>

                <div class="nutrition-label">
                    Calories
                </div>

            </div>


            <div class="nutrition-box">

                <div class="nutrition-number">
                    {protein}g
                </div>

                <div class="nutrition-label">
                    Protein
                </div>

            </div>


            <div class="nutrition-box">

                <div class="nutrition-number">
                    {carbs}g
                </div>

                <div class="nutrition-label">
                    Carbs
                </div>

            </div>


            <div class="nutrition-box">

                <div class="nutrition-number">
                    {fat}g
                </div>

                <div class="nutrition-label">
                    Fat
                </div>

            </div>

        </div>


        <div class="recipe-heading">
            🛒 Ingredients
        </div>

        {ingredients_html}


        <div class="recipe-heading">
            👨‍🍳 Instructions
        </div>

        {instructions_html}

    </div>
    """


    st.html(recipe_html)


# ============================================================
# AI CHEF — GEMINI
# ============================================================

def ask_ai_chef(
    question,
    predicted_food=None,
    recipe=None
):

    """
    Generate a fast, natural AI Chef response for any food-related question.

    FoodLens context is optional: the user can chat with AI Chef even when
    they have not uploaded an image or selected a recipe.
    """

    food_context = (
        format_food_name(predicted_food)
        if predicted_food
        else "No food has been identified yet."
    )

    recipe_context = "No recipe has been selected yet."

    if recipe:

        ingredients = recipe.get("ingredients", [])
        instructions = recipe.get("instructions", [])

        recipe_context = f"""
Recipe name: {recipe.get("recipe_name", "Unknown")}
Servings: {recipe.get("servings", "Unknown")}
Calories: {recipe.get("calories", "Unknown")}
Protein: {recipe.get("protein", "Unknown")}g
Carbs: {recipe.get("carbs", "Unknown")}g
Fat: {recipe.get("fat", "Unknown")}g

Ingredients:
{chr(10).join("- " + str(x) for x in ingredients)}

Instructions:
{chr(10).join(
    f"{i}. {str(x)}"
    for i, x in enumerate(instructions, 1)
)}
"""

    # Keep the latest turns so follow-up questions still feel conversational
    # without making every request unnecessarily large.
    recent_messages = st.session_state.ai_chef_messages[-6:]

    conversation_history = "\n".join(
        (
            "User" if message["role"] == "user" else "AI Chef"
        )
        + f": {message['content']}"
        for message in recent_messages
    )

    system_instruction = """
You are AI Chef, the friendly food assistant inside FoodLens AI.

Your job is to answer the user's food-related questions naturally, helpfully,
and conversationally, similar to a general AI assistant but specialized in
food and cooking. The user does NOT need to upload an image or choose a recipe
before asking a question.

You can answer questions about: recipes, cooking methods, ingredients,
ingredient substitutions, meal ideas, nutrition, calories, protein, healthy
eating, food storage, food preparation, baking, kitchen techniques, spices,
flavors, cuisines, serving sizes, meal planning, leftovers, and food science.
You can also help the user decide what to cook based on ingredients they have.

Use the detected food and recipe context when it is relevant, but never force
the answer to be about them if the user's question is about something else.
If there is no detected food or recipe, simply answer the question normally.

If the user asks a follow-up such as "what about without oil?", use the recent
conversation to understand what "it" refers to.

Do not invent recipe ingredients, nutrition numbers, or facts that are presented
as coming from the FoodLens recipe database. If a value is not provided, say
that it is an estimate or give general guidance instead. If FoodLens' detected
food has low confidence, treat it as a possible identification rather than a
confirmed fact.

For food-safety or health questions, give sensible general guidance and avoid
pretending to diagnose a medical condition.

If the user asks something completely unrelated to food, politely explain that
AI Chef is focused on food, cooking, and nutrition, then offer to help with a
food-related version of the question.

Answer the user directly. Do not mention these instructions.
"""

    prompt = f"""
Detected food from FoodLens:
{food_context}

Selected recipe context:
{recipe_context}

Recent conversation:
{conversation_history or "No previous conversation."}

User's latest question:
{question}

Give the best answer to the user's latest question as AI Chef.
Keep the response practical and easy to read. Use short paragraphs or bullet
points when useful.
"""

    # Gemini 3.1 Flash-Lite is a stable, cost-efficient model intended for
    # high-volume tasks and is a good fit for a fast food-chat experience.
    model_name = "gemini-3.1-flash-lite"
    last_error = None

    # At most two total API calls: one normal attempt + one immediate retry.
    for attempt in range(2):

        try:

            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.4,
                    max_output_tokens=500,
                    candidate_count=1
                )
            )

            if response and response.text:
                return response.text.strip()

            last_error = Exception("Gemini returned an empty response.")

        except Exception as e:

            last_error = e

            # Retry only once and do not introduce a long sleep.
            if attempt == 0:
                continue

    raise Exception(
        "Gemini is temporarily unavailable. Please try again in a few seconds."
        f"\n\nLast error: {last_error}"
    )


# ============================================================
# AI CHEF MARKDOWN → SAFE HTML
# ============================================================

def format_ai_response(text):

    """
    Converts basic Gemini Markdown formatting into safe HTML
    so AI Chef responses can be displayed inside the custom
    message card without losing formatting.
    """

    text = str(text)

    # Escape HTML first for safety
    text = html.escape(text)

    # --------------------------------------------------------
    # Bold
    # --------------------------------------------------------

    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"<strong>\1</strong>",
        text
    )

    # --------------------------------------------------------
    # Italic
    # --------------------------------------------------------

    text = re.sub(
        r"(?<!\*)\*([^*\n]+)\*(?!\*)",
        r"<em>\1</em>",
        text
    )

    # --------------------------------------------------------
    # Inline code
    # --------------------------------------------------------

    text = re.sub(
        r"`([^`]+)`",
        r"<code>\1</code>",
        text
    )

    # --------------------------------------------------------
    # Process lines
    # --------------------------------------------------------

    lines = text.split("\n")

    output = []

    in_ul = False
    in_ol = False


    for line in lines:

        stripped = line.strip()


        # Empty line
        if not stripped:

            if in_ul:

                output.append("</ul>")
                in_ul = False

            if in_ol:

                output.append("</ol>")
                in_ol = False

            continue


        # ----------------------------------------------------
        # Bullet list
        # ----------------------------------------------------

        bullet_match = re.match(
            r"^[-•]\s+(.+)",
            stripped
        )


        if bullet_match:

            if in_ol:

                output.append("</ol>")
                in_ol = False

            if not in_ul:

                output.append("<ul>")
                in_ul = True

            output.append(
                f"<li>{bullet_match.group(1)}</li>"
            )

            continue


        # ----------------------------------------------------
        # Numbered list
        # ----------------------------------------------------

        number_match = re.match(
            r"^\d+\.\s+(.+)",
            stripped
        )


        if number_match:

            if in_ul:

                output.append("</ul>")
                in_ul = False

            if not in_ol:

                output.append("<ol>")
                in_ol = True

            output.append(
                f"<li>{number_match.group(1)}</li>"
            )

            continue


        # ----------------------------------------------------
        # Normal paragraph
        # ----------------------------------------------------

        if in_ul:

            output.append("</ul>")
            in_ul = False

        if in_ol:

            output.append("</ol>")
            in_ol = False


        output.append(
            f"<p>{stripped}</p>"
        )


    if in_ul:
        output.append("</ul>")

    if in_ol:
        output.append("</ol>")


    return "".join(output)


# ============================================================
# PERSONALIZED GREETING
# ============================================================

current_hour = datetime.now().hour


if current_hour < 12:

    greeting = "Good morning"
    greeting_icon = "☀️"

elif current_hour < 17:

    greeting = "Good afternoon"
    greeting_icon = "🌤️"

else:

    greeting = "Good evening"
    greeting_icon = "🌙"


# ============================================================
# HERO SECTION
# ============================================================

st.html(f"""
<div class="hero-card">

    <div class="hero-title">
        🍽️ FoodLens <span>AI</span>
    </div>


    <div style="
        font-size:28px;
        font-weight:800;
        color:#2D241F;
        margin-top:10px;
        margin-bottom:8px;
    ">

        {greeting}, {html.escape(user_name)}!
        {greeting_icon}

    </div>


    <div class="hero-subtitle">
        See it. Know it. Cook it.
    </div>


    <div style="
        font-size:17px;
        color:#5F5149;
        max-width:650px;
        line-height:1.7;
    ">

        Welcome back to your personal food discovery space.
        Upload a food photo and let FoodLens AI turn
        what's on your plate into something delicious.

    </div>

</div>
""")


# ============================================================
# UPLOAD SECTION
# ============================================================

st.html("""
<div class="section-title">
    📸 What are you craving?
</div>
""")


st.html("""
<div class="section-text">
    Upload a photo and let FoodLens AI do the detective work.
</div>
""")


uploaded_file = st.file_uploader(
    "Upload food image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# IMAGE DISPLAY + ANALYSIS
# ============================================================

if uploaded_file is not None:

    st.markdown("### ✨ Your food")

    col1, col2 = st.columns([1.1, 1])


    with col1:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            width=500
        )


    with col2:

        st.html("""
        <div class="result-card">

            <div class="result-label">
                Ready for analysis
            </div>


            <div style="
                font-size:30px;
                font-weight:800;
                color:#2D241F;
                margin-top:8px;
            ">

                🔍 Identify this food

            </div>


            <p style="
                color:#766A62;
                line-height:1.6;
            ">

                FoodLens AI will analyze your image using
                our trained ResNet50 deep-learning model.

            </p>

        </div>
        """)


        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        analyze_button = st.button(
            "🔍 Analyze Food"
        )


        if analyze_button:

            with st.spinner(
                "🧠 FoodLens AI is analyzing your image..."
            ):

                try:

                    model, classes, device = (
                        load_food_model()
                    )


                    predicted_food, confidence = (
                        predict_food(
                            image,
                            model,
                            classes,
                            device
                        )
                    )


                    recipes = load_recipes()


                    matching_recipes = (
                        get_matching_recipes(
                            predicted_food,
                            recipes
                        )
                    )


                    st.session_state["predicted_food"] = (
                        predicted_food
                    )

                    st.session_state["confidence"] = (
                        confidence
                    )

                    st.session_state["matching_recipes"] = (
                        matching_recipes
                    )


                except Exception as e:

                    st.error(
                        "Something went wrong while "
                        f"analyzing the image: {e}"
                    )


# ============================================================
# DISPLAY AI RESULT
# ============================================================

if "predicted_food" in st.session_state:

    predicted_food = (
        st.session_state["predicted_food"]
    )

    confidence = (
        st.session_state["confidence"]
    )

    matching_recipes = (
        st.session_state["matching_recipes"]
    )


    display_food_name = format_food_name(
        predicted_food
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    if confidence >= HIGH_CONFIDENCE:

        st.success(
            "Food identified with high confidence! 🎉"
        )

        result_label = "🧠 FoodLens AI identified"


    elif confidence >= MEDIUM_CONFIDENCE:

        st.html("""
        <div style="
            background:#FFF3E8;
            border:1px solid #F3C9A8;
            border-radius:12px;
            padding:12px 16px;
            color:#000000;
            font-weight:700;
            font-size:15px;
        ">

            FoodLens AI found a possible match.
            The prediction may not be exact. ⚠️

        </div>
        """)

        result_label = "🧠 FoodLens AI's possible match"


    else:

        result_label = "🧠 FoodLens AI's best guess"


    st.html(
        f"""
        <div class="result-card">

            <div class="result-label">
                {html.escape(result_label)}
            </div>


            <div class="result-food">
                {html.escape(display_food_name)}
            </div>


            <div class="confidence-label">
                Model confidence
            </div>


            <div class="confidence-value">
                {confidence:.2f}%
            </div>

        </div>
        """
    )


    if confidence < MEDIUM_CONFIDENCE:

        st.html(
            f"""
            <div class="low-confidence-card">

                <div class="low-confidence-title">
                    ⚠️ FoodLens AI couldn't confidently
                    identify this food.
                </div>


                <div class="low-confidence-text">

                    <b>{html.escape(display_food_name)}</b>
                    is the model's best guess, but the
                    confidence is only
                    <b>{confidence:.2f}%</b>.

                    The uploaded food may not belong to
                    one of the 101 categories supported
                    by the model.

                    Try a clearer image or a supported
                    food category.

                </div>

            </div>
            """
        )


    st.markdown(
        "<br><br>",
        unsafe_allow_html=True
    )


    # ========================================================
    # HIGH CONFIDENCE RECIPES
    # ========================================================

    if confidence >= HIGH_CONFIDENCE:

        st.html("""
        <div class="section-title">
            🍳 Recommended Recipes
        </div>
        """)


        st.html("""
        <div class="section-text">
            Delicious recipes based on the food
            FoodLens AI identified.
        </div>
        """)


        if matching_recipes:

            for recipe in matching_recipes:

                display_recipe(recipe)

        else:

            st.info(
                f"We don't have recipes for "
                f"'{display_food_name}' yet. "
                "More recipes are coming soon! 🍽️"
            )


    # ========================================================
    # MEDIUM CONFIDENCE RECIPES
    # ========================================================

    elif confidence >= MEDIUM_CONFIDENCE:

        st.html("""
        <div class="section-title">
            🍳 Possible Recipe Matches
        </div>
        """)


        st.html(
            f"""
            <div class="section-text">

                These recipes are based on FoodLens AI's
                possible match:
                <b>{html.escape(display_food_name)}</b>.

                Please verify the prediction before cooking.

            </div>
            """
        )


        if matching_recipes:

            for recipe in matching_recipes:

                display_recipe(recipe)

        else:

            st.info(
                f"We don't have recipes for "
                f"'{display_food_name}' yet. "
                "More recipes are coming soon! 🍽️"
            )


    # ========================================================
    # LOW CONFIDENCE RECIPES
    # ========================================================

    else:

        st.html("""
        <div class="section-title">
            🍳 Possible Recipe
        </div>
        """)


        st.html("""
        <div class="section-text">

            Because the prediction is uncertain,
            FoodLens AI won't automatically recommend
            this recipe.

            You can still preview it if you think
            the prediction looks reasonable.

        </div>
        """)


        if matching_recipes:

            for recipe in matching_recipes:

                recipe_name = html.escape(
                    str(recipe["recipe_name"])
                )


                with st.expander(
                    f"🍳 View possible recipe: {recipe_name}",
                    expanded=False
                ):

                    st.html("""
                    <div class="possible-recipe-note">

                        ⚠️ This recipe is based on the
                        model's uncertain prediction.

                        Please verify the food before
                        using this recipe.

                    </div>
                    """)


                    display_recipe(recipe)

        else:

            st.info(
                f"No recipe is available for the model's "
                f"best guess '{display_food_name}' yet."
            )


# ============================================================
# FEATURES
# ============================================================

st.markdown(
    "<br><br>",
    unsafe_allow_html=True
)


st.html("""
<div class="section-title">
    🍴 From photo to plate
</div>
""")


st.html("""
<div class="section-text">

    Everything you need to turn food inspiration
    into your next meal.

</div>
""")


col1, col2, col3 = st.columns(3)


with col1:

    st.html("""
    <div class="feature-card feature-card-1">

        <div class="feature-icon">
            🧠
        </div>

        <div class="feature-title">
            AI Recognition
        </div>

        <div class="feature-text">
            Identify food using our trained
            deep-learning model.
        </div>

    </div>
    """)


with col2:

    st.html("""
    <div class="feature-card feature-card-2">

        <div class="feature-icon">
            🍳
        </div>

        <div class="feature-title">
            Recipe Discovery
        </div>

        <div class="feature-text">
            Find delicious recipes based on
            what you upload.
        </div>

    </div>
    """)


with col3:

    st.html("""
    <div class="feature-card feature-card-3">

        <div class="feature-icon">
            👨‍🍳
        </div>

        <div class="feature-title">
            AI Chef
        </div>

        <div class="feature-text">
            Get cooking help, substitutions
            and serving advice.
        </div>

    </div>
    """)


# ============================================================
# AI CHEF CHAT — ONE SINGLE FLOATING BOX
# ============================================================

st.markdown(
    "<br><br>",
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# CURRENT FOOD / RECIPE CONTEXT
# ------------------------------------------------------------

ai_food = st.session_state.get(
    "predicted_food",
    None
)


ai_recipes = st.session_state.get(
    "matching_recipes",
    []
)


ai_recipe = (
    ai_recipes[0]
    if ai_recipes
    else None
)


# ============================================================
# COMPLETE AI CHEF CONTAINER
# ============================================================

with st.container(
    border=True,
    key="ai_chef_chat_box"
):

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.html("""
    <div class="ai-chef-header">

        <div class="ai-chef-title">
            👨‍🍳 AI Chef
        </div>

        <div class="ai-chef-subtitle">
            Ask your AI Chef anything about food, cooking, recipes,
            ingredients, nutrition or meal ideas.
        </div>

    </div>
    """)


    # --------------------------------------------------------
    # FOOD CONTEXT
    # --------------------------------------------------------

    if ai_food:

        st.html(
            f"""
            <div class="ai-chef-ready">

                🍽️ <strong>AI Chef is ready</strong>

                <br>

                Your detected food:
                <strong>
                    {html.escape(format_food_name(ai_food))}
                </strong>

            </div>
            """
        )

    else:

        st.html("""
        <div class="ai-chef-info">

            💡 No image is required. Ask AI Chef anything about food,
            cooking, recipes, ingredients, nutrition or meal ideas.

        </div>
        """)


    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    if st.session_state.ai_chef_messages:

        st.html("""
        <div class="ai-chef-history">
        """)


        for message in st.session_state.ai_chef_messages:

            message_content = str(
                message["content"]
            )


            if message["role"] == "user":

                safe_content = html.escape(
                    message_content
                ).replace(
                    "\n",
                    "<br>"
                )


                st.html(
                    f"""
                    <div class="ai-chef-user-message">

                        <span class="ai-chef-user-label">
                            👤 You:
                        </span>

                        {safe_content}

                    </div>
                    """
                )


            else:

                formatted_answer = format_ai_response(
                    message_content
                )


                st.html(
                    f"""
                    <div class="ai-chef-ai-message">

                        <span class="ai-chef-ai-label">
                            🤖 AI Chef
                        </span>

                        <div class="ai-chef-ai-content">

                            {formatted_answer}

                        </div>

                    </div>
                    """
                )


        st.html("""
        </div>
        """)


    # ========================================================
    # AI CHEF INPUT FORM
    # ========================================================

    st.html("""
    <div class="ai-chef-input-label">
        Ask your AI Chef anything 🍳
    </div>
    """)


    # --------------------------------------------------------
    # IMPORTANT:
    # clear_on_submit=True safely clears the widget after
    # submission.
    #
    # We NEVER modify:
    # st.session_state.ai_chef_input
    #
    # This permanently fixes the widget-state error.
    # --------------------------------------------------------

    with st.form(
        "ai_chef_form",
        clear_on_submit=True,
        border=False
    ):

        input_col, send_col = st.columns(
            [5, 1],
            vertical_alignment="bottom"
        )


        with input_col:

            chef_question = st.text_input(
                "Ask AI Chef something...",
                placeholder=(
                    "Ask anything about food, cooking, recipes or nutrition..."
                ),
                label_visibility="collapsed",
                key="ai_chef_input"
            )


        with send_col:

            send_question = st.form_submit_button(
                "Send",
                key="ai_chef_send",
                width="stretch"
            )


    # ========================================================
    # PROCESS QUESTION
    # ========================================================

    if send_question:

        clean_question = chef_question.strip()


        if not clean_question:

            st.warning(
                "Please enter a question for AI Chef."
            )


        else:

            # ------------------------------------------------
            # Add user message
            # ------------------------------------------------

            st.session_state.ai_chef_messages.append(
                {
                    "role": "user",
                    "content": clean_question
                }
            )


            # ------------------------------------------------
            # Generate response
            # ------------------------------------------------

            with st.spinner(
                "👨‍🍳 AI Chef is thinking..."
            ):

                try:

                    answer = ask_ai_chef(
                        clean_question,
                        ai_food,
                        ai_recipe
                    )


                    st.session_state.ai_chef_messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )


                    # ------------------------------------------------
                    # IMPORTANT:
                    #
                    # DO NOT DO THIS:
                    #
                    # st.session_state.ai_chef_input = ""
                    #
                    # clear_on_submit=True already clears the widget.
                    # ------------------------------------------------

                    st.rerun()


                except Exception:

                    # Remove failed user question

                    if st.session_state.ai_chef_messages:

                        last_message = (
                            st.session_state.ai_chef_messages[-1]
                        )


                        if last_message["role"] == "user":

                            st.session_state.ai_chef_messages.pop()


                    st.error(
                        "❌ AI Chef could not respond right now."
                    )

                    st.caption(
                        "Please try again in a few seconds."
                    )


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.session_state.ai_chef_messages:

        st.markdown(
            "<div style='height:4px;'></div>",
            unsafe_allow_html=True
        )


        if st.button(
            "🗑️ Clear AI Chef Chat",
            key="clear_ai_chef",
            width="stretch"
        ):

            st.session_state.ai_chef_messages = []

            st.rerun()


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    "<br><br>",
    unsafe_allow_html=True
)


st.html("""
<div class="section-title">
    ⚙️ How FoodLens AI works
</div>
""")


st.html("""
<div style="
    background:#2D241F;
    color:white;
    border-radius:24px;
    padding:35px;
    margin-top:15px;
">

    <div style="
        display:flex;
        justify-content:space-between;
        text-align:center;
    ">

        <div>

            <div style="font-size:35px;">
                📸
            </div>

            <b>1. Upload</b>

            <p style="color:#D8CBC2;">
                Choose a food image
            </p>

        </div>


        <div style="font-size:30px;">
            →
        </div>


        <div>

            <div style="font-size:35px;">
                🧠
            </div>

            <b>2. Analyze</b>

            <p style="color:#D8CBC2;">
                AI identifies the food
            </p>

        </div>


        <div style="font-size:30px;">
            →
        </div>


        <div>

            <div style="font-size:35px;">
                🍳
            </div>

            <b>3. Discover</b>

            <p style="color:#D8CBC2;">
                Explore recipes
            </p>

        </div>


        <div style="font-size:30px;">
            →
        </div>


        <div>

            <div style="font-size:35px;">
                👨‍🍳
            </div>

            <b>4. Cook</b>

            <p style="color:#D8CBC2;">
                Follow & customize
            </p>

        </div>

    </div>

</div>
""")


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    🍽️ FoodLens AI
    &nbsp;•&nbsp;
    See it. Know it. Cook it.

</div>
""")