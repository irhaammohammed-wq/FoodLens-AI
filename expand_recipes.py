import json
import torch
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "food101_resnet50.pth"
RECIPE_PATH = BASE_DIR / "recipes" / "recipes.json"


# ============================================================
# LOAD MODEL CLASSES
# ============================================================

print("=" * 65)
print("FOODLENS AI - MASTER RECIPE EXPANSION")
print("=" * 65)

print("\nLoading Food-101 model classes...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

model_classes = checkpoint["classes"]

print(f"Model classes: {len(model_classes)}")


# ============================================================
# LOAD EXISTING RECIPES
# ============================================================

print("\nLoading existing recipes...")

if RECIPE_PATH.exists():
    with open(RECIPE_PATH, "r", encoding="utf-8") as f:
        recipes = json.load(f)
else:
    recipes = []

print(f"Existing recipes: {len(recipes)}")


# ============================================================
# MASTER RECIPE DATABASE
# ============================================================

new_recipes = [

    # --------------------------------------------------------
    # 1. BABY BACK RIBS
    # --------------------------------------------------------

    {
        "food": "baby_back_ribs",
        "recipe_name": "Classic BBQ Baby Back Ribs",
        "servings": 4,
        "calories": 620,
        "protein": 42,
        "carbs": 28,
        "fat": 36,
        "ingredients": [
            "1.5 kg baby back ribs",
            "2 tbsp brown sugar",
            "1 tbsp paprika",
            "1 tsp garlic powder",
            "1 tsp onion powder",
            "1 tsp black pepper",
            "1 tsp salt",
            "1 cup BBQ sauce"
        ],
        "instructions": [
            "Remove the membrane from the back of the ribs.",
            "Mix brown sugar, paprika, garlic powder, onion powder, pepper and salt.",
            "Rub the seasoning evenly over the ribs.",
            "Bake covered at 150°C for about 2 hours.",
            "Brush generously with BBQ sauce.",
            "Finish uncovered until the sauce becomes sticky and caramelized.",
            "Rest for 10 minutes before serving."
        ]
    },

    # --------------------------------------------------------
    # 2. BEET SALAD
    # --------------------------------------------------------

    {
        "food": "beet_salad",
        "recipe_name": "Roasted Beet Salad",
        "servings": 2,
        "calories": 280,
        "protein": 7,
        "carbs": 30,
        "fat": 15,
        "ingredients": [
            "3 medium beets",
            "2 cups mixed greens",
            "50 g goat cheese",
            "2 tbsp walnuts",
            "1 tbsp olive oil",
            "1 tbsp lemon juice",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Roast the beets until tender.",
            "Cool, peel and slice the beets.",
            "Arrange mixed greens on plates.",
            "Add sliced beets, goat cheese and walnuts.",
            "Drizzle with olive oil and lemon juice.",
            "Season with salt and pepper.",
            "Serve immediately."
        ]
    },

    # --------------------------------------------------------
    # 3. BEIGNETS
    # --------------------------------------------------------

    {
        "food": "beignets",
        "recipe_name": "Classic New Orleans Beignets",
        "servings": 6,
        "calories": 320,
        "protein": 6,
        "carbs": 45,
        "fat": 14,
        "ingredients": [
            "2 cups all-purpose flour",
            "2 tbsp sugar",
            "1 tsp instant yeast",
            "3/4 cup warm milk",
            "1 egg",
            "2 tbsp butter",
            "Pinch of salt",
            "Powdered sugar"
        ],
        "instructions": [
            "Combine flour, sugar, yeast and salt.",
            "Add warm milk, egg and melted butter.",
            "Knead into a soft dough.",
            "Let the dough rise until doubled.",
            "Roll and cut into rectangles.",
            "Fry until golden brown.",
            "Drain and dust generously with powdered sugar."
        ]
    },

    # --------------------------------------------------------
    # 4. BREAD PUDDING
    # --------------------------------------------------------

    {
        "food": "bread_pudding",
        "recipe_name": "Classic Bread Pudding",
        "servings": 6,
        "calories": 390,
        "protein": 9,
        "carbs": 54,
        "fat": 16,
        "ingredients": [
            "6 cups cubed bread",
            "2 cups milk",
            "3 eggs",
            "1/2 cup sugar",
            "1 tsp vanilla",
            "1 tsp cinnamon",
            "2 tbsp butter",
            "1/2 cup raisins"
        ],
        "instructions": [
            "Place bread cubes in a baking dish.",
            "Whisk milk, eggs, sugar, vanilla and cinnamon.",
            "Pour the mixture over the bread.",
            "Add raisins and let the bread soak.",
            "Dot the top with butter.",
            "Bake at 175°C until golden and set.",
            "Cool slightly before serving."
        ]
    },

    # --------------------------------------------------------
    # 5. BREAKFAST BURRITO
    # --------------------------------------------------------

    {
        "food": "breakfast_burrito",
        "recipe_name": "Loaded Breakfast Burrito",
        "servings": 2,
        "calories": 520,
        "protein": 27,
        "carbs": 48,
        "fat": 25,
        "ingredients": [
            "2 large flour tortillas",
            "4 eggs",
            "1/2 cup cooked potatoes",
            "1/2 cup shredded cheese",
            "1/4 cup diced onion",
            "1/4 cup diced tomato",
            "2 tbsp salsa",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Scramble the eggs with salt and pepper.",
            "Warm the potatoes in a skillet.",
            "Place tortillas on a flat surface.",
            "Add eggs, potatoes, cheese, onion and tomato.",
            "Top with salsa.",
            "Fold the sides inward and roll tightly.",
            "Toast the burritos lightly before serving."
        ]
    },

    # --------------------------------------------------------
    # 6. CHEESE PLATE
    # --------------------------------------------------------

    {
        "food": "cheese_plate",
        "recipe_name": "Artisan Cheese Plate",
        "servings": 4,
        "calories": 430,
        "protein": 20,
        "carbs": 20,
        "fat": 32,
        "ingredients": [
            "100 g cheddar cheese",
            "100 g brie",
            "100 g gouda",
            "1/2 cup grapes",
            "1/4 cup walnuts",
            "2 tbsp honey",
            "Crackers"
        ],
        "instructions": [
            "Slice the cheeses into bite-sized pieces.",
            "Arrange the cheeses on a serving board.",
            "Add grapes and walnuts.",
            "Place crackers around the cheese.",
            "Drizzle honey over the board.",
            "Serve at room temperature."
        ]
    },

    # --------------------------------------------------------
    # 7. CLAM CHOWDER
    # --------------------------------------------------------

    {
        "food": "clam_chowder",
        "recipe_name": "Creamy Clam Chowder",
        "servings": 4,
        "calories": 390,
        "protein": 19,
        "carbs": 32,
        "fat": 21,
        "ingredients": [
            "400 g clams",
            "2 potatoes",
            "1 onion",
            "2 celery stalks",
            "2 cups milk",
            "1 cup cream",
            "2 tbsp butter",
            "1 tbsp flour",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Melt butter and sauté onion and celery.",
            "Add flour and stir until smooth.",
            "Gradually add milk and cream.",
            "Add diced potatoes and cook until tender.",
            "Add cleaned clams and simmer until cooked.",
            "Season with salt and pepper.",
            "Serve hot."
        ]
    },

    # --------------------------------------------------------
    # 8. CREME BRULEE
    # --------------------------------------------------------

    {
        "food": "creme_brulee",
        "recipe_name": "Classic Crème Brûlée",
        "servings": 4,
        "calories": 380,
        "protein": 6,
        "carbs": 28,
        "fat": 27,
        "ingredients": [
            "2 cups heavy cream",
            "5 egg yolks",
            "1/3 cup sugar",
            "1 tsp vanilla",
            "4 tbsp sugar for topping"
        ],
        "instructions": [
            "Heat cream with vanilla until warm.",
            "Whisk egg yolks with sugar.",
            "Slowly whisk the warm cream into the yolks.",
            "Pour into ramekins.",
            "Bake in a water bath until just set.",
            "Cool completely.",
            "Sprinkle sugar on top and caramelize carefully before serving."
        ]
    },

    # --------------------------------------------------------
    # 9. CRAB CAKES
    # --------------------------------------------------------

    {
        "food": "crab_cakes",
        "recipe_name": "Golden Crab Cakes",
        "servings": 4,
        "calories": 310,
        "protein": 25,
        "carbs": 18,
        "fat": 15,
        "ingredients": [
            "400 g crab meat",
            "1 egg",
            "1/2 cup breadcrumbs",
            "2 tbsp mayonnaise",
            "1 tsp mustard",
            "1 tbsp lemon juice",
            "2 tbsp chopped parsley",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Combine crab meat, egg, breadcrumbs, mayonnaise and mustard.",
            "Add lemon juice and parsley.",
            "Season with salt and pepper.",
            "Form into small patties.",
            "Chill the patties briefly.",
            "Pan-sear until golden and cooked through.",
            "Serve with lemon."
        ]
    },

    # --------------------------------------------------------
    # 10. CROQUE MADAME
    # --------------------------------------------------------

    {
        "food": "croque_madame",
        "recipe_name": "Classic Croque Madame",
        "servings": 2,
        "calories": 610,
        "protein": 32,
        "carbs": 40,
        "fat": 37,
        "ingredients": [
            "4 slices bread",
            "4 slices ham",
            "1 cup grated Gruyère cheese",
            "1 cup béchamel sauce",
            "2 eggs",
            "1 tbsp butter",
            "Black pepper"
        ],
        "instructions": [
            "Spread béchamel over the bread.",
            "Add ham and cheese between the slices.",
            "Toast until the cheese melts.",
            "Top with more béchamel and cheese.",
            "Bake until bubbling and golden.",
            "Fry the eggs separately.",
            "Place one egg on each sandwich before serving."
        ]
    },

    # --------------------------------------------------------
    # 11. CUP CAKES
    # --------------------------------------------------------

    {
        "food": "cup_cakes",
        "recipe_name": "Classic Vanilla Cupcakes",
        "servings": 8,
        "calories": 290,
        "protein": 4,
        "carbs": 39,
        "fat": 13,
        "ingredients": [
            "1 cup flour",
            "1/2 cup sugar",
            "1/2 cup butter",
            "2 eggs",
            "1 tsp vanilla",
            "1 tsp baking powder",
            "1/3 cup milk"
        ],
        "instructions": [
            "Cream butter and sugar together.",
            "Beat in the eggs and vanilla.",
            "Add flour and baking powder.",
            "Mix in milk until smooth.",
            "Divide batter into cupcake liners.",
            "Bake at 175°C until golden.",
            "Cool and decorate as desired."
        ]
    },

    # --------------------------------------------------------
    # 12. DEVILED EGGS
    # --------------------------------------------------------

    {
        "food": "deviled_eggs",
        "recipe_name": "Classic Deviled Eggs",
        "servings": 4,
        "calories": 210,
        "protein": 12,
        "carbs": 3,
        "fat": 17,
        "ingredients": [
            "6 eggs",
            "3 tbsp mayonnaise",
            "1 tsp mustard",
            "1 tsp lemon juice",
            "Paprika",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Boil the eggs until fully cooked.",
            "Cool and peel the eggs.",
            "Slice each egg in half.",
            "Remove the yolks and mash them.",
            "Mix yolks with mayonnaise, mustard and lemon juice.",
            "Season with salt and pepper.",
            "Pipe the filling into the egg whites.",
            "Sprinkle with paprika."
        ]
    },

    # --------------------------------------------------------
    # 13. EDAMAME
    # --------------------------------------------------------

    {
        "food": "edamame",
        "recipe_name": "Garlic Chili Edamame",
        "servings": 2,
        "calories": 190,
        "protein": 17,
        "carbs": 14,
        "fat": 8,
        "ingredients": [
            "300 g edamame",
            "2 cloves garlic",
            "1 tsp chili flakes",
            "1 tsp sesame oil",
            "1 tsp soy sauce"
        ],
        "instructions": [
            "Steam or boil the edamame until tender.",
            "Heat sesame oil in a pan.",
            "Sauté garlic briefly.",
            "Add chili flakes and soy sauce.",
            "Toss in the edamame.",
            "Cook for 1–2 minutes.",
            "Serve warm."
        ]
    },

    # --------------------------------------------------------
    # 14. EGGS BENEDICT
    # --------------------------------------------------------

    {
        "food": "eggs_benedict",
        "recipe_name": "Classic Eggs Benedict",
        "servings": 2,
        "calories": 480,
        "protein": 24,
        "carbs": 27,
        "fat": 30,
        "ingredients": [
            "2 English muffins",
            "4 eggs",
            "4 slices ham",
            "2 egg yolks",
            "100 g butter",
            "1 tbsp lemon juice",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Toast the English muffins.",
            "Warm the ham.",
            "Prepare poached eggs until the whites are set.",
            "Whisk egg yolks with lemon juice.",
            "Slowly whisk in melted butter to make hollandaise.",
            "Place ham on the muffins.",
            "Top with poached eggs and hollandaise.",
            "Season with pepper."
        ]
    },

    # --------------------------------------------------------
    # 15. ESCARGOTS
    # --------------------------------------------------------

    {
        "food": "escargots",
        "recipe_name": "Garlic Butter Escargots",
        "servings": 2,
        "calories": 290,
        "protein": 18,
        "carbs": 5,
        "fat": 22,
        "ingredients": [
            "200 g cleaned escargots",
            "4 tbsp butter",
            "3 cloves garlic",
            "2 tbsp parsley",
            "1 tbsp lemon juice",
            "Black pepper"
        ],
        "instructions": [
            "Melt butter in a skillet.",
            "Add minced garlic and cook briefly.",
            "Add the cleaned escargots.",
            "Cook thoroughly until heated through.",
            "Add parsley and lemon juice.",
            "Season with black pepper.",
            "Serve immediately."
        ]
    },

    # --------------------------------------------------------
    # 16. FILET MIGNON
    # --------------------------------------------------------

    {
        "food": "filet_mignon",
        "recipe_name": "Garlic Butter Filet Mignon",
        "servings": 2,
        "calories": 520,
        "protein": 48,
        "carbs": 2,
        "fat": 34,
        "ingredients": [
            "2 filet mignon steaks",
            "1 tbsp olive oil",
            "2 tbsp butter",
            "2 cloves garlic",
            "1 sprig rosemary",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Season the steaks generously with salt and pepper.",
            "Heat olive oil in a heavy skillet.",
            "Sear the steaks on both sides.",
            "Add butter, garlic and rosemary.",
            "Baste the steaks with the butter.",
            "Cook to your preferred safe doneness using a thermometer.",
            "Rest before serving."
        ]
    },

    # --------------------------------------------------------
    # 17. FOIE GRAS
    # --------------------------------------------------------

    {
        "food": "foie_gras",
        "recipe_name": "Seared Foie Gras",
        "servings": 2,
        "calories": 460,
        "protein": 10,
        "carbs": 12,
        "fat": 40,
        "ingredients": [
            "200 g foie gras",
            "1 tbsp sugar",
            "1 apple",
            "1 tbsp butter",
            "1 tbsp balsamic glaze",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Slice the foie gras into thick pieces.",
            "Season lightly with salt and pepper.",
            "Heat a dry skillet over medium-high heat.",
            "Sear briefly on both sides until golden.",
            "Cook the apple slices in butter.",
            "Arrange apple slices with the foie gras.",
            "Drizzle with balsamic glaze."
        ]
    },

    # --------------------------------------------------------
    # 18. FRENCH TOAST
    # --------------------------------------------------------

    {
        "food": "french_toast",
        "recipe_name": "Cinnamon French Toast",
        "servings": 2,
        "calories": 410,
        "protein": 13,
        "carbs": 48,
        "fat": 19,
        "ingredients": [
            "4 thick bread slices",
            "2 eggs",
            "1/2 cup milk",
            "1 tsp cinnamon",
            "1 tsp vanilla",
            "1 tbsp butter",
            "Maple syrup",
            "Fresh berries"
        ],
        "instructions": [
            "Whisk eggs, milk, cinnamon and vanilla.",
            "Dip each bread slice into the mixture.",
            "Heat butter in a skillet.",
            "Cook the bread on both sides until golden.",
            "Serve with maple syrup and berries."
        ]
    },

    # --------------------------------------------------------
    # 19. FRIED CALAMARI
    # --------------------------------------------------------

    {
        "food": "fried_calamari",
        "recipe_name": "Crispy Fried Calamari",
        "servings": 3,
        "calories": 380,
        "protein": 23,
        "carbs": 36,
        "fat": 17,
        "ingredients": [
            "500 g cleaned calamari",
            "1 cup flour",
            "1/2 tsp paprika",
            "1/2 tsp garlic powder",
            "Salt",
            "Black pepper",
            "Lemon wedges"
        ],
        "instructions": [
            "Cut calamari into rings.",
            "Pat the calamari dry.",
            "Mix flour with paprika, garlic powder, salt and pepper.",
            "Coat the calamari evenly.",
            "Fry in hot oil until golden and cooked.",
            "Drain on a rack or paper towel.",
            "Serve with lemon wedges."
        ]
    },

    # --------------------------------------------------------
    # 20. FROZEN YOGURT
    # --------------------------------------------------------

    {
        "food": "frozen_yogurt",
        "recipe_name": "Berry Frozen Yogurt",
        "servings": 4,
        "calories": 180,
        "protein": 8,
        "carbs": 28,
        "fat": 4,
        "ingredients": [
            "2 cups Greek yogurt",
            "1 cup mixed berries",
            "1/4 cup honey",
            "1 tsp vanilla"
        ],
        "instructions": [
            "Blend berries until smooth.",
            "Mix yogurt, honey and vanilla.",
            "Swirl the berry puree into the yogurt.",
            "Transfer to a freezer-safe container.",
            "Freeze until firm.",
            "Let soften slightly before serving."
        ]
    },

    # --------------------------------------------------------
    # 21. GREEK SALAD
    # --------------------------------------------------------

    {
        "food": "greek_salad",
        "recipe_name": "Fresh Greek Salad",
        "servings": 2,
        "calories": 280,
        "protein": 9,
        "carbs": 15,
        "fat": 21,
        "ingredients": [
            "2 tomatoes",
            "1 cucumber",
            "1/2 red onion",
            "1/2 cup feta cheese",
            "1/4 cup olives",
            "2 tbsp olive oil",
            "1 tbsp lemon juice",
            "Oregano",
            "Salt"
        ],
        "instructions": [
            "Chop the tomatoes and cucumber.",
            "Slice the red onion.",
            "Combine vegetables with olives.",
            "Add feta cheese.",
            "Drizzle with olive oil and lemon juice.",
            "Sprinkle with oregano.",
            "Season lightly with salt."
        ]
    },

    # --------------------------------------------------------
    # 22. GYOZA
    # --------------------------------------------------------

    {
        "food": "gyoza",
        "recipe_name": "Japanese Chicken Gyoza",
        "servings": 3,
        "calories": 340,
        "protein": 20,
        "carbs": 38,
        "fat": 13,
        "ingredients": [
            "250 g ground chicken",
            "1 cup finely chopped cabbage",
            "2 green onions",
            "1 tsp grated ginger",
            "1 tbsp soy sauce",
            "1 tsp sesame oil",
            "Gyoza wrappers"
        ],
        "instructions": [
            "Mix chicken, cabbage, green onion and ginger.",
            "Add soy sauce and sesame oil.",
            "Place filling into gyoza wrappers.",
            "Fold and seal the wrappers.",
            "Pan-fry the gyoza until the bottoms are golden.",
            "Add a small amount of water and cover.",
            "Steam until the filling is thoroughly cooked.",
            "Serve with dipping sauce."
        ]
    },

    # --------------------------------------------------------
    # 23. GUACAMOLE
    # --------------------------------------------------------

    {
        "food": "guacamole",
        "recipe_name": "Fresh Homemade Guacamole",
        "servings": 4,
        "calories": 220,
        "protein": 3,
        "carbs": 12,
        "fat": 19,
        "ingredients": [
            "3 ripe avocados",
            "1 tomato",
            "1/4 red onion",
            "1 lime",
            "2 tbsp cilantro",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Mash the avocados in a bowl.",
            "Dice the tomato and onion.",
            "Add tomato, onion and cilantro.",
            "Squeeze fresh lime juice over the mixture.",
            "Season with salt and pepper.",
            "Mix gently and serve."
        ]
    },

    # --------------------------------------------------------
    # 24. HOT AND SOUR SOUP
    # --------------------------------------------------------

    {
        "food": "hot_and_sour_soup",
        "recipe_name": "Chinese Hot and Sour Soup",
        "servings": 4,
        "calories": 190,
        "protein": 12,
        "carbs": 18,
        "fat": 7,
        "ingredients": [
            "4 cups chicken or vegetable stock",
            "150 g mushrooms",
            "100 g tofu",
            "2 tbsp soy sauce",
            "2 tbsp rice vinegar",
            "1 tsp chili paste",
            "1 egg",
            "1 tbsp cornstarch",
            "2 green onions"
        ],
        "instructions": [
            "Bring stock to a simmer.",
            "Add mushrooms and tofu.",
            "Stir in soy sauce, vinegar and chili paste.",
            "Mix cornstarch with water and add to the soup.",
            "Slowly pour in the beaten egg while stirring.",
            "Simmer until slightly thickened.",
            "Garnish with green onions."
        ]
    },

    # --------------------------------------------------------
    # 25. HOT DOG
    # --------------------------------------------------------

    {
        "food": "hot_dog",
        "recipe_name": "Classic Loaded Hot Dog",
        "servings": 2,
        "calories": 390,
        "protein": 17,
        "carbs": 34,
        "fat": 21,
        "ingredients": [
            "2 hot dog sausages",
            "2 hot dog buns",
            "2 tbsp mustard",
            "2 tbsp ketchup",
            "1/4 cup diced onion",
            "Pickle relish"
        ],
        "instructions": [
            "Cook the sausages thoroughly according to package instructions.",
            "Toast the buns lightly.",
            "Place the sausages inside the buns.",
            "Add mustard and ketchup.",
            "Top with onion and relish.",
            "Serve immediately."
        ]
    },

    # --------------------------------------------------------
    # 26. HUEVOS RANCHEROS
    # --------------------------------------------------------

    {
        "food": "huevos_rancheros",
        "recipe_name": "Mexican Huevos Rancheros",
        "servings": 2,
        "calories": 420,
        "protein": 22,
        "carbs": 34,
        "fat": 23,
        "ingredients": [
            "4 eggs",
            "2 corn tortillas",
            "2 tomatoes",
            "1/2 onion",
            "1 chili",
            "1/2 cup black beans",
            "1/4 cup cheese",
            "Cilantro",
            "Salt"
        ],
        "instructions": [
            "Cook tomatoes, onion and chili into a simple salsa.",
            "Warm the black beans.",
            "Toast the tortillas.",
            "Cook the eggs until the whites are set.",
            "Place tortillas on plates.",
            "Add beans and eggs.",
            "Top with salsa, cheese and cilantro."
        ]
    },

    # --------------------------------------------------------
    # 27. HUMMUS
    # --------------------------------------------------------

    {
        "food": "hummus",
        "recipe_name": "Creamy Homemade Hummus",
        "servings": 4,
        "calories": 240,
        "protein": 8,
        "carbs": 21,
        "fat": 15,
        "ingredients": [
            "1 can chickpeas",
            "1/4 cup tahini",
            "2 tbsp lemon juice",
            "1 garlic clove",
            "2 tbsp olive oil",
            "Salt",
            "Water as needed"
        ],
        "instructions": [
            "Drain and rinse the chickpeas.",
            "Blend chickpeas, tahini, lemon juice and garlic.",
            "Add olive oil.",
            "Blend until smooth.",
            "Add water gradually to reach the desired texture.",
            "Season with salt.",
            "Serve with olive oil on top."
        ]
    },

    # --------------------------------------------------------
    # 28. LOBSTER BISQUE
    # --------------------------------------------------------

    {
        "food": "lobster_bisque",
        "recipe_name": "Creamy Lobster Bisque",
        "servings": 4,
        "calories": 360,
        "protein": 25,
        "carbs": 18,
        "fat": 23,
        "ingredients": [
            "400 g cooked lobster meat",
            "1 onion",
            "1 carrot",
            "1 celery stalk",
            "2 cups seafood stock",
            "1 cup cream",
            "2 tbsp butter",
            "1 tbsp tomato paste",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Sauté onion, carrot and celery in butter.",
            "Add tomato paste and cook briefly.",
            "Pour in seafood stock.",
            "Simmer until the vegetables are tender.",
            "Blend until smooth.",
            "Add cream and lobster meat.",
            "Heat gently without boiling.",
            "Season and serve."
        ]
    },

    # --------------------------------------------------------
    # 29. LOBSTER ROLL
    # --------------------------------------------------------

    {
        "food": "lobster_roll_sandwich",
        "recipe_name": "Classic Lobster Roll",
        "servings": 2,
        "calories": 470,
        "protein": 31,
        "carbs": 38,
        "fat": 21,
        "ingredients": [
            "300 g cooked lobster meat",
            "2 split-top buns",
            "2 tbsp mayonnaise",
            "1 tbsp lemon juice",
            "1 celery stalk",
            "1 tbsp chives",
            "1 tbsp butter",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Chop the cooked lobster meat.",
            "Mix lobster with mayonnaise, lemon juice and celery.",
            "Add chives, salt and pepper.",
            "Butter the buns.",
            "Toast the buns lightly.",
            "Fill the buns with lobster mixture.",
            "Serve immediately."
        ]
    },

    # --------------------------------------------------------
    # 30. MACARONS
    # --------------------------------------------------------

    {
        "food": "macarons",
        "recipe_name": "French Almond Macarons",
        "servings": 8,
        "calories": 190,
        "protein": 4,
        "carbs": 25,
        "fat": 9,
        "ingredients": [
            "1 cup almond flour",
            "1 1/2 cups powdered sugar",
            "3 egg whites",
            "1/4 cup sugar",
            "1 tsp vanilla",
            "Buttercream filling"
        ],
        "instructions": [
            "Sift almond flour and powdered sugar.",
            "Whip egg whites with sugar until glossy.",
            "Fold the dry ingredients into the meringue.",
            "Pipe small circles onto a baking tray.",
            "Let them rest before baking.",
            "Bake until the shells are set.",
            "Cool completely.",
            "Sandwich with buttercream."
        ]
    },

    # --------------------------------------------------------
    # 31. MISO SOUP
    # --------------------------------------------------------

    {
        "food": "miso_soup",
        "recipe_name": "Japanese Miso Soup",
        "servings": 2,
        "calories": 90,
        "protein": 6,
        "carbs": 9,
        "fat": 3,
        "ingredients": [
            "3 cups dashi or vegetable stock",
            "2 tbsp miso paste",
            "100 g tofu",
            "1 tbsp wakame",
            "2 green onions"
        ],
        "instructions": [
            "Heat the stock gently.",
            "Soak wakame until softened.",
            "Add tofu and wakame.",
            "Turn off the heat.",
            "Dissolve miso paste in a small amount of warm broth.",
            "Return the mixture to the pot.",
            "Garnish with green onions.",
            "Serve warm."
        ]
    },

    # --------------------------------------------------------
    # 32. MUSSELS
    # --------------------------------------------------------

    {
        "food": "mussels",
        "recipe_name": "Garlic White Wine Mussels",
        "servings": 2,
        "calories": 330,
        "protein": 30,
        "carbs": 12,
        "fat": 15,
        "ingredients": [
            "1 kg fresh mussels",
            "3 cloves garlic",
            "1 cup white wine",
            "2 tbsp butter",
            "2 tbsp parsley",
            "1 tbsp lemon juice"
        ],
        "instructions": [
            "Clean and rinse the mussels.",
            "Sauté garlic in butter.",
            "Add white wine.",
            "Add mussels and cover the pan.",
            "Steam until the mussels open and are fully cooked.",
            "Discard any mussels that remain closed.",
            "Add parsley and lemon juice.",
            "Serve immediately."
        ]
    },

    # --------------------------------------------------------
    # 33. NACHOS
    # --------------------------------------------------------

    {
        "food": "nachos",
        "recipe_name": "Loaded Cheesy Nachos",
        "servings": 4,
        "calories": 510,
        "protein": 21,
        "carbs": 48,
        "fat": 28,
        "ingredients": [
            "200 g tortilla chips",
            "1 cup shredded cheddar cheese",
            "1/2 cup black beans",
            "1 tomato",
            "1/4 onion",
            "1 jalapeño",
            "1/4 cup sour cream",
            "Salsa"
        ],
        "instructions": [
            "Spread tortilla chips on a baking tray.",
            "Add cheese and black beans.",
            "Bake until the cheese melts.",
            "Top with tomato, onion and jalapeño.",
            "Add salsa and sour cream.",
            "Serve immediately."
        ]
    },

    # --------------------------------------------------------
    # 34. OMELETTE
    # --------------------------------------------------------

    {
        "food": "omelette",
        "recipe_name": "Vegetable Cheese Omelette",
        "servings": 1,
        "calories": 320,
        "protein": 21,
        "carbs": 8,
        "fat": 23,
        "ingredients": [
            "3 eggs",
            "1/4 cup bell pepper",
            "1/4 cup onion",
            "1/4 cup mushrooms",
            "1/4 cup shredded cheese",
            "1 tsp butter",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Whisk the eggs with salt and pepper.",
            "Sauté the vegetables in butter.",
            "Pour in the eggs.",
            "Cook until the bottom is set.",
            "Add cheese to one side.",
            "Fold the omelette.",
            "Cook until the eggs are fully set.",
            "Serve hot."
        ]
    },

    # --------------------------------------------------------
    # 35. ONION RINGS
    # --------------------------------------------------------

    {
        "food": "onion_rings",
        "recipe_name": "Crispy Onion Rings",
        "servings": 3,
        "calories": 360,
        "protein": 7,
        "carbs": 48,
        "fat": 16,
        "ingredients": [
            "2 large onions",
            "1 cup flour",
            "1 egg",
            "1 cup breadcrumbs",
            "1/2 tsp paprika",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Slice onions into rings.",
            "Separate the rings.",
            "Coat each ring in flour.",
            "Dip into beaten egg.",
            "Coat with breadcrumbs.",
            "Fry until golden and crispy.",
            "Drain and season with salt."
        ]
    },

    # --------------------------------------------------------
    # 36. OYSTERS
    # --------------------------------------------------------

    {
        "food": "oysters",
        "recipe_name": "Classic Baked Garlic Oysters",
        "servings": 2,
        "calories": 230,
        "protein": 18,
        "carbs": 8,
        "fat": 14,
        "ingredients": [
            "12 fresh oysters",
            "2 tbsp butter",
            "2 cloves garlic",
            "2 tbsp breadcrumbs",
            "1 tbsp parsley",
            "1 tbsp lemon juice"
        ],
        "instructions": [
            "Carefully clean and shuck the oysters.",
            "Arrange them securely on a baking tray.",
            "Mix melted butter with garlic and parsley.",
            "Spoon the mixture over the oysters.",
            "Sprinkle with breadcrumbs.",
            "Bake until the oysters are thoroughly cooked.",
            "Finish with lemon juice.",
            "Serve immediately."
        ]
    },

    # --------------------------------------------------------
    # 37. PAD THAI
    # --------------------------------------------------------

    {
        "food": "pad_thai",
        "recipe_name": "Classic Chicken Pad Thai",
        "servings": 2,
        "calories": 520,
        "protein": 29,
        "carbs": 65,
        "fat": 17,
        "ingredients": [
            "180 g rice noodles",
            "200 g chicken breast",
            "1 egg",
            "1/2 cup bean sprouts",
            "2 green onions",
            "2 tbsp crushed peanuts",
            "2 tbsp fish sauce",
            "1 tbsp lime juice",
            "1 tbsp brown sugar"
        ],
        "instructions": [
            "Soak the rice noodles until softened.",
            "Cook the chicken in a hot wok.",
            "Add the egg and scramble.",
            "Add the noodles.",
            "Mix fish sauce, lime juice and brown sugar.",
            "Pour the sauce into the wok.",
            "Toss with bean sprouts and green onions.",
            "Top with crushed peanuts."
        ]
    },

    # --------------------------------------------------------
    # 38. PAELLA
    # --------------------------------------------------------

    {
        "food": "paella",
        "recipe_name": "Spanish Chicken Paella",
        "servings": 4,
        "calories": 560,
        "protein": 31,
        "carbs": 65,
        "fat": 19,
        "ingredients": [
            "2 cups short-grain rice",
            "300 g chicken",
            "1 onion",
            "1 bell pepper",
            "1 cup peas",
            "4 cups chicken stock",
            "1 tsp paprika",
            "1/2 tsp saffron",
            "2 tbsp olive oil",
            "Salt"
        ],
        "instructions": [
            "Heat olive oil in a wide pan.",
            "Brown the chicken pieces.",
            "Add onion and bell pepper.",
            "Stir in paprika and rice.",
            "Add stock and saffron.",
            "Arrange the chicken evenly.",
            "Cook without excessive stirring until the rice is tender.",
            "Add peas near the end.",
            "Rest briefly before serving."
        ]
    },

    # --------------------------------------------------------
    # 39. PANNA COTTA
    # --------------------------------------------------------

    {
        "food": "panna_cotta",
        "recipe_name": "Vanilla Panna Cotta",
        "servings": 4,
        "calories": 310,
        "protein": 5,
        "carbs": 24,
        "fat": 22,
        "ingredients": [
            "2 cups heavy cream",
            "1/3 cup sugar",
            "1 tsp vanilla",
            "2 tsp gelatin",
            "3 tbsp water",
            "Fresh berries"
        ],
        "instructions": [
            "Bloom gelatin in cold water.",
            "Heat cream, sugar and vanilla gently.",
            "Remove from heat.",
            "Stir in the gelatin.",
            "Pour into serving glasses.",
            "Refrigerate until set.",
            "Serve with fresh berries."
        ]
    },

    # --------------------------------------------------------
    # 40. PEKING DUCK
    # --------------------------------------------------------

    {
        "food": "peking_duck",
        "recipe_name": "Crispy Peking Duck",
        "servings": 4,
        "calories": 590,
        "protein": 38,
        "carbs": 24,
        "fat": 39,
        "ingredients": [
            "1 whole duck",
            "2 tbsp honey",
            "1 tbsp soy sauce",
            "1 tsp five-spice powder",
            "Chinese pancakes",
            "Cucumber",
            "Spring onions"
        ],
        "instructions": [
            "Clean and prepare the duck.",
            "Brush the skin with a mixture of honey and soy sauce.",
            "Season lightly with five-spice powder.",
            "Roast until the skin is crisp and the duck is thoroughly cooked.",
            "Rest before carving.",
            "Slice the meat and crispy skin.",
            "Serve with pancakes, cucumber and spring onions."
        ]
    },

    # --------------------------------------------------------
    # 41. PHO
    # --------------------------------------------------------

    {
        "food": "pho",
        "recipe_name": "Vietnamese Chicken Pho",
        "servings": 4,
        "calories": 430,
        "protein": 29,
        "carbs": 53,
        "fat": 10,
        "ingredients": [
            "200 g rice noodles",
            "500 g chicken",
            "6 cups chicken broth",
            "1 onion",
            "1 piece ginger",
            "1 cinnamon stick",
            "2 star anise",
            "Bean sprouts",
            "Fresh herbs",
            "Lime"
        ],
        "instructions": [
            "Simmer chicken broth with onion, ginger, cinnamon and star anise.",
            "Cook the chicken thoroughly and shred it.",
            "Prepare rice noodles according to package instructions.",
            "Strain the broth.",
            "Place noodles in bowls.",
            "Add shredded chicken.",
            "Pour hot broth over the noodles.",
            "Top with bean sprouts, herbs and lime."
        ]
    },

    # --------------------------------------------------------
    # 42. PORK CHOP
    # --------------------------------------------------------

    {
        "food": "pork_chop",
        "recipe_name": "Herb Garlic Pork Chops",
        "servings": 2,
        "calories": 480,
        "protein": 42,
        "carbs": 5,
        "fat": 32,
        "ingredients": [
            "2 pork chops",
            "1 tbsp olive oil",
            "2 tbsp butter",
            "2 cloves garlic",
            "1 tsp rosemary",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Season pork chops with salt and pepper.",
            "Heat olive oil in a skillet.",
            "Sear the pork chops on both sides.",
            "Add butter, garlic and rosemary.",
            "Baste the chops.",
            "Cook until they reach a safe internal temperature.",
            "Rest before serving."
        ]
    },

    # --------------------------------------------------------
    # 43. POUTINE
    # --------------------------------------------------------

    {
        "food": "poutine",
        "recipe_name": "Classic Canadian Poutine",
        "servings": 3,
        "calories": 620,
        "protein": 18,
        "carbs": 72,
        "fat": 30,
        "ingredients": [
            "500 g potatoes",
            "1 cup cheese curds",
            "1 cup beef gravy",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Cut potatoes into fries.",
            "Cook the fries until golden and crispy.",
            "Heat the gravy.",
            "Place fries on serving plates.",
            "Top with cheese curds.",
            "Pour hot gravy over the fries.",
            "Serve immediately."
        ]
    },

    # --------------------------------------------------------
    # 44. PRIME RIB
    # --------------------------------------------------------

    {
        "food": "prime_rib",
        "recipe_name": "Herb Crusted Prime Rib",
        "servings": 6,
        "calories": 680,
        "protein": 49,
        "carbs": 3,
        "fat": 51,
        "ingredients": [
            "1.5 kg prime rib roast",
            "2 tbsp olive oil",
            "3 cloves garlic",
            "1 tbsp rosemary",
            "1 tbsp thyme",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Pat the roast dry.",
            "Rub with olive oil, garlic and herbs.",
            "Season generously with salt and pepper.",
            "Roast until the desired safe internal temperature is reached.",
            "Rest the roast before slicing.",
            "Slice and serve with pan juices."
        ]
    },

    # --------------------------------------------------------
    # 45. PULLED PORK SANDWICH
    # --------------------------------------------------------

    {
        "food": "pulled_pork_sandwich",
        "recipe_name": "BBQ Pulled Pork Sandwich",
        "servings": 4,
        "calories": 540,
        "protein": 36,
        "carbs": 51,
        "fat": 22,
        "ingredients": [
            "700 g pork shoulder",
            "1 cup BBQ sauce",
            "4 burger buns",
            "1 cup coleslaw",
            "1 onion",
            "1 tsp paprika",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Season the pork with paprika, salt and pepper.",
            "Slow-cook the pork until tender and thoroughly cooked.",
            "Shred the pork with forks.",
            "Mix with BBQ sauce.",
            "Toast the buns.",
            "Fill each bun with pulled pork.",
            "Top with coleslaw.",
            "Serve warm."
        ]
    },

    # --------------------------------------------------------
    # 46. RAVIOLI
    # --------------------------------------------------------

    {
        "food": "ravioli",
        "recipe_name": "Cheese Ravioli with Tomato Sauce",
        "servings": 2,
        "calories": 490,
        "protein": 20,
        "carbs": 62,
        "fat": 18,
        "ingredients": [
            "250 g cheese ravioli",
            "1 cup tomato sauce",
            "1/4 cup Parmesan cheese",
            "1 tbsp olive oil",
            "1 clove garlic",
            "Fresh basil"
        ],
        "instructions": [
            "Boil ravioli according to package instructions.",
            "Heat olive oil in a pan.",
            "Sauté garlic briefly.",
            "Add tomato sauce.",
            "Add cooked ravioli.",
            "Toss gently with the sauce.",
            "Top with Parmesan and basil."
        ]
    },

    # --------------------------------------------------------
    # 47. RED VELVET CAKE
    # --------------------------------------------------------

    {
        "food": "red_velvet_cake",
        "recipe_name": "Classic Red Velvet Cake",
        "servings": 8,
        "calories": 470,
        "protein": 6,
        "carbs": 61,
        "fat": 23,
        "ingredients": [
            "2 cups flour",
            "1 cup sugar",
            "2 eggs",
            "1/2 cup butter",
            "1 cup buttermilk",
            "2 tbsp cocoa powder",
            "1 tsp vanilla",
            "Red food coloring",
            "Cream cheese frosting"
        ],
        "instructions": [
            "Cream butter and sugar.",
            "Beat in eggs and vanilla.",
            "Mix flour and cocoa separately.",
            "Add dry ingredients alternately with buttermilk.",
            "Add red food coloring.",
            "Bake until a toothpick comes out clean.",
            "Cool completely.",
            "Frost with cream cheese frosting."
        ]
    },

    # --------------------------------------------------------
    # 48. RISOTTO
    # --------------------------------------------------------

    {
        "food": "risotto",
        "recipe_name": "Creamy Mushroom Risotto",
        "servings": 3,
        "calories": 430,
        "protein": 11,
        "carbs": 61,
        "fat": 16,
        "ingredients": [
            "1 cup Arborio rice",
            "200 g mushrooms",
            "4 cups vegetable stock",
            "1 onion",
            "2 tbsp butter",
            "1/2 cup Parmesan",
            "1/2 cup white wine",
            "Black pepper"
        ],
        "instructions": [
            "Sauté onion and mushrooms in butter.",
            "Add Arborio rice and toast lightly.",
            "Pour in white wine.",
            "Add warm stock one ladle at a time.",
            "Stir regularly until the rice becomes creamy and tender.",
            "Stir in Parmesan.",
            "Season with black pepper.",
            "Serve immediately."
        ]
    },

    # --------------------------------------------------------
    # 49. SASHIMI
    # --------------------------------------------------------

    {
        "food": "sashimi",
        "recipe_name": "Fresh Salmon Sashimi",
        "servings": 2,
        "calories": 240,
        "protein": 30,
        "carbs": 2,
        "fat": 12,
        "ingredients": [
            "250 g sushi-grade salmon",
            "Soy sauce",
            "Wasabi",
            "Pickled ginger",
            "Daikon radish"
        ],
        "instructions": [
            "Use properly handled sushi-grade salmon from a reputable source.",
            "Keep the fish properly chilled.",
            "Slice the salmon into thin even pieces with a clean sharp knife.",
            "Arrange the slices on a chilled plate.",
            "Serve with soy sauce, wasabi and pickled ginger."
        ]
    },

    # --------------------------------------------------------
    # 50. SCALLOPS
    # --------------------------------------------------------

    {
        "food": "scallops",
        "recipe_name": "Pan Seared Garlic Scallops",
        "servings": 2,
        "calories": 290,
        "protein": 30,
        "carbs": 5,
        "fat": 17,
        "ingredients": [
            "300 g sea scallops",
            "1 tbsp olive oil",
            "2 tbsp butter",
            "2 cloves garlic",
            "1 tbsp lemon juice",
            "Parsley",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Pat scallops completely dry.",
            "Season with salt and pepper.",
            "Heat olive oil in a skillet.",
            "Sear scallops until golden on both sides and cooked through.",
            "Add butter and garlic.",
            "Baste the scallops briefly.",
            "Finish with lemon juice and parsley."
        ]
    },

    # --------------------------------------------------------
    # 51. SEAWEED SALAD
    # --------------------------------------------------------

    {
        "food": "seaweed_salad",
        "recipe_name": "Japanese Wakame Seaweed Salad",
        "servings": 2,
        "calories": 110,
        "protein": 3,
        "carbs": 13,
        "fat": 5,
        "ingredients": [
            "30 g dried wakame",
            "1 cucumber",
            "1 tbsp rice vinegar",
            "1 tsp sesame oil",
            "1 tsp soy sauce",
            "1 tsp sesame seeds"
        ],
        "instructions": [
            "Soak dried wakame according to package instructions.",
            "Drain and squeeze out excess water.",
            "Slice the cucumber thinly.",
            "Combine wakame and cucumber.",
            "Mix vinegar, sesame oil and soy sauce.",
            "Pour dressing over the salad.",
            "Top with sesame seeds."
        ]
    },

    # --------------------------------------------------------
    # 52. SHRIMP AND GRITS
    # --------------------------------------------------------

    {
        "food": "shrimp_and_grits",
        "recipe_name": "Southern Shrimp and Grits",
        "servings": 3,
        "calories": 510,
        "protein": 31,
        "carbs": 48,
        "fat": 23,
        "ingredients": [
            "300 g shrimp",
            "1 cup cornmeal grits",
            "3 cups water or stock",
            "1/2 cup cheddar cheese",
            "2 cloves garlic",
            "1 tbsp butter",
            "1 tbsp olive oil",
            "Paprika",
            "Salt",
            "Black pepper"
        ],
        "instructions": [
            "Cook grits until creamy.",
            "Stir in cheddar cheese.",
            "Season shrimp with paprika, salt and pepper.",
            "Heat olive oil in a skillet.",
            "Cook shrimp thoroughly.",
            "Add garlic and butter.",
            "Serve shrimp over creamy grits."
        ]
    },

    # --------------------------------------------------------
    # 53. SPRING ROLLS
    # --------------------------------------------------------

    {
        "food": "spring_rolls",
        "recipe_name": "Crispy Vegetable Spring Rolls",
        "servings": 4,
        "calories": 290,
        "protein": 7,
        "carbs": 39,
        "fat": 12,
        "ingredients": [
            "8 spring roll wrappers",
            "1 cup shredded cabbage",
            "1 carrot",
            "1/2 bell pepper",
            "1/2 cup bean sprouts",
            "1 tbsp soy sauce",
            "1 tsp sesame oil"
        ],
        "instructions": [
            "Stir-fry cabbage, carrot, bell pepper and bean sprouts.",
            "Add soy sauce and sesame oil.",
            "Cool the filling slightly.",
            "Place filling into wrappers.",
            "Roll tightly and seal.",
            "Cook until golden and crispy.",
            "Serve with dipping sauce."
        ]
    },

    # --------------------------------------------------------
    # 54. STRAWBERRY SHORTCAKE
    # --------------------------------------------------------

    {
        "food": "strawberry_shortcake",
        "recipe_name": "Classic Strawberry Shortcake",
        "servings": 6,
        "calories": 360,
        "protein": 6,
        "carbs": 48,
        "fat": 16,
        "ingredients": [
            "2 cups flour",
            "1/4 cup sugar",
            "1 tbsp baking powder",
            "1/2 cup butter",
            "3/4 cup milk",
            "2 cups strawberries",
            "1 cup whipped cream"
        ],
        "instructions": [
            "Mix flour, sugar and baking powder.",
            "Cut in butter.",
            "Add milk and form a soft dough.",
            "Shape into shortcakes.",
            "Bake until golden.",
            "Slice strawberries and mix with a little sugar.",
            "Split the shortcakes.",
            "Layer strawberries and whipped cream between them."
        ]
    },

    # --------------------------------------------------------
    # 55. TAKOYAKI
    # --------------------------------------------------------

    {
        "food": "takoyaki",
        "recipe_name": "Japanese Takoyaki",
        "servings": 3,
        "calories": 330,
        "protein": 16,
        "carbs": 37,
        "fat": 13,
        "ingredients": [
            "1 cup flour",
            "2 eggs",
            "2 cups dashi",
            "150 g cooked octopus",
            "2 green onions",
            "Tenkasu",
            "Takoyaki sauce",
            "Japanese mayonnaise",
            "Bonito flakes"
        ],
        "instructions": [
            "Whisk flour, eggs and dashi into a thin batter.",
            "Heat a takoyaki pan and lightly oil it.",
            "Pour batter into the molds.",
            "Add pieces of cooked octopus, green onion and tenkasu.",
            "Turn the balls as they cook.",
            "Cook until golden and fully set.",
            "Top with takoyaki sauce and mayonnaise.",
            "Finish with bonito flakes."
        ]
    },

    # --------------------------------------------------------
    # 56. TIRAMISU
    # --------------------------------------------------------

    {
        "food": "tiramisu",
        "recipe_name": "Classic Italian Tiramisu",
        "servings": 6,
        "calories": 430,
        "protein": 7,
        "carbs": 39,
        "fat": 28,
        "ingredients": [
            "250 g mascarpone",
            "3 eggs",
            "1/3 cup sugar",
            "200 g ladyfingers",
            "1 cup strong coffee",
            "2 tbsp cocoa powder"
        ],
        "instructions": [
            "Separate the eggs.",
            "Whisk yolks with sugar until creamy.",
            "Mix in mascarpone.",
            "Whip the egg whites until fluffy and fold them in.",
            "Dip ladyfingers briefly into cooled coffee.",
            "Layer ladyfingers and mascarpone mixture.",
            "Refrigerate for several hours.",
            "Dust with cocoa powder before serving."
        ]
    },

    # --------------------------------------------------------
    # 57. TUNA TARTARE
    # --------------------------------------------------------

    {
        "food": "tuna_tartare",
        "recipe_name": "Fresh Tuna Tartare",
        "servings": 2,
        "calories": 260,
        "protein": 30,
        "carbs": 7,
        "fat": 12,
        "ingredients": [
            "250 g sushi-grade tuna",
            "1 tbsp soy sauce",
            "1 tsp sesame oil",
            "1 tsp lime juice",
            "1 green onion",
            "1/2 avocado",
            "Sesame seeds"
        ],
        "instructions": [
            "Use properly handled sushi-grade tuna from a reputable source.",
            "Keep the tuna properly chilled.",
            "Dice the tuna into small cubes.",
            "Mix with soy sauce, sesame oil and lime juice.",
            "Add chopped green onion.",
            "Serve with diced avocado.",
            "Garnish with sesame seeds."
        ]
    }
]


# ============================================================
# EXISTING FOOD CLASSES
# ============================================================

existing_foods = set(
    recipe["food"]
    for recipe in recipes
)


# ============================================================
# ADD ONLY MISSING RECIPES
# ============================================================

added_count = 0
skipped_count = 0
invalid_count = 0

print("\nProcessing master recipe list...")
print("-" * 65)

for recipe in new_recipes:

    food = recipe["food"]

    # Check against model classes
    if food not in model_classes:
        print(f"INVALID MODEL CLASS: {food}")
        invalid_count += 1
        continue

    # Prevent duplicate food entries
    if food in existing_foods:
        print(f"Already exists: {food}")
        skipped_count += 1
        continue

    recipes.append(recipe)
    existing_foods.add(food)

    print(f"Added: {food}")
    added_count += 1


# ============================================================
# REMOVE INVALID EXISTING RECIPES
# ============================================================

print("\nChecking existing recipes for invalid classes...")

cleaned_recipes = []

removed_invalid = []

for recipe in recipes:

    food = recipe.get("food")

    if food in model_classes:
        cleaned_recipes.append(recipe)
    else:
        removed_invalid.append(food)


recipes = cleaned_recipes


# ============================================================
# SAVE DATABASE
# ============================================================

with open(RECIPE_PATH, "w", encoding="utf-8") as f:
    json.dump(
        recipes,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

recipe_foods = set(
    recipe["food"]
    for recipe in recipes
)

missing_classes = sorted(
    set(model_classes) - recipe_foods
)

extra_classes = sorted(
    recipe_foods - set(model_classes)
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 65)
print("RECIPE DATABASE EXPANSION COMPLETE")
print("=" * 65)

print(f"\nFood-101 classes:       {len(model_classes)}")
print(f"Total recipes:         {len(recipes)}")
print(f"Recipe-covered foods:  {len(recipe_foods)}")
print(f"Recipes added:         {added_count}")
print(f"Already existed:       {skipped_count}")
print(f"Invalid new recipes:   {invalid_count}")

if removed_invalid:
    print("\nInvalid existing recipes removed:")
    for food in removed_invalid:
        print(f"- {food}")

print("\n" + "-" * 65)

if missing_classes:
    print(f"Missing classes: {len(missing_classes)}")
    for food in missing_classes:
        print(f"- {food}")
else:
    print("Missing classes:        0")

print(f"Extra invalid classes:  {len(extra_classes)}")

if extra_classes:
    for food in extra_classes:
        print(f"- {food}")

print("-" * 65)

if len(recipe_foods) == len(model_classes) and not missing_classes and not extra_classes:
    print("SUCCESS!")
    print("All 101 Food-101 classes now have recipes.")
else:
    print("WARNING: Recipe coverage is not complete.")

print("=" * 65)