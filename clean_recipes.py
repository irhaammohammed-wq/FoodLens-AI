import json
import torch


# ============================================================
# FOODLENS AI - RECIPE DATABASE CLEANUP
# ============================================================

RECIPES_FILE = "recipes/recipes.json"
MODEL_FILE = "models/food101_resnet50.pth"


# ============================================================
# LOAD MODEL CLASS NAMES
# ============================================================

print("=" * 60)
print("FOODLENS AI - RECIPE DATABASE CLEANUP")
print("=" * 60)

print("\nLoading trained model classes...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=False
)

model_classes = set(checkpoint["classes"])

print(f"Model classes: {len(model_classes)}")


# ============================================================
# LOAD RECIPE DATABASE
# ============================================================

print("\nLoading recipe database...")

with open(RECIPES_FILE, "r", encoding="utf-8") as file:
    recipes = json.load(file)

print(f"Recipes before cleanup: {len(recipes)}")


# ============================================================
# FIND INVALID FOOD CLASSES
# ============================================================

invalid_foods = sorted(
    set(recipe["food"] for recipe in recipes)
    - model_classes
)

if invalid_foods:
    print("\nInvalid food classes found:")

    for food in invalid_foods:
        print(f"- {food}")

else:
    print("\nNo invalid food classes found.")


# ============================================================
# REMOVE INVALID RECIPES
# ============================================================

cleaned_recipes = [
    recipe
    for recipe in recipes
    if recipe["food"] in model_classes
]


removed_count = len(recipes) - len(cleaned_recipes)


# ============================================================
# SAVE CLEAN DATABASE
# ============================================================

with open(RECIPES_FILE, "w", encoding="utf-8") as file:
    json.dump(
        cleaned_recipes,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

unique_foods = sorted(
    set(recipe["food"] for recipe in cleaned_recipes)
)

print("\n" + "=" * 60)
print("CLEANUP COMPLETE")
print("=" * 60)

print(f"Recipes removed: {removed_count}")
print(f"Recipes remaining: {len(cleaned_recipes)}")
print(f"Unique foods remaining: {len(unique_foods)}")

print("\nRemaining recipe food classes:")

for food in unique_foods:
    print(f"- {food}")

print("\n" + "=" * 60)
print("DATABASE NOW MATCHES THE TRAINED MODEL")
print("=" * 60)