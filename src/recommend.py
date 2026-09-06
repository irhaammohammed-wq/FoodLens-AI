import pandas as pd


# ==========================================
# LOAD RECIPE DATA
# ==========================================

recipes = pd.read_csv("./data/recipes.csv")


# ==========================================
# GET FOOD NAME
# ==========================================

predicted_food = "pizza"


# ==========================================
# FIND MATCHING RECIPES
# ==========================================

matching_recipes = recipes[
    recipes["food_name"] == predicted_food
]


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 50)
print(f"RECIPE RECOMMENDATIONS FOR: {predicted_food.upper()}")
print("=" * 50)

if matching_recipes.empty:
    print("\nNo recipes found.")

else:
    for index, recipe in matching_recipes.iterrows():

        print(f"\n🍽️ Recipe: {recipe['recipe_name']}")
        print(f"👥 Servings: {recipe['servings']}")
        print(f"🔥 Calories: {recipe['calories']} kcal")
        print(f"💪 Protein: {recipe['protein_g']} g")
        print(f"🍞 Carbs: {recipe['carbs_g']} g")
        print(f"🥑 Fat: {recipe['fat_g']} g")

        print("\n🥕 Ingredients:")
        ingredients = recipe["ingredients"].split(";")

        for ingredient in ingredients:
            print("-", ingredient.strip())

        print("\n👨‍🍳 Instructions:")
        instructions = recipe["instructions"].split(".")

        step_number = 1

        for instruction in instructions:

            if instruction.strip():

                print(
                    f"{step_number}. {instruction.strip()}."
                )

                step_number += 1

        print("\n" + "-" * 50)


print("\nDone!")