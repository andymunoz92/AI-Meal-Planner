from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import FewShotPromptTemplate
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
import os
# from langchain.output_parsers import ResponseSchema, StructuredOutputParser

app = Flask(__name__)
CORS(app)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_token = "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"
model_id = "meta-llama/Llama-3.2-3B"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='mps')

"""pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    max_new_tokens=8192,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
    num_beams=4,  # Added beam search
    early_stopping=True,  # Now works with beam search
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    length_penalty=1.0  # Added to control output length
)

# Create LangChain HF Pipeline
llm = HuggingFacePipeline(pipeline=pipe)"""

# Define example templates
diet_template = """
Diet: {diet_name}
***Focus:** {focus}
***Foods to eat:** {foods_to_eat}
***Foods to avoid:** {foods_to_avoid}
"""

budget_template = """
Budget: {budget_level}
***Grocery list:** {grocery_list}
***Meal planning tips:** {meal_planning_tips}
"""

cooking_template = """
Cooking Skill: {cooking_skill}
***Recipes:** {recipes}
***Cooking techniques:** {cooking_techniques}
"""

# Define our examples
diet_examples = [
    {
        "diet_name": "Paleo Diet",
        "focus": "Consuming foods similar to those eaten by our Paleolithic ancestors",
        "foods_to_eat": "Meats, fish, poultry, eggs, vegetables, fruits, nuts, and seeds",
        "foods_to_avoid": "Grains, legumes, dairy products, added sugar, and processed oils"
    },
    {
        "diet_name": "Keto Diet",
        "focus": "Achieving ketosis through high-fat, very low-carb eating",
        "foods_to_eat": "Fatty meats, fish, eggs, dairy, low-carb vegetables, nuts, and healthy oils",
        "foods_to_avoid": "Grains, sugars, most fruits, starchy vegetables, and processed foods"
    },
    {
        "diet_name": "Mediterranean Diet",
        "focus": "Heart-healthy eating based on traditional Mediterranean cuisine",
        "foods_to_eat": "Vegetables, fruits, whole grains, legumes, fish, olive oil, and moderate wine",
        "foods_to_avoid": "Red meat, processed foods, refined grains, and added sugars"
    }
]

budget_examples = [
    {
        "budget_level": "Low",
        "grocery_list": "Buy in bulk, shop for seasonal produce, and plan meals around affordable protein sources",
        "meal_planning_tips": "Cook meals in advance, use leftovers, and avoid dining out"
    },
    {
        "budget_level": "Medium",
        "grocery_list": "Shop for a mix of organic and conventional produce, buy in bulk, and plan meals around moderate-cost protein sources",
        "meal_planning_tips": "Cook meals in advance, use leftovers, and dine out occasionally"
    },
    {
        "budget_level": "High",
        "grocery_list": "Shop for organic and specialty produce, buy in bulk, and plan meals around high-end protein sources",
        "meal_planning_tips": "Hire a personal chef, dine out frequently, and use meal delivery services"
    }
]

cooking_examples = [
    {
        "cooking_skill": "Beginner",
        "recipes": "Simple meals like grilled chicken, roasted vegetables, and one-pot dishes",
        "cooking_techniques": "Basic knife skills, cooking methods like boiling and steaming, and meal prep"
    },
    {
        "cooking_skill": "Intermediate",
        "recipes": "More complex meals like stir-fries, curries, and roasted meats",
        "cooking_techniques": "Advanced knife skills, cooking methods like sautéing and braising, and meal planning"
    },
    {
        "cooking_skill": "Advanced",
        "recipes": "Intricate meals like sushi, molecular gastronomy, and haute cuisine",
        "cooking_techniques": "Expert knife skills, advanced cooking methods like sous vide and foamification, and plating techniques"
    }
]

# Create the example prompt templates
diet_prompt = PromptTemplate(
    input_variables=["diet_name", "focus", "foods_to_eat", "foods_to_avoid"],
    template=diet_template
)

budget_prompt = PromptTemplate(
    input_variables=["budget_level", "grocery_list", "meal_planning_tips"],
    template=budget_template
)

cooking_prompt = PromptTemplate(
    input_variables=["cooking_skill", "recipes", "cooking_techniques"],
    template=cooking_template
)

# Comment out parser schema
# response_schema = ResponseSchema(
#     name="response",
#     description="Extract everything after 'Response:'"
# )
# parser = StructuredOutputParser.from_response_schemas([response_schema])

# For diet template
diet_few_shot_prompt = FewShotPromptTemplate(
    examples=diet_examples,
    example_prompt=diet_prompt,
    prefix=f"""You are Chef Alex, a professional nutritionist with over 15 years of experience specializing in dietary planning. You specialize in {{diet_type}} cuisine and have helped hundreds of clients achieve their dietary goals while enjoying delicious, satisfying meals. Your expertise focuses on understanding nutritional requirements, creating balanced meal plans, and explaining the science behind different dietary approaches.

Here are some examples of dietary recommendations:""",
    suffix="""Now, provide detailed dietary advice.

Diet Type: {diet_type}
You focus on: Explaining the nutritional benefits of ingredients

Current Query: {user_query}

Please provide a detailed response including:
1. Diet Description (following the example format above)
2. Foods to eat and avoid
3. Nutritional benefits of each food group

Response:""",
    input_variables=["diet_type", "user_query"]
)

# For budget template
budget_few_shot_prompt = FewShotPromptTemplate(
    examples=budget_examples,
    example_prompt=budget_prompt,
    prefix=f"""You are Chef Alex, a culinary economics expert with over 15 years of experience in budget-conscious meal planning. You have helped hundreds of clients optimize their {{budget_level}} food budget without sacrificing quality or nutrition. Your expertise focuses on smart shopping strategies, cost-effective ingredient substitutions, and maximizing value in meal preparation.

Here are some examples of budget-friendly recommendations:""",
    suffix="""Now, provide detailed budget advice.

Diet Type: {diet_type}
Budget: {budget_level}
Current Query: {user_query}

Please provide a detailed response including:
1. Budget Description (following the example format above)
2. Grocery List
3. Meal Planning Tips

Response:""",
    input_variables=["diet_type", "budget_level", "user_query"]
)

# For cooking template
cooking_few_shot_prompt = FewShotPromptTemplate(
    examples=cooking_examples,
    example_prompt=cooking_prompt,
    prefix=f"""You are Chef Alex, a professional chef and cooking instructor with over 15 years of experience in culinary education. You specialize in {{cooking_skill}} cuisine and have helped hundreds of clients achieve their dietary goals while enjoying delicious, satisfying meals. Your expertise focuses on teaching proper techniques, kitchen safety, efficient meal preparation, and building cooking confidence.

Here are some examples of cooking recommendations:""",
    suffix="""Now, following the same format, provide detailed advice for the following:

Diet Type: {diet_type}
Cooking Skill: {cooking_skill}

You focus on:
- Creating easy-to-follow recipes for {cooking_skill} level cooks
- Providing practical cooking tips and substitutions

Current Query: {user_query}

Please provide a detailed response including:
1. Cooking Description (following the example format above)
2. 2-3 Detailed Recipes (with ingredients, instructions, and estimated costs)
3. Practical Cooking Tips

Response:""",
    input_variables=["diet_type", "cooking_skill", "user_query"]
)

"""# Create the chains using the new RunnableSequence approach
diet_chain = diet_few_shot_prompt | llm
budget_chain = budget_few_shot_prompt | llm
cooking_chain = cooking_few_shot_prompt | llm"""

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        data = request.json
        user_query = data.get('prompt', '')
        parameters = data.get('parameters', {})

        if not user_query:
            return jsonify({'error': 'Prompt is required'}), 400

        print(f"Received prompt: {user_query}")
        print(f"Parameters: {parameters}")

        try:
            # Create the pipeline with the provided parameters
            current_generator = pipeline(
                'text-generation',
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                max_new_tokens=parameters.get('max_new_tokens', 8192),
                do_sample=True,
                top_k=parameters.get('top_k', 50),
                top_p=parameters.get('top_p', 0.9),
                temperature=parameters.get('temperature', 0.7),
                repetition_penalty=parameters.get('repetition_penalty', 1.1),
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                length_penalty=1.0
            )

            # Create LangChain HF Pipeline
            llm = HuggingFacePipeline(pipeline=current_generator)


            # Create the chains using the new RunnableSequence approach
            diet_chain = diet_few_shot_prompt | llm
            budget_chain = budget_few_shot_prompt | llm
            cooking_chain = cooking_few_shot_prompt | llm

            # Get selected options from the frontend
            diet_type = request.json.get('diet', 'carnivore')
            cooking_skill = request.json.get('skill', 'beginner')
            budget_level = request.json.get('budget', 'low')

            # Generate responses using the new invoke method
            diet_response = diet_chain.invoke({
                "diet_type": diet_type,
                "user_query": user_query
            })

            budget_response = budget_chain.invoke({
                "diet_type": diet_type,
                "budget_level": budget_level,
                "user_query": user_query
            })

            cooking_response = cooking_chain.invoke({
                "diet_type": diet_type,
                "cooking_skill": cooking_skill,
                "user_query": user_query
            })

            print(f"Generated response length: {len(diet_response)}")
            print(f"Generated response length: {len(budget_response)}")
            print(f"Generated response length: {len(cooking_response)}")

            # Print responses to terminal
            print("Diet Response:")
            print(diet_response)
            print("\nBudget Response:")
            print(budget_response)
            print("\nCooking Response:")
            print(cooking_response)

            return jsonify({
                'status': 'success',
                'diet_response': diet_response.split("Response:")[1].strip() if "Response:" in diet_response else diet_response,
                'budget_response': budget_response.split("Response:")[1].strip() if "Response:" in budget_response else budget_response,
                'cooking_response': cooking_response.split("Response:")[1].strip() if "Response:" in cooking_response else cooking_response
            })

        except Exception as e:
            print(f"Generation error: {str(e)}")
            return jsonify({'error': f'Failed to generate response: {str(e)}'}), 500

    except Exception as e:
        print(f"Request error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=8000, host='0.0.0.0')