# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "0d7229a9-f246-4c18-85d7-07a3fdf6fe89",
# META       "default_lakehouse_name": "LH_demo_genai_functions",
# META       "default_lakehouse_workspace_id": "34e689bf-86c6-4205-a493-a4698e61ffe6",
# META       "known_lakehouses": [
# META         {
# META           "id": "0d7229a9-f246-4c18-85d7-07a3fdf6fe89"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # **GenAI Usecase:** 
# # PySpark UDFs to summarize text, translate text, detect sentiment, correct grammar/spelling, and extract entities using the Gemini API.

# MARKDOWN ********************

# ## <mark>Install and import libraries & dependencies</mark>

# CELL ********************

# Ensure google-generativeai and its dependencies are up to date
%pip install google-generativeai --upgrade
%pip install typing-extensions --upgrade
%pip install tenacity --upgrade

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Below Gemini API key is free-tier. Due to demo/development and testing purposes, I directly set the API key variable within a notebook cell. 
# You should never hardcode secrets in a notebook that will be shared or committed to version control. This prevents the key from ever being exposed in your code.
# Replace with your Gemini API key. Below key will be deleted after demo.
gemini_api_key = "AIzaSyBQ69kQKb6Jhr1934vr5HyyL_X7JVlPGjY" 
broadcasted_key = spark.sparkContext.broadcast(gemini_api_key)
api_key = broadcasted_key.value

# Define LLM model to use
model_name = "gemini-2.0-flash-lite"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import os
import google.generativeai as genai
from tenacity import retry, wait_exponential, wait_fixed, stop_after_attempt, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
spark = SparkSession.builder.appName("Gemini-PySpark-UDF").getOrCreate()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## <mark>Introduce Product Catalog table</mark>
# The "product_catalog" table contains detailed information about a variety of products, including their names, descriptions, origins, reviews, and specific properties. With these columns: 
# 
# - product_name
# - product_description
# - product_origin
# - product_review
# - product_properties

# CELL ********************

df = spark.read.format("delta").table("product_catalog")
display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## <mark>Define function to summarize text</mark>

# CELL ********************

def summarize_text(text):
    # Handle empty input gracefully
    if not text:
        return None
    # Get the configured Gemini model
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    # Function to generate content from the model with a retry strategy.
    @retry(
        wait=wait_exponential(multiplier=3, min=5, max=90),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(ResourceExhausted)
    )
    def _generate_content_with_retry(prompt):
        response = model.generate_content(prompt)
        return response.text
    # Call the API with the retry mechanism.
    try:
        prompt = f"Summarize the following text, only return the summarized text, with no additional commentary. Text: '{text}'."
        response_text = _generate_content_with_retry(prompt)
        return response_text
    except Exception as e:
        return "API_ERROR"

summarize_udf = udf(summarize_text, StringType())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### demo summarize "product_description"

# CELL ********************

df_summary = df.select("product_name","product_description")
display(df_summary)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_summary = df_summary.withColumn("GenAI_summary_product_description", summarize_udf('product_description'))
display(df_summary)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## <mark>Define function to translate a given text to a target language</mark>

# CELL ********************

def translate_text(text, target_language="English"):
    """
    Translates a given text to a target language using the Gemini API.
    
    Args:
        text (str): The input text to be translated.
        target_language (str): The language to translate the text into.
    
    Returns:
        str: The translated text, or an error message if the translation fails.
    """
    # Handle empty input gracefully
    if not text:
        return None
    # Get the configured Gemini model
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    # Function to generate content from the model with a retry strategy.
    @retry(
        wait=wait_exponential(multiplier=3, min=5, max=90),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(ResourceExhausted)
    )
    def _generate_content_with_retry(prompt):
        response = model.generate_content(prompt)
        return response.text
    # Call the API with the retry mechanism.
    try:
        prompt = f"Translate the following text to {target_language}: '{text}'. Only return the translated text, with no additional commentary."
        response_text = _generate_content_with_retry(prompt)
        return response_text
    except Exception as e:
        return "API_ERROR"

translate_udf = udf(translate_text, StringType())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### demo translate "product_review" to English

# CELL ********************

df_translate = df.select("product_name","product_review")
display(df_translate)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_translate = df_translate.withColumn("GenAI_translated_product_review", translate_udf('product_review', lit('english')))
display(df_translate)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## <mark>Define function to detect sentiment of a given text and returns a score between 0 and 1. Scores closer to 1 are positive, closer to 0 are negative.</mark>

# CELL ********************

def detect_sentiment(text):
    """
    Detects the sentiment of a given text and returns a score between 0 and 1.
    Scores closer to 1 are positive, closer to 0 are negative.
    
    Args:
        text (str): The text to analyze for sentiment.
        
    Returns:
        float: The sentiment score (0-1), or None if analysis fails.
    """
    # Handle empty input gracefully
    if not text:
        return None
    # Get the configured Gemini model
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    # Function to generate content from the model with a retry strategy.
    @retry(
        wait=wait_exponential(multiplier=3, min=5, max=90),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(ResourceExhausted)
    )
    def _generate_content_with_retry(prompt):
        response = model.generate_content(prompt)
        return response.text
    # Call the API with the retry mechanism.
    try:
        prompt = (f"Analyze the sentiment of the following text and return a numerical score "
                  f"between 0.0 (very negative) and 1.0 (very positive). "
                  f"Do not include any other text besides the score.\nText: '{text}'")
        response_text = _generate_content_with_retry(prompt)
        # Extract the score from the response and convert it to a float.
        score = float(response_text.strip())
        return score
    except Exception as e:
        return "API_ERROR"

sentiment_udf = udf(detect_sentiment, FloatType())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### demo detect of "product_review". By scoring from 0 to 1. Scores closer to 1 are positive, closer to 0 are negative.

# CELL ********************

df_sentiment = df_translate.withColumn("GenAI_sentiment_product_review", sentiment_udf('GenAI_translated_product_review'))
display(df_sentiment)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## <mark>Define function to corrects the spelling, grammar, and punctuation of a given text.</mark>

# CELL ********************

def correct_text(text):
    """
    Corrects the spelling, grammar, and punctuation of a given text.
    
    Args:
        text (str): The text to be corrected.
        
    Returns:
        str: The corrected text, or an error message if the correction fails.
    """
    # Handle empty input gracefully
    if not text:
        return None
    # Get the configured Gemini model
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    # Function to generate content from the model with a retry strategy.
    @retry(
        wait=wait_exponential(multiplier=3, min=5, max=90),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(ResourceExhausted)
    )
    def _generate_content_with_retry(prompt):
        response = model.generate_content(prompt)
        return response.text
    # Call the API with the retry mechanism.
    try:
        prompt = (f"Correct the spelling, grammar, and punctuation of the following text. "
                  f"Only return the corrected text, with no additional commentary.\nText: '{text}'")
        response_text = _generate_content_with_retry(prompt)
        return response_text
    except Exception as e:
        return "API_ERROR"

correction_udf = udf(correct_text, StringType())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### demo correct grammar of "product_properties"

# CELL ********************

df_correct_text = df.select("product_name","product_properties")
display(df_correct_text)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_correct_text = df_correct_text.withColumn("GenAI_correct_product_properties", correction_udf('product_properties'))
display(df_correct_text)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## <mark>Define function to extracts person names, locations, and organization names from a given text.</mark>

# CELL ********************

def extract_entities(text):
    """
    Extracts person names, locations, and organization names from a given text
    using a structured response from the Gemini API.
    
    The function returns a JSON string containing the extracted entities.
    
    Args:
        text (str): The text to analyze for entities.
        
    Returns:
        str: A JSON string of extracted entities, or an error message if the extraction fails.
    """
    # Handle empty input gracefully
    if not text:
        return None
    # Get the configured Gemini model
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    # Function to generate content from the model with a retry strategy.
    @retry(
        wait=wait_exponential(multiplier=3, min=5, max=90),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(ResourceExhausted)
    )
    def _generate_content_with_retry(prompt):
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "OBJECT",
                    "properties": {
                        "persons": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "locations": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "organizations": {"type": "ARRAY", "items": {"type": "STRING"}}
                    }
                }
            }
        )
        return response.text
    # Call the API with the retry mechanism.
    try:
        prompt = (f"Extract all person names, locations, and organization names from the following text. "
                  f"If no entities of a certain type are found, use an empty list.\nText: '{text}'")
        response_text = _generate_content_with_retry(prompt)
        return response_text
    except Exception as e:
        return "API_ERROR"

entity_extraction_udf = udf(extract_entities, StringType())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### demo extract person names, locations, and organization names of "product_origin" into json format.

# CELL ********************

df_extract_entities = df.select("product_name","product_origin")
display(df_extract_entities)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_extract_entities = df_extract_entities.withColumn("GenAI_entities_product_origin", entity_extraction_udf('product_origin'))
display(df_extract_entities)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## <mark>Define function to assigns a category label to a given text from a provided list of labels.</mark>

# CELL ********************

def categorize_text(text, labels):
    """
    Assigns a category label to a given text from a provided list of labels.
    
    The function returns a single string label.
    
    Args:
        text (str): The text to be categorized.
        labels (list): A list of possible labels to choose from.
        
    Returns:
        str: The category label, or an error message if categorization fails.
    """
    # Handle empty input gracefully
    if (not text) or (not labels):
        return None
    # Get the configured Gemini model
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    # Function to generate content from the model with a retry strategy.
    @retry(
        wait=wait_exponential(multiplier=3, min=5, max=90),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(ResourceExhausted)
    )
    def _generate_content_with_retry(prompt):
        response = model.generate_content(prompt)
        return response.text
    # Call the API with the retry mechanism.
    try:
        
        prompt = (f"Categorize the following text into one of the following labels: "
                  f"{labels}."
                  f"Only return the label.\nText: '{text}'")
        response_text = _generate_content_with_retry(prompt)
        return response_text.strip()
    except Exception as e:
        return "API_ERROR"

category_udf = udf(categorize_text, StringType())


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### demo categorize "product_name" into 'engineering, home appliance, automotive, power tools'

# CELL ********************

df_categorize = df.select("product_name")
display(df_categorize)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_categorize = df_categorize.withColumn("GenAI_labels_product_name", category_udf('product_name', lit('engineering, home appliance, automotive, power tools')))
display(df_categorize)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
