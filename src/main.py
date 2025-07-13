# -*- coding: utf-8 -*-
"""
Product Return Agent using LangGraph
Workflow:
1. Ask user to upload a picture of the product.
2. Validate the picture: valid, AI-generated, or photoshopped.
   - If AI-generated: return 'AI-generated picture. please re-upload picture'.
   - If photoshopped: return 'photoshopped picture, please re-upload the picture'.
   - If valid: move to next step.
3. Ask user to input product title and reasons to return this product in text.
4. Based on user's provided return reasons, search similar product in 'amazon_review_pets.csv'.
   - If user wants a cheaper product, search by 'price' and recommend a similar but cheaper product.
   - If cannot find a better product, call a tool to do websearch and return a better product.
"""
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

import os
import sys
import logging
import json
import pandas as pd
from typing_extensions import TypedDict
from typing import Annotated, Sequence
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
#from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
import re
import openai
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
#from langchain_ollama import ChatOllama
#from tavily import TavilyClient
from fuzzywuzzy import fuzz
#from image_deter import detect  # or however your model is imported

from dotenv import load_dotenv

pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

load_dotenv()  # This will load variables from .env into the environment

# Now you can access your keys like this:
openai_api_key = os.getenv("OPENAI_API_KEY")
#tavily_api_key = os.getenv("TAVILY_API_KEY")

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- State Definition ---
class ReturnAgentState(TypedDict):
    user_id: str
    image_path: str
    image_validation: str
    product_title: str
    return_reason: str
    recommendation: str
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Comprehensive list of return reason categories
RETURN_REASON_CATEGORIES = [
    "cheaper", "wrong size", "not as described", "defective", "changed mind",
    "late delivery", "wrong item", "poor quality", "missing parts", "better alternative",
    "expired", "other"
]

# --- Node 1: Ask for Image Upload ---
def image_upload_node(state: ReturnAgentState):
    # Use image_path from state if provided (e.g., from Streamlit UI)
    image_path = state.get("image_path", "")
    interactive = state.get("interactive", False)
    if not image_path and interactive:
        image_path = input("Please upload a picture of the product (enter file path): ")
    state["image_path"] = image_path
    state["messages"] = []
    return state

# --- Node 2: Validate Image ---
def image_validation_node(state: ReturnAgentState):
    # If image_validation is already set (e.g., for testing), use the provided label and skip detection
    if state.get("image_validation"):
        return state
    # Otherwise, call the detect function (if available)
    try:
        from image_deter import detect  # Only import if needed
        image_path = state.get("image_path", "")
        validation_result = detect(image_path)
        state["image_validation"] = validation_result
    except ImportError:
        # If detect is not available and no label is set, set to 'unknown' or raise an error
        state["image_validation"] = "unknown"
    return state

# --- Node 3: Ask for Product Info ---
def product_info_node(state: ReturnAgentState):
    # Use values from state if provided (e.g., from Streamlit UI)
    title = state.get("product_title", "")
    reason = state.get("return_reason", "")
    interactive = state.get("interactive", False)
    if not title and interactive:
        title = input("Enter the product title: ")
    if not reason and interactive:
        reason = input("Enter the reason for returning the product: ")
    # LLM intent extraction
    intent = llm_extract_intent(reason)
    state["product_title"] = title
    state["return_reason"] = reason
    state["intent"] = intent
    return state

# --- Node 4: Search CSV ---
def csv_search_node(state: ReturnAgentState):
    title = state.get("product_title", "")
    reason = state.get("return_reason", "")
    intent = state.get("intent", "")
    csv_path = os.path.join("data", "meta_amazon_review_pets.csv")
    if not os.path.exists(csv_path):
        state["recommendation"] = f"Data file {csv_path} not found."
        return state
    df = pd.read_csv(csv_path)
    # Only consider products with rating >= 4.0
    df = df[pd.to_numeric(df['average_rating'], errors='coerce') >= 4.0]
    if df.empty:
        state["recommendation"] = ""
        return state
    # Step 1: Compute fuzzy similarity between user title and all product titles
    df["title_fuzz"] = df["title"].apply(lambda t: fuzz.token_set_ratio(title, str(t)))
    best_title_fuzz = df["title_fuzz"].max()
    if best_title_fuzz < 50 or pd.isna(best_title_fuzz):
        state["recommendation"] = ""
        return state
    # Step 2: Cluster products with similar titles (threshold, e.g., 80)
    title_threshold = 80
    matched_title_df = df[df["title_fuzz"] >= title_threshold]
    if matched_title_df.empty:
        state["recommendation"] = ""
        return state
    # Step 3: Compute similarity between user_text and metadata fields
    user_text = f"{title} {reason}"
    def get_product_metadata_text(row):
        fields = [
            str(row.get('title', '')),
            str(row.get('features', '')),
            str(row.get('description', '')),
            str(row.get('categories', '')),
            str(row.get('details', '')),
        ]
        return ' '.join(fields)
    matched_title_df["jaccard"] = matched_title_df.apply(lambda row: jaccard_similarity(user_text, get_product_metadata_text(row)), axis=1)
    # Step 4: Recommend all products with similarity >= 0.2 and average_rating >= 4.0
    threshold = 0.2
    recs = matched_title_df[matched_title_df["jaccard"] >= threshold]
    if recs.empty:
        # If none meet the threshold, recommend the single most similar product
        best_row = matched_title_df.sort_values(by="jaccard", ascending=False).iloc[0]
        recommendations = [best_row]
    else:
        recommendations = [row for _, row in recs.iterrows()]
    # Format recommendations as text blocks
    formatted_blocks = []
    for row in recommendations:
        features = row.get('features', '')
        # Try to extract first 1-2 bullet points or up to 300 chars
        if isinstance(features, str) and features.startswith("["):
            import ast
            try:
                features_list = ast.literal_eval(features)
                features_str = '\n'.join(f"- {f}" for f in features_list[:2])
            except Exception:
                features_str = features[:300]
        else:
            features_str = features[:300]
        block = (
            f"Title: {row.get('title', '')}\n"
            f"Average Rating: {row.get('average_rating', '')}\n"
            f"Features:\n{features_str}\n"
            f"- Price: ${row.get('price', '')}\n"
            f"- ASIN: {row.get('parent_asin', '')}"
        )
        formatted_blocks.append(block)
    state["recommendation"] = '\n\n'.join(formatted_blocks)
    return state

# --- Node 5: Web Search Fallback ---
def web_search_node(state: ReturnAgentState):
    state["recommendation"] = (
        "No similar product was found in the database. "
        "You may click the Amazon search button below to do a web search."
    )
    return state

# --- End Node: Output Recommendation ---
def end_node(state: ReturnAgentState):
    # Do not print anything here; let the caller handle output
    return state

# --- Conditional Edges ---
def should_validate_image(state: ReturnAgentState):
    return "validate"

def should_ask_product_info(state: ReturnAgentState):
    if state["image_validation"] == 'valid':
        return "info"
    else:
        return "end"

def should_search_web(state: ReturnAgentState):
    if state["recommendation"]:
        return "end"
    else:
        return "web"

# --- Build Graph ---
workflow = StateGraph(ReturnAgentState)
workflow.add_node("upload", image_upload_node)
workflow.add_node("validate", image_validation_node)
workflow.add_node("info", product_info_node)
workflow.add_node("csv", csv_search_node)
workflow.add_node("web", web_search_node)
workflow.add_node("end", end_node)

workflow.set_entry_point("upload")
workflow.add_edge("upload", "validate")
workflow.add_conditional_edges("validate", should_ask_product_info, {"info": "info", "end": "end"})
workflow.add_edge("info", "csv")
workflow.add_conditional_edges("csv", should_search_web, {"end": "end", "web": "web"})
workflow.add_edge("web", "end")

# Compile the graph
graph = workflow.compile(checkpointer=MemorySaver())

# Display the workflow diagram
# display(
#     Image(
#         graph.get_graph().draw_mermaid_png(
#             draw_method=MermaidDrawMethod.API,
#         )
#     )
# )

def run_agent():
    initial_state = {
        "user_id": "user1",
        "image_path": "",
        "image_validation": "",
        "product_title": "",
        "return_reason": "",
        "recommendation": "",
        "messages": [],
        "interactive": True,
    }
    config = {"configurable": {"thread_id": "test-thread"}}
    result_state = graph.invoke(initial_state, config)

def llm_extract_intent(reason):
    prompt = (
        f"Classify the following product return reason into one of these intents: {', '.join(RETURN_REASON_CATEGORIES)}. "
        "Return only the intent word.\n"
        f"Reason: {reason}\nIntent:"
    )
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0,
    )
    return response.choices[0].message.content.strip().lower()

def jaccard_similarity(str1, str2):
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform([str1, str2])
    if X.shape[1] == 0:
        return 0.0
    return jaccard_score(X.toarray()[0], X.toarray()[1])

def validate_image(image_path):
    state = {
        "image_path": image_path,
        "image_validation": "",
        "user_id": "user1",
        "product_title": "",
        "return_reason": "",
        "recommendation": "",
        "messages": [],
        "interactive": True,
    }
    state = image_validation_node(state)
    return state["image_validation"]

def run_agent_streamlit(image_path, product_title, return_reason, image_validation_override=None):
    # If testing with override, skip image upload/validation and use the provided label
    if image_validation_override is not None:
        if image_validation_override == 'ai-generated':
            return 'Invalid picture: AI-generated. Please re-upload a valid picture.'
        elif image_validation_override == 'photoshopped':
            return 'Invalid picture: Photoshopped. Please re-upload a valid picture.'
        elif image_validation_override != 'valid':
            return 'Invalid picture. Please re-upload a valid picture.'
        initial_state = {
            "user_id": "user1",
            "image_path": image_path or "dummy.jpg",
            "image_validation": image_validation_override,
            "product_title": product_title,
            "return_reason": return_reason,
            "recommendation": "",
            "messages": [],
            "interactive": True,
        }
        config = {"configurable": {"thread_id": "test-thread"}}
        result_state = graph.invoke(initial_state, config)
        return result_state.get("recommendation", "No recommendation found.")
    # Normal app usage: require image upload and validation
    initial_state = {
        "user_id": "user1",
        "image_path": image_path,
        "image_validation": "",
        "product_title": product_title,
        "return_reason": return_reason,
        "recommendation": "",
        "messages": [],
        "interactive": True,
    }
    # Run image validation node if not overridden
    initial_state = image_validation_node(initial_state)
    config = {"configurable": {"thread_id": "app-thread"}}
    result_state = graph.invoke(initial_state, config)
    return result_state.get("recommendation", "No recommendation found.")

if __name__ == "__main__":
    run_agent()
