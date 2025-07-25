import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, Faithfulness,  FaithfulnesswithHHEM,ResponseRelevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Important: Make sure your main agent logic can be imported
from main import run_agent_streamlit

# --- Setup ---
# Load API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Initialize LLM and embeddings
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
embeddings = OpenAIEmbeddings(api_key=api_key)
evaluator_llm = LangchainLLMWrapper(llm)

print(f"Initialized LLM: {type(llm).__name__}")
print(f"Wrapped LLM: {type(evaluator_llm).__name__}")

# Load product database for context
csv_path = os.path.join("../data", "meta_amazon_review_pets.csv")
if not os.path.exists(csv_path):
    raise ValueError(f"Product database not found at {csv_path}")
df = pd.read_csv(csv_path)
product_db_context = "Available products in database:\n" + "\n".join(
    f"- {row['title']} (Price: ${row['price']}, Rating: {row['average_rating']})"
    for _, row in df.head(10).iterrows()  # Include first 10 products as sample context
)

# --- Main Logic ---
if __name__ == "__main__":
    print("--- Running Ragas Evaluation with LangchainLLMWrapper ---")
    
    test_cases = [
        {
            "test_id": "Cheaper Alternative",
            "product_title": "Fido's Favorite Dog Leash",
            "return_reason": "It's a great leash, but I found it too expensive. I'm looking for a cheaper one.",
        },
        {
            "test_id": "Poor Quality",
            "product_title": "Kitty's Scratch Post",
            "return_reason": "This scratching post fell apart in a week. The quality is terrible.",
        },
        {
            "test_id": "Wrong Size",
            "product_title": "Large Dog Bed",
            "return_reason": "I bought the large, but it's too small for my Great Dane. I need something bigger.",
        }
    ]
    
    evaluation_data = []
    print("--- Generating recommendations for evaluation ---")
    
    for case in test_cases:
        print(f"Running test case: {case['test_id']}...")
        
        # Generate user input and get recommendation
        user_input = f"Title: {case['product_title']}\nReason: {case['return_reason']}"
        recommendation = run_agent_streamlit(
            image_path="dummy.jpg",
            product_title=case["product_title"],
            return_reason=case["return_reason"],
            image_validation_override='valid'
        )
        
        evaluation_data.append({
            'question': user_input,
            'answer': recommendation,
            'contexts': [product_db_context],
            'test_id': case['test_id']
        })

    # Convert to Ragas dataset
    evaluation_dataset = Dataset.from_list(evaluation_data)
    
    # Initialize metrics
    metrics = [
        LLMContextPrecisionWithoutReference(),
        Faithfulness(),      # Measures if the answer uses only information from the provided context
        FaithfulnesswithHHEM(), # Measures if the answer uses only information from the provided context
        ResponseRelevancy()
    ]
    
    print("\n--- Running Ragas Evaluation (this may take a moment) ---")
    print(f"Evaluating {len(evaluation_data)} samples with {len(metrics)} metrics...")
    
    try:
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=metrics,
            llm=evaluator_llm
        )
        
        print("Evaluation completed successfully!")
        
        # Convert results to DataFrame
        results_df = result.to_pandas()
        results_df['test_id'] = [case['test_id'] for case in evaluation_data]
        
        # Debug: Show all available columns
        print(f"\nAvailable columns in results: {list(results_df.columns)}")
        
        # Create a clean results table with expected metric names
        clean_results = pd.DataFrame()
        clean_results['Test Case'] = results_df['test_id']
        
        # Map the actual column names to our expected metrics
        column_mapping = {}
        for col in results_df.columns:
            if 'context_precision' in col.lower() or 'precision' in col.lower():
                column_mapping['Context Precision'] = col
            elif 'faithfulness' in col.lower() and 'hhem' not in col.lower():
                column_mapping['Faithfulness'] = col
            elif 'faithfulness' in col.lower() and 'hhem' in col.lower():
                column_mapping['Faithfulness with HHEM'] = col
            elif 'relevancy' in col.lower() or 'relevance' in col.lower():
                column_mapping['Response Relevancy'] = col
        
        # Add mapped columns to clean results
        for clean_name, actual_col in column_mapping.items():
            if actual_col in results_df.columns:
                clean_results[clean_name] = results_df[actual_col].round(3)
            else:
                clean_results[clean_name] = None
                print(f"Warning: {clean_name} metric not found in results")
        
        print("\n\n--- RAGAS EVALUATION RESULTS TABLE ---")
        print(clean_results.to_string(index=False))
        
        # Show summary statistics
        print("\n--- Summary Statistics ---")
        metric_cols = [col for col in clean_results.columns if col != 'Test Case']
        for col in metric_cols:
            if clean_results[col].notna().any():
                values = clean_results[col].dropna()
                if not values.empty:
                    print(f"{col}:")
                    print(f"  Mean: {values.mean():.3f}")
                    print(f"  Min:  {values.min():.3f}")
                    print(f"  Max:  {values.max():.3f}")
            else:
                print(f"{col}: No valid scores")
        
        # Save clean results
        clean_results.to_csv("ragas_clean_results.csv", index=False)
        print("\nResults saved:")
        print("- Clean table: 'ragas_clean_results.csv'")
        
        # Print metric explanations
        print("\n--- Metric Explanations ---")
        print("Context Precision: Measures if the retrieved product information is precise and relevant")
        print("Faithfulness: Measures if the answer uses only information from the provided context")
        print("Response Relevancy: Measures if the recommendation directly addresses the user's query")
        print("Scale: 0.0 (poor) to 1.0 (excellent) for all metrics")
        
        # Check if all metrics were successfully evaluated
        missing_metrics = []
        if 'Context Precision' not in clean_results.columns or clean_results['Context Precision'].isna().all():
            missing_metrics.append('Context Precision')
        if 'Faithfulness' not in clean_results.columns or clean_results['Faithfulness'].isna().all():
            missing_metrics.append('Faithfulness')
        if 'Response Relevancy' not in clean_results.columns or clean_results['Response Relevancy'].isna().all():
            missing_metrics.append('Response Relevancy')
            
        if missing_metrics:
            print(f"\n⚠️  Warning: The following metrics were not successfully evaluated: {', '.join(missing_metrics)}")
            print("This might be due to:")
            print("- Metric compatibility issues with the current Ragas version")
            print("- Missing required data fields")
            print("- LLM API errors during evaluation")
        else:
            print("\n✅ All 3 metrics evaluated successfully!")
            
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        print("This might be due to compatibility issues or missing data fields.")
        
        # Save the input data for debugging
        input_df = pd.DataFrame(evaluation_data)
        input_df.to_csv("evaluation_input_data.csv", index=False)
        print("Input data saved to 'evaluation_input_data.csv' for debugging.")