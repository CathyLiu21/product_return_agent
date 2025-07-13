# Product Return Agent

A conversational agent for product returns and recommendations, built with Python, LangGraph, and Streamlit. The agent helps users upload product images, validates them, collects return reasons, and recommends better products from a metadata dataset or via web search.

---

## Features
- **Image Upload & Validation**: Detects valid, AI-generated, or photoshopped images.
- **Conversational UI**: Collects product title and return reason from the user.
- **Metadata-Based Recommendation**: Finds similar or better products from a local dataset.
- **Manual Amazon Search**: If no suitable product is found locally, or if you're not satisfied with the recommendation, you can click a button to search on Amazon.
- **Graceful Error Handling**: If the web search is blocked or fails, the app provides a direct link for you to continue the search manually.
- **Streamlit UI**: User-friendly chatbot interface with test mode for easy testing.
- **Powered by MCP**: The web search tool is provided via an external MCP (Multi-turn Conversation Protocol) server.

---

## Project Structure (Current)

```
product_return_agent/
├── data/
│   ├── amazon_review_pets.csv
│   ├── meta_amazon_review_pets.csv
│   └── ...
├── src/
│   ├── main.py
│   ├── extract_pet_review_sample.py
│   ├── product_return_agent_ui.app.py
│   └── ...
├── test/
│   ├── test_agent.py
│   ├── cli_agent_test.py
│   ├── view_data.ipynb
│   └── ...
├── requirements.txt
├── README.md
├── .gitignore
├── workflow.mmd
├── workflow.png
├── venv/
│   └── ... (virtual environment files)
└── ...
```

---

## Setup

1. **Clone the repository**
2. **Create and activate a Python 3.10+ virtual environment**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up API keys**
   - Create a `.env` file with your OpenAI and Tavily API keys:
     ```env
     OPENAI_API_KEY=your_openai_key
     ```
5. **Place all data files in the `data/` directory**
   - Example: `data/meta_amazon_review_pets.csv`, `data/amazon_review_pets.csv`, etc.

---

## Usage

### **Run the Streamlit UI**
```bash
streamlit run src/product_return_agent_ui.py
```
- Upload a product image or use test mode.
- Enter product title and reason for return.
- View recommendations in a chat-like interface.
- You can also search on Amazon and ask the agent to find products if you don't like the producted recommended.
- **Note:** All data files must be in the `data/` directory.

### **Web Search with MCP (Model Context Protocol)**

The web search functionality is powered by an external MCP server that exposes web scraping tools. The Streamlit application communicates with this server to perform searches on Amazon.

**1. Install MCP with CLI support**

Before running the MCP server, install the MCP package with CLI support:

```bash
pip install "mcp[cli]"
```

**2. Start the MCP Server**

Before running the Streamlit app, you must start the MCP proxy server in a separate terminal. This server uses the `@mzxrai/mcp-webresearch` tool to visit web pages.

```bash
mcp-proxy --pass-environment npx -- @smithery/cli@latest run @mzxrai/mcp-webresearch --config '{}'
```

When the server starts, it will log the port it is running on. For example:

```
INFO:     Uvicorn running on http://127.0.0.1:52368 (Press CTRL+C to quit)
```
Note: the port number can be different on your machine.

**3. Configure the Port in the Application**

You must ensure the port number in the application code matches the port the MCP server is running on.

-   **Check the MCP server's terminal output** for the running port (e.g., `52368`).
-   **Update the URL** in `src/product_return_agent_ui.py` inside the `fetch_amazon_product_urls` function:

    ```python
    async def fetch_amazon_product_urls(query, mcp_url="http://localhost:52368/sse"):
        # ... function code ...
    ```

**4. Run the Streamlit App**

With the MCP server running, you can now start the Streamlit app. The "Search on Amazon" button will now correctly call the MCP server to fetch results.

### **Automated Test of the Agent**
```bash
python test/test_agent.py
```
- Runs a set of automated test cases for the agent logic with sample input and prints the results for each scenario.
- **Note:** All data files must be in the `data/` directory.

### **Interactive CLI Test of the Agent**
```bash
python test/cli_agent_test.py
```
- Prompts you for image path, product title, and reason for return, then runs the agent and prints the recommendation.
- Useful for manual, human-in-the-loop testing.
- **Note:** All data files must be in the `data/` directory.

---

## Workflow Diagram

The workflow is visualized in `workflow.png`:

![Workflow Diagram](workflow.png)


---

## Notes
- The agent uses fuzzy string matching for robust product title clustering.
- Test mode allows you to skip image upload and simulate different validation results.
- For best results, ensure your virtual environment is activated before running scripts.
- Both automated and interactive CLI testing are supported via `test_agent.py` and `cli_agent_test.py`.
- All scripts and the UI expect data files to be in the `data/` directory.
- If no suitable product is found locally, the agent will automatically perform a web search and return a clearly formatted web result as a fallback.

---

## License
MIT 