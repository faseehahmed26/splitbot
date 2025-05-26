# splitbot

Splitbot is a Python application designed to [**Please describe the main purpose of splitbot here**]. 
It appears to be a bot that interacts via interfaces like Telegram/Whatsapp and can process images (possibly for splitting bills or expenses).

## Features

*   [**List key feature 1, e.g., Expense parsing from images**]
*   [**List key feature 2, e.g., User state management**]
*   [**List key feature 3, e.g., Interaction via Telegram/Whatsapp**]
*   Orchestrates various services (LLM, OCR, Voice).
*   Includes monitoring capabilities.
*   Containerized using Docker for easy deployment.

## Project Structure

```
splitbot/
├── .git/                   # Git version control
├── .venv/                  # Python virtual environment (if used locally)
├── .vscode/                # VS Code editor specific settings
├── config/                 # Configuration files (e.g., settings.py)
├── core/                   # Core logic (e.g., processor.py, state_manager.py)
├── interfaces/             # Communication interfaces (e.g., telegram.py, whatsapp.py)
├── logs/                   # Application logs
├── monitoring/             # Monitoring setup (e.g., langfuse_client.py)
├── services/               # External service integrations (llm, ocr, voice)
├── tests/                  # Automated tests
├── utils/                  # Utility scripts or modules
├── .gitignore              # Specifies intentionally untracked files that Git should ignore
├── cloud_vision_api_key.json # API key for Google Cloud Vision (ensure this is not committed if sensitive)
├── Dockerfile              # Instructions to build the Docker image
├── docker-compose.yml      # Defines and runs multi-container Docker applications
├── fallback_state.json     # Fallback state storage (likely for development/testing)
├── gen-lang-client.json    # Potentially for language client generation or configuration
├── main.py                 # Main application entry point
├── README.md               # This file
└── requirements.txt        # Python package dependencies
```

## Getting Started

### Prerequisites

*   Python 3.x
*   Pip (Python package installer)
*   Docker (optional, for containerized deployment)
*   Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/faseehahmed26/splitbot.git
    cd splitbot
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `\.venv\\Scripts\\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration:**
    *   Set up necessary API keys and configurations. This might involve:
        *   Copying a template configuration file (e.g., `config/settings.example.py` to `config/settings.py`) and filling in your details.
        *   Setting environment variables.
        *   Ensure `cloud_vision_api_key.json` is correctly placed and configured if you intend to use Google Cloud Vision. **Warning: Do not commit sensitive API keys directly to your repository if it's public. Use environment variables or a git-ignored configuration file.**

5.  **Database/Services:**
    *   The project mentions Redis in logs. Ensure Redis is running and accessible if it's a required dependency.
    *   Set up other external services as needed.

### Running the Application

*   **Locally:**
    ```bash
    python main.py
    ```
    (Or the specific command to run your application)

*   **With Docker (if `docker-compose.yml` is configured):**
    ```bash
    docker-compose up --build
    ```

*   **With Docker (using `Dockerfile` directly):**
    ```bash
    docker build -t splitbot .
    docker run [OPTIONS] splitbot
    ```
    (You might need to pass environment variables or map ports in `[OPTIONS]`)


## Usage

[**Provide instructions on how to use the bot/application. For example:**]
*   How to interact with it (e.g., commands for the Telegram bot).
*   Examples of input and expected output.

## Testing

The `tests/` directory suggests that tests are available.
[**Provide instructions on how to run the tests, e.g.:**]
```bash
# Example:
# pytest
# or
# python -m unittest discover tests
```

## Contributing

[**If you are open to contributions, provide guidelines here. For example:**]
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

## License

[**Specify the license for your project, e.g., MIT License. If you haven't chosen one, you might consider adding one.**]

---

**Note:** Please fill in the bracketed placeholders `[**...**]` with specific details about your project. You might also want to add sections like "Deployment", "API Reference" (if applicable), or "Troubleshooting". 