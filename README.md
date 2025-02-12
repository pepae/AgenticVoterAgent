# Agentic Voting Agent

The **Agentic Voting Agent** is a Flask-based web application designed to provide personalized political recommendations. Users answer a series of policy-related questions, and the app analyzes their responses to recommend a political party based on predefined party programmes. The app also simulates an automated voting system using a custom backend.

---

## **Features**
- Dynamic questionnaire with Likert-scale questions.
- Party programme summaries generated by a large language model (LLM).
- Automated analysis of user responses to recommend a political party.
- Custom pretend voting mechanism to simulate election voting.
- Detailed status updates and logs displayed on the frontend.
- Asynchronous loading of questionnaire and vote casting.
- Built-in debugging and error-handling functionality.

---

## **Project Structure**
```
AgenticVotingAdvisor/
│
├── app.py                 # Core Flask backend
├── static/
│   └── index.html          # Main frontend interface
├── party_programmes.json   # JSON file containing detailed party programmes
└── private_key.json        # Auto-generated private key for mock signing
```

---

## **Dependencies**

The project requires Python 3.x and the following packages:
- Flask: Web framework
- Requests: For making HTTP requests
- Eth-account: Ethereum account utilities (only used for legacy code handling)
  
You can install dependencies using:

```sh
pip install Flask requests eth-account
```

Ensure your Python environment is active when installing.

---

## **Setup Instructions**

1. **Clone the Repository**  
   Clone this project to your local machine:
   ```sh
   git clone https://github.com/your-repo-name/agentic-voting-advisor.git
   cd agentic-voting-advisor
   ```

2. **Install Dependencies**  
   Run the following command to install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. **Configure Party Programmes**  
   Modify the `party_programmes.json` file to include the desired political parties and their programme details.

4. **Run the Application**  
   Launch the app with:
   ```sh
   python app.py
   ```
   The application will be available at `http://localhost:5000`.

---

## **Usage**

### **Frontend Interface**
The interface consists of two main sections:
- **Questionnaire**: Users answer multiple-choice questions with Likert-scale options.
- **Vote Status**: Displays the recommended party and vote results, with real-time status updates.

### **Steps**
1. The questionnaire loads automatically on the homepage.
2. Users complete the questionnaire and click the **Submit Answers** button.
3. The app evaluates responses and displays the recommended party.
4. A simulated vote is cast for the recommended party, with results shown in the status log.

---

## **Backend Endpoints**

| Endpoint               | Method | Description                                   |
|------------------------|--------|-----------------------------------------------|
| `/api/questions`       | GET    | Fetches the questionnaire questions.          |
| `/api/party_programmes`| GET    | Returns the full list of party programmes.     |
| `/api/summaries`       | GET    | Returns the summarized party programmes.       |
| `/api/submit_answers`  | POST   | Submits user answers for evaluation.           |
| `/api/vote`            | POST   | Casts a vote for the recommended party.        |

---

## **Customization**

### **Party Programmes**
Update `party_programmes.json` to modify the available parties and their programme details.

### **LLM Model**
Change the LLM settings in the `app.py` file by modifying:
```python
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2-vision:11b"
```
Adjust the URL or model name to fit your LLM deployment.

---

## **Troubleshooting**

### **Common Issues**
1. **"Failed to parse JSON output" Error**  
   Ensure that the `clean_output` function properly removes code fences and `<think>` tags. Verify that the LLM returns valid JSON.

2. **Empty Questionnaire**  
   If the questionnaire is empty, check for errors in the LLM response or prompt configuration.

3. **Frontend Not Loading**  
   Ensure the Flask app is running and accessible at `http://localhost:5000`.

---

## **Contributing**

Feel free to fork the repository and submit pull requests. Contributions are welcome!

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgements**

- Built using Flask and other open-source libraries.
- Uses LLM integration for dynamic content generation.

---

**Enjoy using the Agentic Voting Advisor!**