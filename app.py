import os
from flask import Flask, request, jsonify, send_from_directory
import requests, json, re, time
from eth_account import Account
from eth_account.messages import encode_typed_data

app = Flask(__name__)

# Load full party programmes from file
with open('party_programmes.json', 'r') as f:
    party_programmes = json.load(f)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2-vision:11b"  # adjust as needed

# Global variable to store the plain-text summary
programme_summary_text = ""

# ---------- Private Key Management ----------
PRIVATE_KEY_FILE = "private_key.json"

def load_or_generate_private_key():
    if not os.path.exists(PRIVATE_KEY_FILE):
        # Generate a new private key
        acct = Account.create()
        pk_data = {"private_key": acct.key.hex(), "address": acct.address}
        with open(PRIVATE_KEY_FILE, "w") as f:
            json.dump(pk_data, f)
        print("Generated new private key for address:", pk_data["address"])
        return pk_data["private_key"]
    else:
        with open(PRIVATE_KEY_FILE, "r") as f:
            pk_data = json.load(f)
        print("Loaded existing private key for address:", pk_data["address"])
        return pk_data["private_key"]

PRIVATE_KEY = load_or_generate_private_key()
# ---------- End Private Key Management ----------

def clean_output(text):
    """
    Remove markdown code fences and any <think>...</think> blocks,
    then return the cleaned text.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

def summarize_programmes():
    """
    Ask the LLM to summarize the party programmes into plain text.
    For each party, provide exactly 3 bullet points (using '-' for bullets)
    that capture its key points. The output is plain text with the party name on its own line
    followed by its 3 bullet points.
    """
    prompt = (
        "Summarize the following party programmes. For each party, provide exactly 3 bullet points (using '-' for bullets) "
        "that capture its key points. Format your answer as plain text, listing the party name on its own line, "
        "followed by its 3 bullet points (each on a new line). Do not include any extra explanation or commentary. "
        "Party programmes: " + json.dumps(party_programmes)
    )
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "max_tokens": 300,
                "stream": False
            }
        )
    except Exception as e:
        return "Request to Ollama failed: " + str(e)
    
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        return "Failed to decode JSON from Ollama: " + str(e) + " | Response text: " + response.text

    output = clean_output(data.get("response", ""))
    return output

programme_summary_text = summarize_programmes()
print("Programme summary generated:")
print(programme_summary_text)

def generate_questions():
    """
    Generate 10 Likert-scale questions based on the plain-text programme summary.
    Each question is a policy statement (e.g. "Wealthy people should be taxed at 60%+. Do you agree?")
    with exactly 5 fixed Likert options: "Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree".
    Each question must have an 'id' (unique integer), 'text' (the statement), and an 'options' array.
    IMPORTANT: Return only raw JSON.
    """
    prompt = (
        "Generate a JSON array of 10 Likert-scale questions based on the following programme summary. "
        "The questions are designed to determine which party or political direction the person is tending towards. "
        "Each question must be a policy statement derived from the summary (for example, "
        "\"Wealthy people should be taxed at 60% or more. Do you agree?\") and include the following fixed answer options: "
        "\"Strongly Disagree\", \"Disagree\", \"Neutral\", \"Agree\", \"Strongly Agree\". "
        "Each question object must have an 'id' (a unique integer), 'text' (the statement), and an 'options' array with these 5 choices. "
        "Return only raw JSON without any markdown formatting, explanation, or extra text. "
        "Programme summary: " + programme_summary_text
    )
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "max_tokens": 500,
                "stream": False
            }
        )
    except Exception as e:
        return {"error": "Request to Ollama failed: " + str(e)}
    
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        return {"error": "Failed to decode JSON from Ollama: " + str(e) +
                " | Response text: " + response.text}
    
    output = clean_output(data.get("response", ""))
    if not output:
        return {"error": "Ollama returned an empty response."}
    try:
        questions = json.loads(output)
        return questions
    except json.JSONDecodeError as e:
        return {"error": "Failed to parse JSON output: " + str(e) +
                " | Output was: " + output}

def evaluate_answers(user_answers):
    """
    Determine the party that best matches the user's views using the plain-text programme summary and user answers.
    The LLM must return valid JSON in the format: {"best_party": "PartyName"}.
    """
    prompt = (
        "Determine the party that best matches the user's views based on the following data. "
        "Return only valid JSON in the format: {\"best_party\": \"PartyName\"}. "
        "IMPORTANT: Output only raw JSON without markdown formatting, explanation, or extra text. "
        "Programme summary: " + programme_summary_text +
        " User answers: " + json.dumps(user_answers)
    )
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "max_tokens": 200,
                "stream": False
            }
        )
    except Exception as e:
        return {"error": "Request to Ollama failed: " + str(e)}
    
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        return {"error": "Failed to decode JSON from Ollama: " + str(e) +
                " | Response text: " + response.text}
    
    output = clean_output(data.get("response", ""))
    if not output:
        return {"error": "Ollama returned an empty response."}
    try:
        result = json.loads(output)
        return result
    except json.JSONDecodeError as e:
        return {"error": "Failed to parse JSON output: " + str(e) +
                " | Output was: " + output}

@app.route('/api/questions', methods=['GET'])
def get_questions():
    questions = generate_questions()
    return jsonify(questions)

@app.route('/api/party_programmes', methods=['GET'])
def get_programmes():
    return jsonify(party_programmes)

@app.route('/api/summaries', methods=['GET'])
def get_summaries():
    return jsonify({"summary": programme_summary_text})

@app.route('/api/submit_answers', methods=['POST'])
def submit_answers():
    user_answers = request.json.get('answers', {})
    result = evaluate_answers(user_answers)
    return jsonify(result)

@app.route('/api/vote', methods=['POST'])
def vote():
    data = request.json
    # Use provided choice or default to 3 (Neutral)
    choice = data.get("choice", 3)
    reason = data.get("reason", "")
    timestamp = str(int(time.time()))
    
    # Domain name must match your Snapshot space name.
    domain_data = {
        "name": "realgovernment.eth",
        "version": "0.1.4",
        "chainId": 1
    }
    message_types = {
        "Vote": [
            {"name": "version", "type": "string"},
            {"name": "timestamp", "type": "string"},
            {"name": "space", "type": "string"},
            {"name": "type", "type": "string"},
            {"name": "payload", "type": "VotePayload"}
        ],
        "VotePayload": [
            {"name": "proposal", "type": "string"},
            {"name": "choice", "type": "uint32"},
            {"name": "reason", "type": "string"}
        ]
    }
    message_data = {
        "version": "0.1.4",
        "timestamp": timestamp,
        "space": "realgovernment.eth",
        "type": "vote",
        "payload": {
            "proposal": "0xdab100fd43291674206382dff3ed99f003b2722a3aaa125e907d6314d5473200",
            "choice": int(choice),
            "reason": reason
        }
    }
    
    try:
        account = Account.from_key(PRIVATE_KEY)
        # Sign using the three-argument form to produce an EIP-712 signature.
        signed = Account.sign_typed_data(PRIVATE_KEY, domain_data, message_types, message_data)
    except Exception as e:
        return jsonify({"error": "Signing failed: " + str(e)}), 500

    vote_payload = {
        "address": account.address,
        # Sending the message as a JSON string is acceptable if it matches what Snapshot expects.
        "msg": json.dumps(message_data),
        "sig": signed.signature.hex(),
        "version": "0.1.4",
        "type": "vote"
    }

    try:
        response = requests.post("https://hub.snapshot.org/api/message", json=vote_payload)
        result = response.json()
    except Exception as e:
        return jsonify({"error": "Failed to cast vote: " + str(e)}), 500

    return jsonify(result)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
