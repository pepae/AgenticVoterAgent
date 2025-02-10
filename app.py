import os
from flask import Flask, request, jsonify, send_from_directory
import requests, json, re, time

app = Flask(__name__)

# Load party programmes from file (each object must have a "party" key)
with open('party_programmes.json', 'r') as f:
    party_programmes = json.load(f)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2-vision:11b"  # adjust as needed

# Global variable to store the plain-text summary
programme_summary_text = ""

def clean_output(text):
    """Remove markdown code fences and clean JSON output."""
    # Log raw response for debugging
    print("DEBUG: Raw LLM output:", text)

    # Remove code fences (e.g., ```json and ```)
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)

    # Log cleaned response for verification
    print("DEBUG: Cleaned output:", text)

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
    except Exception as e:
        return "Failed to decode JSON from Ollama: " + str(e) + " | Response text: " + response.text

    output = clean_output(data.get("response", ""))
    return output

programme_summary_text = summarize_programmes()
print("Programme summary generated:")
print(programme_summary_text)

def generate_questions():
    """
    Generate 10 Likert-scale questions based on the programme summary.
    Each question is a policy statement (e.g. "Wealthy people should be taxed at 60%+. Do you agree?")
    with exactly 5 fixed Likert options: "Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree".
    Each question must have an 'id', 'text', and an 'options' array.
    """
    prompt = (
        "Generate a JSON array of 10 Likert-scale questions based on the following programme summary. "
        "The questions are designed to determine which party or political direction the person is tending towards. "
        "Each question must be a policy statement derived from the summary (for example, "
        "\"Wealthy people should be taxed at 60% or more. Do you agree?\") and include the fixed answer options: "
        "\"Strongly Disagree\", \"Disagree\", \"Neutral\", \"Agree\", \"Strongly Agree\". "
        "Each question object must have an 'id' (unique integer), 'text' (the statement), and an 'options' array with these 5 choices. "
        "Return only raw JSON without extra text. This is important, return only the JSON, no other text! Programme summary: " + programme_summary_text
    )
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "max_tokens": 500, "stream": False}
        )
    except Exception as e:
        return {"error": "Request to Ollama failed: " + str(e)}
    
    try:
        data = response.json()
    except Exception as e:
        return {"error": "Failed to decode JSON from Ollama: " + str(e) + " | Response text: " + response.text}
    
    output = clean_output(data.get("response", ""))
    if not output:
        return {"error": "Ollama returned an empty response."}
    try:
        questions = json.loads(output)
        return questions
    except Exception as e:
        return {"error": "Failed to parse JSON output: " + str(e) + " | Output was: " + output}

def evaluate_answers(user_answers):
    """
    Determine the party that best matches the user's views using the programme summary and user answers.
    Returns valid JSON in the format: {"best_party": "<PartyName>"}.
    The prompt now lists available parties to guide the LLM.
    """
    available = ", ".join([party["party"] for party in party_programmes if "party" in party])
    prompt = (
        "Based on the following programme summary and user answers, determine which party best matches the user's views. "
        "Available parties: " + available + ". "
        "Output a JSON object exactly in the format: {\"best_party\": \"<PartyName>\"} where <PartyName> is one of the available parties. "
        "Do not include any extra text. "
        "Programme summary: " + programme_summary_text +
        " User answers: " + json.dumps(user_answers)
    )
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "max_tokens": 200, "stream": False}
        )
    except Exception as e:
        return {"error": "Request to Ollama failed: " + str(e)}
    
    try:
        data = response.json()
    except Exception as e:
        return {"error": "Failed to decode JSON from Ollama: " + str(e) + " | Response text: " + response.text}
    
    output = clean_output(data.get("response", ""))
    if not output or output.lower().strip() == "undefined":
        return {"error": "LLM returned undefined or empty response."}
    try:
        result = json.loads(output)
        return result
    except Exception as e:
        return {"error": "Failed to parse JSON output: " + str(e) + " | Output was: " + output}

# --- Custom Voting Endpoint (Pretend Voting App) ---
# In-memory vote tally using party names from party_programmes.
votes = {}
for party in party_programmes:
    if "party" in party:
        votes[party["party"].strip()] = 0

@app.route('/api/vote', methods=['POST'])
def vote():
    data = request.json
    recommended = data.get("best_party")
    if not recommended or recommended.lower().strip() == "undefined":
        return jsonify({"error": "No valid recommended party provided."}), 400
    norm_rec = recommended.lower().strip()
    # Build a mapping from normalized party names to canonical names.
    mapping = {p["party"].lower().strip(): p["party"].strip() for p in party_programmes if "party" in p}
    # Fallback: if "left" appears in the recommended name, assume "Linke"
    if norm_rec not in mapping and "left" in norm_rec:
        for key, value in mapping.items():
            if "linke" in key or "left" in key:
                mapping[norm_rec] = value
                break
    if norm_rec not in mapping:
        return jsonify({"error": "Recommended party not found in mapping"}), 400
    chosen_party = mapping[norm_rec]
    votes[chosen_party] += 1
    return jsonify({"message": f"Vote recorded for {chosen_party}", "votes": votes})

# --- Other Endpoints ---
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

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
