import urllib.request
import json
import time
import sys

def post_json(url, data=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8') if data is not None else b'',
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def get_json(url):
    req = urllib.request.Request(url, method='GET')
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def main():
    base_url = "http://127.0.0.1:8080/api"
    print("Checking active agents...")
    try:
        agents = get_json(f"{base_url}/agents")
        agent_ids = [a['id'] for a in agents[:2]]
        print(f"Using Agent IDs: {agent_ids}")
    except Exception as e:
        print(f"Error fetching agents: {e}")
        sys.exit(1)
        
    topic = "Çocuk yapmak ahlaken doğru bi davranış mı ?"
    print(f"Creating debate on: '{topic}'...")
    try:
        debate = post_json(f"{base_url}/debates", {
            "topic": topic,
            "agent_ids": agent_ids,
            "max_rounds": 5
        })
        debate_id = debate['id']
        print(f"Created Debate ID: {debate_id}")
    except Exception as e:
        print(f"Error creating debate: {e}")
        sys.exit(1)

    print("Starting debate (Turn 1)...")
    try:
        post_json(f"{base_url}/debates/{debate_id}/start")
        print("Debate started successfully.")
    except Exception as e:
        print(f"Error starting debate: {e}")
        sys.exit(1)

    # Advance remaining 9 turns
    for turn in range(2, 11):
        print(f"Advancing turn {turn}/10...")
        try:
            post_json(f"{base_url}/debates/{debate_id}/advance")
            # Wait a few seconds for the LLM to complete generation
            time.sleep(3)
        except Exception as e:
            print(f"Error advancing turn {turn}: {e}")
            sys.exit(1)

    print("Debate turns completed. Triggering ModernBERT analysis...")
    try:
        # The /analyze endpoint runs full ModernBERT extraction and classification
        analysis_result = post_json(f"{base_url}/debates/{debate_id}/analyze")
        print("ModernBERT analysis triggered and completed.")
    except Exception as e:
        print(f"Error running ModernBERT analysis: {e}")
        sys.exit(1)

    print("Fetching argument map data...")
    try:
        map_data = get_json(f"{base_url}/debates/{debate_id}/argument-map")
        
        # Save to file
        output_file = "morality_debate_analysis.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(map_data, f, ensure_ascii=False, indent=2)
            
        print(f"Argument map and analysis results saved to {output_file}")
    except Exception as e:
        print(f"Error fetching argument map: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
