import json

# load json data from file
with open("messages_1.json", "r") as f:
    messages = json.load(f)
    print(messages[0]["role"])
with open("messages_copy.json", "w") as f2:
    json.dump(messages, f2, indent=2, ensure_ascii=False)