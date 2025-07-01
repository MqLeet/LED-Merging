import json
result = {}
with open("question.jsonl","r") as fr:
    for file in fr.readlines():
        d = json.loads(file)
        result[d["question_id"]] = [d["turns"][0]]
with open("test_cases.json","w") as fw:
    json.dump(result, fw, indent=4)