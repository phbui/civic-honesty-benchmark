import json, random
random.seed(20261019)
Q = [json.loads(l) for l in open("results/questions_canonical.jsonl")]
by = {}
for q in Q: by.setdefault(q["cls"], []).append(q)
take = {"answerable": 33, "unanswerable": 33, "unreliable": 24}
sub = []
for c, n in take.items():
    sub += random.sample(by[c], n)
random.shuffle(sub)
ids = {q["id"] for q in sub}
with open("results/questions_groq_subset90.jsonl", "w") as f:
    for q in sub: f.write(json.dumps(q) + "\n")
with open("results/gold_2026-10-19_groq_subset90.jsonl", "w") as f:
    for l in open("results/gold_2026-10-19.jsonl"):
        r = json.loads(l)
        if r.get("question_id", r.get("id")) in ids: f.write(l)
print({c: sum(1 for q in sub if q["cls"] == c) for c in take}, len(sub))
