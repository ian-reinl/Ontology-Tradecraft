import csv
import anthropic
from dotenv import load_dotenv

load_dotenv()                   # puts ANTHROPIC_API_KEY into the environment
client = anthropic.Anthropic()  # finds the key on its own

PROMPT = """Rewrite this CCO textual definition in the form
"A/An <label> is a/an <genus> that <differentia>."
Do not add or remove meaning. Expand abbreviations. Keep the word "designed" if present.
Label: {label}
Definition: {definition}
Return only the rewritten definition."""

def normalize(label, definition):
    msg = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": PROMPT.format(label=label, definition=definition)}],
    )
    return next(b.text for b in msg.content if b.type == "text").strip()

def main(in_path="data/definitions.csv", out_path="data/definitions_enriched.csv"):
    with open(in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["definition_enriched"] = normalize(row["label"], row["definition"])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["IRI", "label", "definition", "definition_enriched"])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()