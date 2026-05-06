import re
from dotenv import load_dotenv
from langsmith import evaluate, Client

load_dotenv()

from tools import add, multiply, weather, web_search, query_db

client = Client()


def get_or_create_dataset(name, description, examples):
    try:
        ds = client.create_dataset(name, description=description)
    except Exception:
        ds = client.read_dataset(dataset_name=name)
        for ex in client.list_examples(dataset_id=ds.id):
            client.delete_example(ex.id)
    client.create_examples(dataset_id=ds.id, examples=examples)


get_or_create_dataset("eval-math",       "Math tool",     [{"inputs": {"a": 10,  "b": 5},                                                      "outputs": {"output": "15"}}])
get_or_create_dataset("eval-convert",    "Convert tool",  [{"inputs": {"a": 100, "b": 0.621371},                                               "outputs": {"output": "62.14"}}])
get_or_create_dataset("eval-weather",    "Weather tool",  [{"inputs": {"city": "London"},                                                       "outputs": {"output": "°C"}}])
get_or_create_dataset("eval-web-search", "Web search",    [{"inputs": {"query": "capital of France"},                                           "outputs": {"output": "Paris"}}])
get_or_create_dataset("eval-shelter",    "Shelter tool",  [{"inputs": {"query": "SELECT COUNT(*) FROM animals WHERE type='dog'"},                "outputs": {"output": "1"}}])


# ── Targets ───────────────────────────────────────────────────────────────────

def math_target(inputs):    return {"output": str(add.invoke({"a": inputs["a"], "b": inputs["b"]}))}
def convert_target(inputs): return {"output": str(multiply.invoke({"a": inputs["a"], "b": inputs["b"]}))}
def weather_target(inputs): return {"output": weather.invoke({"city": inputs["city"]})}
def search_target(inputs):  return {"output": web_search.invoke({"query": inputs["query"]})}
def shelter_target(inputs): return {"output": query_db.invoke({"query": inputs["query"]})}


# ── Code evaluators ───────────────────────────────────────────────────────────

def math_evaluator(inputs: dict, reference_outputs: dict, outputs: dict) -> dict:
    expected = float(reference_outputs.get("output") or "0")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", outputs.get("output") or "")
    score = int(bool(numbers) and abs(float(numbers[0]) - expected) < 0.01)
    return {"key": "math_correct", "score": score}


def convert_evaluator(inputs: dict, reference_outputs: dict, outputs: dict) -> dict:
    expected = float(reference_outputs.get("output") or "0")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", outputs.get("output") or "")
    score = int(bool(numbers) and abs(float(numbers[0]) - expected) <= expected * 0.01)
    return {"key": "conversion_correct", "score": score}


def weather_evaluator(inputs: dict, reference_outputs: dict, outputs: dict) -> dict:
    output = outputs.get("output") or ""
    has_temp = bool(re.search(r"-?\d+(?:\.\d+)?°C", output))
    has_city = inputs.get("city", "").lower() in output.lower()
    return {"key": "has_temperature", "score": int(has_temp and has_city)}


def search_evaluator(inputs: dict, reference_outputs: dict, outputs: dict) -> dict:
    expected = (reference_outputs.get("output") or "").lower()
    score = int(expected in (outputs.get("output") or "").lower())
    return {"key": "keyword_found", "score": score}


def shelter_evaluator(inputs: dict, reference_outputs: dict, outputs: dict) -> dict:
    has_number = bool(re.search(r"\b\d+\b", outputs.get("output") or ""))
    return {"key": "has_count", "score": int(has_number)}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    experiments = [
        ("Math",          "eval-math",       math_target,    math_evaluator),
        ("Unit Convert",  "eval-convert",    convert_target, convert_evaluator),
        ("Weather",       "eval-weather",    weather_target, weather_evaluator),
        ("Web Search",    "eval-web-search", search_target,  search_evaluator),
        ("Animal Shelter","eval-shelter",    shelter_target, shelter_evaluator),
    ]

    for name, dataset, target, evaluator in experiments:
        print(f"Running {name}...")
        evaluate(
            target,
            data=dataset,
            evaluators=[evaluator],
            experiment_prefix=f"eval-{name.lower().replace(' ', '-')}",
        )
