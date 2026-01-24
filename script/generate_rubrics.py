import json
import litellm
import os
import pandas as pd
from tqdm import tqdm
import requests
import argparse
import datasets
import http.client
import litellm
from litellm.caching import Cache
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# TODO: set up your litellm api key
litellm.cache = Cache(type="disk", disk_cache_dir="litellm_cache/")

# --------------- Prompts ---------------
retrieval_prompt_w_snippets = """I will write a query that tests literature knowledge and give you a list of  snippets. Your task is to come back with a list of elements you would expect to see in the answer, and associate relevant snippets.
 
Each element should include an "ingredient", which is a detailed descriptor of what is expected in an answer, a "handle", which is a short descriptor that slightly abstracts away from the ingredient description (don't make the handles specific), and "specifics", which lists relevant texts and citations from the snippets given below. Finally group the elements into 3 categories: "Answer Critical": necessary elements without this you'd not have an answer, "Valuable": key supporting information are useful but not as necessary as "Answer Critical", and "Context": elaborations or background that help understanding. 

Rules:
* When you pull from the snippets, make sure that the text and citations exactly match the provided snippets.
* Include as many unique and relevant citations as you can. 
* For Context ingredients, only provide ingredients that have at least 2 citations.
* Ingredient descriptor should always start with a verb of command. It is a requirement so we want command verbs like "Mention", "Specify", "Describe", "Include", "Discuss" etc that specifies what a good answer should contain.
* It is acceptable to include ingredients even if no snippets support them or if the available snippets are insufficient. However, only do so when you have high confidence that the ingredient is essential for a well-grounded answer.

Return your answer in a json format, structured like this: 

{ "Question":<query>, 
"Answer Critical": [ { "Ingredient": <ingredient>, "Handle":<handle>, 
"Specifics":[{"Text":<snippet text>, "Citation":<corpus_id>}...]}...], 
"Valuable": [same structure as answer critical], "Context": [same structure] 
}

###
"""

retrieval_prompt_no_snippets = """I will provide a query that requires up-to-date, real-world knowledge. Produce a comprehensive long-form report that synthesizes all necessary information. Your task is to come back with a list of elements you would expect to see in the answer. Each element should include an "ingredient", which is a detailed descriptor of what is expected in an answer (start with a verb whenever it makes sense) and a "handle", which is a short descriptor that slightly abstracts away from the ingredient description. Finally group the elements into 3 categories: "Answer Critical": necessary elements without this you'd not have an answer, "Valuable": key supporting information are useful but not as necessary as "Answer Critical", and "Context": elaborations or background that help understanding.

Respond with a json.
Example:
QUERY: What are advantages and disadvantages of top methods for picking the right number of topics in topic modeling?
{
"Question": "What are advantages and disadvantages of top methods for picking the right number of topics in topic modeling?",
"Answer Critical": [
  {
    "Ingredient": "Mention the most important methods for topic modeling, such as elbow method, coherence metrics, and likelihood, together with their advantages and disadvantages.",
    "Handle": "Advantages and Disadvantages of Methods",
  }
],
"Valuable": [
  {
    "Ingredient": "Highlight the importance of picking the right number of topics in the performance of topic models.",
    "Handle": "Importance of Selection Methods",
  },
  {
    "Ingredient": "Mention that the choice of the method depends on different factors such as computational capability, the intended application, and maybe a mix of various methods would be required.",
    "Handle": "Constraints and Dependencies",
  }
],
"Context": [
  {
    "Ingredient": "Define topic modeling and its applications.",
    "Handle": "Background",
  },
  {
    "Ingredient": "Name LDA as one of the most important methods for topic modeling.",
    "Handle": "Most Important Method",
  }
]
}


Example 2:
QUERY: Describe what is known about overfitting in in-context learning.
{
"Question": "Describe what is known about overfitting in in-context learning.",
"Answer Critical":
[
{
"Ingredient": "Define overfitting specifically in the context of in-context learning",
"Handle": "Definition"
},
{
"Ingredient": "Discuss the causes of overfitting in in-context learning.",
"Handle": "Causes"
}
],
"Valuable":
[
{
"Ingredient": "Describe known methods to prevent or reduce ICL overfitting",
"Handle": "Mitigation Strategies"
}
],
"Context":
[
{
"Ingredient": "Provide an explanation of overfitting in the fine-tuning or training frameworks.",
"Handle": "Background"
},
{
"Ingredient": "Provide background for in-context learning",
"Handle": "Background"
}
}
####
"""


# --------------- Util for data loading and saving ---------------
def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# TODO: Set S2_API_KEY in your environment variables
# --------------- S2 API ---------------
def retrieve_s2_snippet(query, limit=10):
    params = {
        "query": query,
        "limit": limit,
        "minCitationCount": 10,
        "fields": "snippet.text",
    }
    key = os.getenv("S2_API_KEY")
    headers = {"x-api-key": key} if key else {}

    # Call S2 snippet search
    resp = requests.get(
        "https://api.semanticscholar.org/graph/v1/snippet/search",
        params=params,
        headers=headers,
        timeout=30,
    )
    data = resp.json()

    if isinstance(data, str):
        data = json.loads(data)

    processed = []

    for idx, item in enumerate(data.get("data", [])):
        paper = item.get("paper", {}) or {}
        text = ((item.get("snippet") or {}).get("text") or "").strip()
        title = paper.get("title") or ""
        paper_id = paper.get("corpusId") or paper.get("paperId") or paper.get("id")

        rec = {"paperID": paper_id, "title": title, "text": text}
        processed.append(rec)
    print(
        "{0} snippets are retrieved. Top1: {1}".format(
            len(processed), processed[0]["title"]
        )
    )
    return processed


# --------------- Web search API ---------------
# TODO: Set SERPER_API_KEY in your environment variables

def serper_search(query):
    conn = http.client.HTTPSConnection("google.serper.dev")
    key = os.getenv("SERPER_API_KEY")
    payload = json.dumps({"q": query})
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return json.loads(data.decode("utf-8"))["organic"]


def serper_scrape(url):
    conn = http.client.HTTPSConnection("scrape.serper.dev")
    key = os.getenv("SERPER_API_KEY")
    payload = json.dumps({"url": url})
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    conn.request("POST", "/", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return json.loads(data.decode("utf-8"))["text"]


def accept_up_to_max_words(
    scraped_text: str, max_words: int = 1000, add_ellipsis: bool = True
) -> str:
    """
    Return scraped_text truncated to at most `max_words` whitespace-delimited words.
    Uses a streaming regex to avoid creating large intermediate lists.
    """
    if not scraped_text:
        return scraped_text

    count = 0
    end_idx = None
    for m in re.finditer(r"\S+", scraped_text):
        count += 1
        end_idx = m.end()
        if count >= max_words:
            break

    # If fewer than max_words, return as-is
    if count < max_words:
        return scraped_text

    truncated = scraped_text[:end_idx].rstrip()
    return (truncated + "...") if add_ellipsis else truncated


def retrieve_web(query, run_scrape=True):
    serper_results = serper_search(query)
    snippets = []
    if run_scrape is True:
        for idx, s in enumerate(serper_results):
            if idx < 3:
                scraped_text = accept_up_to_max_words(serper_scrape(s["link"]))

                snippets.append(
                    {
                        "title": s["title"],
                        "text": "Summary: {0}\nFull text:\n{1}".format(
                            s["snippet"], scraped_text
                        ),
                    }
                )
            else:
                snippets.append({"title": s["title"], "text": s["snippet"]})

    else:
        for s in serper_results:
            snippets.append({"title": s["title"], "text": s["snippet"]})
    print(
        "{0} snippets are retrieved. Top1: {1}".format(
            len(snippets), snippets[0]["title"]
        )
    )
    return snippets


def extract_json_from_response(response):
    json_start = response.find("{")
    json_end = response.rfind("}") + 1
    if json_start == -1 or json_end == -1:
        return None

    try:
        return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        try:
            return json.loads(response[json_start:json_end] + "]}")
        except json.JSONDecodeError:
            print(
                f"Could not decode JSON from response: {response[json_start:json_end]}"
            )
        return None


def call_llm(query, output_file, error_file, no_retrieval=False, search_tool="s2"):
    if no_retrieval is True:
        prompt = retrieval_prompt_no_snippets + f"Query: {query}"
    else:
        if search_tool == "s2":
            snippets = retrieve_s2_snippet(query)
            time.sleep(1)
        else:
            snippets = retrieve_web(query)

        snippets_text = " ".join(
            [item["title"] + "\n" + item["text"] for item in snippets]
        )
        print(snippets_text[:1000])
        prompt = (
            retrieval_prompt_w_snippets + f"Query: {query}\nSnippets:\n{snippets_text}"
        )

    msgs = [{"role": "user", "content": prompt}]
    resp = litellm.completion(
        model="gpt-4.1",
        messages=msgs,
    )

    output = resp.choices[0].message.content
    cost = litellm.completion_cost(completion_response=resp)

    output = extract_json_from_response(output)
    if output is None:
        with open(error_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "query": query,
                    }
                )
                + "\n"
            )
    else:
        with open(output_file, "a") as f:
            f.write(json.dumps(output) + "\n")

    return {"query": query, "cost": cost}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_path", default=None, help="Path to input HF")
    ap.add_argument("--input", help="Path to write input JSONL")
    ap.add_argument("--output", required=True, help="Path to write output JSONL")
    ap.add_argument("--error", required=True, help="Path to write error JSONL")
    ap.add_argument(
        "--no_retrieval",
        action="store_true",
        help="set true to generate retrieval-free rubrics.",
    )
    ap.add_argument("--source", help="source task name")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.hf_path is not None:
        input_data = datasets.load_dataset(args.hf_path)["train"]
    else:
        input_data = read_jsonl(args.input)

    results = []
    # For OS queries
    # queries = [item["instance"]["query"] for item in input_data if "instance" in item]
    if args.source == "searcharena":
        queries = [
            item["query"]
            for item in input_data
            if "query" in item and item["source"] == "searcharena"
        ]  # search arena
        search_tool = "web"
    elif args.source == "openscholar":
        queries = [
            item["instance"]["query"] for item in input_data if "instance" in item
        ]
        search_tool = "s2"
    else:
        queries = [item["question"] for item in input_data if "question" in item]
        search_tool = "web"

    # this filter out search arena and healthbench
    results = []
    for q in tqdm(queries, total=len(queries)):
        try:
            res = call_llm(
                q, args.output, args.error, args.no_retrieval, search_tool=search_tool
            )
            results.append(res)
        except Exception as e:
            # don't crash the whole run; log and continue
            with open(args.error, "a", encoding="utf-8") as f:
                f.write(json.dumps({"query": q, "error": str(e)}) + "\n")
    costs = 0
    for result in results:
        if "cost" in result:
            costs += result["cost"]
    print(f"Total cost: {costs}")


if __name__ == "__main__":
    main()
