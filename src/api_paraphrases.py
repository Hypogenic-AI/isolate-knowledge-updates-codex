import json
import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt


def _get_client_and_model():
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if openai_key:
        return OpenAI(api_key=openai_key), "gpt-4.1"
    if openrouter_key:
        # OpenRouter supports OpenAI-compatible client with base_url override.
        return OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1"), "openai/gpt-4.1"

    raise RuntimeError("No OPENAI_API_KEY or OPENROUTER_API_KEY found in environment")


@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(5))
def _call_api(client: OpenAI, model: str, prompt: str, n: int) -> list[str]:
    system = (
        "You generate paraphrases of a short prompt. "
        "Return ONLY valid JSON: a list of strings."
    )
    user = (
        f"Paraphrase the prompt exactly {n} times. Keep mathematical meaning identical. "
        f"Prompt: {prompt}\n"
        "Rules: keep it short, do not include explanations, return JSON list only."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.7,
        max_tokens=400,
    )
    content = resp.choices[0].message.content.strip()
    return json.loads(content)


def generate_paraphrases(prompt: str, n: int = 20) -> dict:
    client, model = _get_client_and_model()
    paraphrases = _call_api(client, model, prompt, n)
    return {
        "prompt": prompt,
        "paraphrases": paraphrases,
        "model": model,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def main() -> None:
    out_path = Path("results/paraphrases.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_paraphrases("2+2=", n=20)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
