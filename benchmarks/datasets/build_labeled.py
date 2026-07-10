"""Build labeled good/bad benchmark datasets from the seed good-only rows.

Reads the immutable `*_seed.jsonl` good-only sources and writes the labeled
`*.jsonl` datasets the benchmark actually runs on. Idempotent: it never mutates
its inputs, so re-running regenerates byte-identical output every time. Each seed
row (a known-correct output) becomes one `good` row; a documented transformation
produces a matching `bad` row with a `failure_kind` from the taxonomy in
docs/plans/benchmark_v2_accuracy.md.

The bad outputs are realistic failure modes — wrong entities, contradictions,
omissions, hallucinated additions, context-unfaithful claims — not gibberish, so
the benchmark tests whether a tool can tell a real mistake from a real answer.

Usage:
    python -m benchmarks.datasets.build_labeled
"""

from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).parent


# --- qa_factual: curated wrong answers per question -------------------------
# Each entry: the distractor answer + the failure kind it represents. Wrong,
# but plausible — a confident-sounding incorrect fact, the hard case for a judge.
QA_BAD = {
    "What is the capital of France?": ("The capital of France is Berlin.", "wrong_entity"),
    "Who wrote Romeo and Juliet?": (
        "Romeo and Juliet was written by Charles Dickens.",
        "wrong_entity",
    ),
    "What is the chemical symbol for water?": (
        "The chemical symbol for water is CO2.",
        "wrong_entity",
    ),
    "In what year did World War II end?": ("World War II ended in 1918.", "wrong_entity"),
    "What planet is closest to the Sun?": (
        "Venus is the planet closest to the Sun.",
        "wrong_entity",
    ),
    "How many sides does a hexagon have?": ("A hexagon has eight sides.", "wrong_entity"),
    "What is the speed of light in a vacuum?": (
        "The speed of light in a vacuum is about 150,000 kilometres per second.",
        "wrong_entity",
    ),
    "Who painted the Mona Lisa?": ("The Mona Lisa was painted by Pablo Picasso.", "wrong_entity"),
    "What is the largest ocean on Earth?": (
        "The Atlantic Ocean is the largest ocean on Earth.",
        "wrong_entity",
    ),
    "What gas do plants absorb during photosynthesis?": (
        "Plants absorb oxygen during photosynthesis.",
        "contradiction",
    ),
    "What is the boiling point of water at sea level in Celsius?": (
        "Water boils at 50 degrees Celsius at sea level.",
        "wrong_entity",
    ),
    "How many bones are in the adult human body?": (
        "The adult human body has 312 bones.",
        "wrong_entity",
    ),
    "What is the longest river in the world?": (
        "The Thames is the longest river in the world.",
        "wrong_entity",
    ),
    "What element has the atomic number 1?": ("Helium has the atomic number 1.", "wrong_entity"),
    "In which country is the Great Wall located?": (
        "The Great Wall is located in Japan.",
        "wrong_entity",
    ),
    "What is the smallest prime number?": ("The smallest prime number is 1.", "wrong_entity"),
    "Who invented the telephone?": ("The telephone was invented by Thomas Edison.", "wrong_entity"),
    "What is the currency of Japan?": ("The currency of Japan is the Won.", "wrong_entity"),
    "How many continents are there on Earth?": (
        "There are five continents on Earth.",
        "wrong_entity",
    ),
    "What is the square root of 144?": ("The square root of 144 is 14.", "wrong_entity"),
}

# Extra hand-authored QA pairs to grow the set past 100 total rows. Each yields
# one good + one bad, so this list plus the 20 seed rows gives 40 questions x 2.
QA_EXTRA = [
    # (question, correct_output, reference, bad_output, failure_kind)
    (
        "What is the tallest mountain on Earth?",
        "Mount Everest is the tallest mountain on Earth.",
        "Mount Everest",
        "K2 is the tallest mountain on Earth.",
        "wrong_entity",
    ),
    (
        "Who developed the theory of relativity?",
        "Albert Einstein developed the theory of relativity.",
        "Albert Einstein",
        "Isaac Newton developed the theory of relativity.",
        "wrong_entity",
    ),
    (
        "What is the hardest natural substance?",
        "Diamond is the hardest natural substance.",
        "Diamond",
        "Gold is the hardest natural substance.",
        "wrong_entity",
    ),
    (
        "How many players are on a soccer team on the field?",
        "A soccer team has eleven players on the field.",
        "Eleven",
        "A soccer team has nine players on the field.",
        "wrong_entity",
    ),
    (
        "What is the freezing point of water in Celsius?",
        "Water freezes at 0 degrees Celsius.",
        "0 degrees Celsius",
        "Water freezes at 32 degrees Celsius.",
        "wrong_entity",
    ),
    (
        "What is the largest planet in the solar system?",
        "Jupiter is the largest planet in the solar system.",
        "Jupiter",
        "Saturn is the largest planet in the solar system.",
        "wrong_entity",
    ),
    (
        "Who was the first person to walk on the Moon?",
        "Neil Armstrong was the first person to walk on the Moon.",
        "Neil Armstrong",
        "Buzz Aldrin was the first person to walk on the Moon.",
        "wrong_entity",
    ),
    (
        "What language has the most native speakers?",
        "Mandarin Chinese has the most native speakers.",
        "Mandarin Chinese",
        "English has the most native speakers.",
        "wrong_entity",
    ),
    (
        "What is the powerhouse of the cell?",
        "The mitochondria is the powerhouse of the cell.",
        "Mitochondria",
        "The nucleus is the powerhouse of the cell.",
        "wrong_entity",
    ),
    (
        "What is the capital of Japan?",
        "The capital of Japan is Tokyo.",
        "Tokyo",
        "The capital of Japan is Kyoto.",
        "wrong_entity",
    ),
    (
        "How many colours are in a rainbow?",
        "A rainbow has seven colours.",
        "Seven",
        "A rainbow has five colours.",
        "wrong_entity",
    ),
    (
        "What is the primary gas in Earth's atmosphere?",
        "Nitrogen is the primary gas in Earth's atmosphere.",
        "Nitrogen",
        "Oxygen is the primary gas in Earth's atmosphere.",
        "contradiction",
    ),
    (
        "Who painted the ceiling of the Sistine Chapel?",
        "Michelangelo painted the ceiling of the Sistine Chapel.",
        "Michelangelo",
        "Raphael painted the ceiling of the Sistine Chapel.",
        "wrong_entity",
    ),
    (
        "What is the smallest country in the world?",
        "Vatican City is the smallest country in the world.",
        "Vatican City",
        "Monaco is the smallest country in the world.",
        "wrong_entity",
    ),
    (
        "What is the study of living organisms called?",
        "The study of living organisms is called biology.",
        "Biology",
        "The study of living organisms is called geology.",
        "wrong_entity",
    ),
    (
        "How many strings does a standard guitar have?",
        "A standard guitar has six strings.",
        "Six",
        "A standard guitar has four strings.",
        "wrong_entity",
    ),
    (
        "What is the largest mammal on Earth?",
        "The blue whale is the largest mammal on Earth.",
        "Blue whale",
        "The elephant is the largest mammal on Earth.",
        "wrong_entity",
    ),
    (
        "What is the chemical symbol for gold?",
        "The chemical symbol for gold is Au.",
        "Au",
        "The chemical symbol for gold is Ag.",
        "wrong_entity",
    ),
    (
        "In which year did the first man land on the Moon?",
        "The first Moon landing was in 1969.",
        "1969",
        "The first Moon landing was in 1979.",
        "wrong_entity",
    ),
    (
        "What is the national language of Brazil?",
        "The national language of Brazil is Portuguese.",
        "Portuguese",
        "The national language of Brazil is Spanish.",
        "wrong_entity",
    ),
    (
        "What is the capital of Australia?",
        "The capital of Australia is Canberra.",
        "Canberra",
        "The capital of Australia is Sydney.",
        "wrong_entity",
    ),
    (
        "Who discovered penicillin?",
        "Alexander Fleming discovered penicillin.",
        "Alexander Fleming",
        "Louis Pasteur discovered penicillin.",
        "wrong_entity",
    ),
    (
        "What is the largest desert in the world?",
        "The Antarctic Desert is the largest desert in the world.",
        "Antarctic Desert",
        "The Sahara is the largest desert in the world.",
        "wrong_entity",
    ),
    (
        "How many teeth does a typical adult human have?",
        "A typical adult human has 32 teeth.",
        "32",
        "A typical adult human has 28 teeth.",
        "wrong_entity",
    ),
    (
        "What is the fastest land animal?",
        "The cheetah is the fastest land animal.",
        "Cheetah",
        "The lion is the fastest land animal.",
        "wrong_entity",
    ),
    (
        "What is the main ingredient in traditional bread?",
        "The main ingredient in traditional bread is flour.",
        "Flour",
        "The main ingredient in traditional bread is sugar.",
        "wrong_entity",
    ),
    (
        "Who wrote the play Hamlet?",
        "Hamlet was written by William Shakespeare.",
        "William Shakespeare",
        "Hamlet was written by Christopher Marlowe.",
        "wrong_entity",
    ),
    (
        "What is the chemical symbol for sodium?",
        "The chemical symbol for sodium is Na.",
        "Na",
        "The chemical symbol for sodium is So.",
        "wrong_entity",
    ),
    (
        "What is the largest internal organ in the human body?",
        "The liver is the largest internal organ in the human body.",
        "Liver",
        "The heart is the largest internal organ in the human body.",
        "wrong_entity",
    ),
    (
        "In which year did the Titanic sink?",
        "The Titanic sank in 1912.",
        "1912",
        "The Titanic sank in 1920.",
        "wrong_entity",
    ),
]


def _load(name: str) -> list[dict]:
    rows = []
    with open(HERE / name, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))
    return rows


def _good(row: dict, **extra) -> dict:
    out = dict(row)
    out["label"] = "good"
    out["failure_kind"] = None
    out.update(extra)
    return out


def _bad(row: dict, output: str, failure_kind: str, **extra) -> dict:
    out = dict(row)
    out["output"] = output
    out["label"] = "bad"
    out["failure_kind"] = failure_kind
    out.update(extra)
    return out


def build_qa_factual() -> list[dict]:
    labeled: list[dict] = []
    for row in _load("qa_factual_seed.jsonl"):
        labeled.append(_good(row))
        bad_output, kind = QA_BAD[row["input"]]
        labeled.append(_bad(row, bad_output, kind))
    for q, good_out, ref, bad_out, kind in QA_EXTRA:
        base = {"input": q, "reference_output": ref, "category": "factual_qa", "source": "authored"}
        labeled.append(_good({**base, "output": good_out}))
        labeled.append(_bad({**base, "output": good_out}, bad_out, kind))
    return labeled


def build_summarization() -> list[dict]:
    """Good = the reference summary. Bad = a summary that drops a key fact
    (omission) or injects a fact not in the source (hallucination), alternating."""
    labeled: list[dict] = []
    rows = _load("summarization_seed.jsonl")
    for i, row in enumerate(rows):
        labeled.append(_good(row))
        ref = row["reference_output"]
        if i % 2 == 0:
            # omission: keep only the first clause, dropping later required facts
            clipped = ref.split(",")[0].rstrip(".") + "."
            labeled.append(_bad(row, clipped, "omission"))
        else:
            # hallucination: append an invented, source-absent claim
            injected = ref.rstrip(".") + ", and the project received a Nobel Prize for its impact."
            labeled.append(_bad(row, injected, "hallucination"))
    return labeled


def build_rag_retrieval() -> list[dict]:
    """Good = context-faithful answer. Bad = a claim not supported by the
    retrieved context (unfaithful_to_context) or a direct contradiction."""
    labeled: list[dict] = []
    rows = _load("rag_retrieval_seed.jsonl")
    for i, row in enumerate(rows):
        labeled.append(_good(row))
        ref = row["reference_output"]
        if i % 2 == 0:
            bad = (
                ref.rstrip(".")
                + ", according to a 2050 government study not present in the passage."
            )
            labeled.append(_bad(row, bad, "unfaithful_to_context"))
        else:
            # contradiction: negate the first assertion of the reference
            bad = ref.replace("no ", "significant ").replace("improves", "worsens")
            if bad == ref:
                bad = "Contrary to the passage, " + ref[0].lower() + ref[1:]
            labeled.append(_bad(row, bad, "contradiction"))
    return labeled


def _write(name: str, rows: list[dict]) -> None:
    path = HERE / name
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    good = sum(1 for r in rows if r["label"] == "good")
    bad = len(rows) - good
    print(f"  {name}: {len(rows)} rows ({good} good / {bad} bad)")


def main() -> None:
    print("Building labeled datasets:")
    _write("qa_factual.jsonl", build_qa_factual())
    _write("summarization.jsonl", build_summarization())
    _write("rag_retrieval.jsonl", build_rag_retrieval())


if __name__ == "__main__":
    main()
