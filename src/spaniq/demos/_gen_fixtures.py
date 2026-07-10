"""Internal helper to write RAG breakage JSONL fixtures. Run once."""

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "rag_breakage"
PROMPT = "What is the refund policy for orders placed in the last 30 days?"

with open(FIXTURES / "baselines.json") as f:
    baselines = json.load(f)

with_ctx = [{"input": PROMPT, "output": o} for o in baselines]

no_ctx = [
    {"input": PROMPT, "output": o}
    for o in [
        "I'm not entirely sure about the specific refund details. Generally most stores offer some return policy. I'd recommend checking the website or contacting support.",
        "Typically refund policies vary by retailer. I don't have the specific details, but you might want to check the terms and conditions or reach out to customer service.",
        "I'm not certain of the exact refund terms. Most retailers generally offer a return window. I'd suggest checking the official policy page or contacting support.",
        "The refund policy details aren't something I can confirm with certainty. I'd recommend checking your order confirmation email or the store's website.",
        "I don't have the specific refund policy details available. Generally speaking, many retailers offer returns within a reasonable timeframe. Please check the website.",
        "Without specific policy information, refund terms vary. You should check the store's terms and conditions or contact customer support for accurate details.",
        "I'm unable to provide the exact refund policy details right now. I would recommend visiting the website's help section or contacting our support team.",
        "The specific refund policy details aren't available to me at the moment. Typically stores have return windows, but for exact terms please check the website.",
        "I don't have access to the specific refund policy right now. Generally retailers offer some form of return policy. Please check the website or contact support.",
        "I'm not sure about the exact refund terms without access to the policy. I'd suggest checking the FAQ section or reaching out to our support team.",
        "Without the policy details, I can't give you exact information. Most stores typically offer some return window. Please check the website for accurate information.",
        "I don't have the specific details of our refund policy available. I would recommend checking the website or contacting customer support.",
        "The exact refund terms aren't something I can confirm without the policy document. Generally speaking, return policies vary. Please check the official policy page.",
        "I'm uncertain about the specific refund timeline without access to the policy. Typically returns are handled case by case. I'd suggest contacting support.",
        "Without access to the specific policy, I can only say generally that refund terms vary. Please check the website or contact support for exact details.",
        "I don't have the refund policy details right now. Generally most retailers offer some return option. For specific terms, please check the website.",
        "The specific refund policy isn't available to me at this time. I'd recommend checking your order details or contacting support for accurate information.",
        "I'm not certain about the exact refund terms. Typically these things vary by purchase. Please check the store's official policy page.",
        "Without the policy in front of me, I can't give you exact details. Generally stores have some form of return policy. Please contact support for specifics.",
        "I don't have access to the specific refund terms right now. I would suggest checking the website FAQ or contacting customer support.",
    ]
]

with open(FIXTURES / "traces_with_context.jsonl", "w") as f:
    for r in with_ctx:
        f.write(json.dumps(r) + "\n")

with open(FIXTURES / "traces_no_context.jsonl", "w") as f:
    for r in no_ctx:
        f.write(json.dumps(r) + "\n")

print("rag_breakage fixtures written")
