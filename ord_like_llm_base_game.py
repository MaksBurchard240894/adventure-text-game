#!/usr/bin/env python3
"""
LLM-powered Ord-like minimalist adventure.
Uses your fine-tuned model to generate 3-word story beats.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import random

MODEL_PATH = "ord_tuned_distilgpt2"  # your trained model folder

# --- Load model ---
device = 0 if torch.cuda.is_available() else -1
print(f"Loading model on device {device}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

# --- Generation helper ---
def generate_event(history, previous_outcome=None):
    """
    Generate a new event given story history and last outcome.
    """
    # Build a short narrative context
    context = " ".join(
        [f"{e}/{c}/{o}" for e, c, o in history[-3:]]
    )
    if previous_outcome:
        context += f" {previous_outcome}"
    prompt = f"{context}\nNext Event:"

    result = pipe(
        prompt,
        max_new_tokens=24,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.2,
        num_return_sequences=1,
    )[0]["generated_text"]

    text = result[len(prompt):].replace("\n", " ")
    # Simple fallbacks
    event_name = previous_outcome.capitalize() if previous_outcome else "Start"
    choices, outcomes = ["Continue", "Wait"], ["...", "..."]

    # Extract rough structure
    if "Event:" in text:
        try:
            event_name = text.split("Event:")[1].split("Choices:")[0].strip().split()[0]
        except Exception:
            pass
    if "Choices:" in text:
        part = text.split("Choices:")[1].split("Outcomes:")[0]
        parts = [x.strip().strip(".") for x in part.split("/") if x.strip()]
        if len(parts) >= 2:
            choices = parts[:2]
    if "Outcomes:" in text:
        part = text.split("Outcomes:")[1]
        outs = [x.strip().strip(".") for x in part.split("/") if x.strip()]
        if len(outs) >= 2:
            outcomes = outs[:2]

    return event_name.capitalize(), choices, outcomes


# --- Gameplay loop ---
def play_game():
    print("=" * 32)
    print("   ORD-LIKE LLM ADVENTURE   ")
    print("=" * 32)
    print("Type 1 or 2 to choose.")
    print("Type q to quit.\n")

    history = []
    event_name, choices, outcomes = generate_event()

    while True:
        print(f"\n{event_name}")
        print(f"1. {choices[0]}")
        print(f"2. {choices[1]}")

        user = input("> ").strip().lower()
        if user in ("q", "quit", "exit"):
            break
        if user not in ("1", "2"):
            print("Please type 1, 2, or q.")
            continue

        idx = int(user) - 1
        outcome = outcomes[idx] if idx < len(outcomes) else "..."
        print(outcome)

        history.append((event_name, choices[idx], outcome))

        # generate next event based on this outcome
        event_name, choices, outcomes = generate_event(previous_outcome=outcome)

        # occasional end trigger
        if random.random() < 0.1:
            print("\nEnd.")
            break

    print("\nYour journey:")
    for e, c, o in history:
        print(f"{e} / {c} / {o}")
    print("\nThanks for playing!")


if __name__ == "__main__":
    play_game()
