"""
Three-Word Adventure
A minimalist text-based adventure driven by an LLM.
Each turn:
- 1 word situation
- 2 one-word options
- Player picks one
- 1 word outcome appears
- After 3 seconds, screen clears and next situation + options appear
"""

from groq import Groq
import os
import time
import platform

# --- CONFIGURATION ---

# Insert your Groq API key here
api_key = input("Please enter your Groq API key: ").strip()
client = Groq(api_key=api_key)

MODEL_PRIMARY = "llama-3.3-70b-versatile"
MODEL_FALLBACK = "llama-3.1-8b-instant"

TEMPERATURE = 0.9
MAX_HISTORY_LINES = 15
DISPLAY_DELAY = 3  # seconds


# --- HELPER FUNCTIONS ---

def clear_screen():
    """Clears the terminal screen cross-platform."""
    os.system("cls" if platform.system() == "Windows" else "clear")


def parse_story_step(step_text: str):
    """
    Split the LLM response into separate parts (Outcome, Situation, Options)
    """
    lines = step_text.splitlines()
    outcome_line = next((l for l in lines if l.lower().startswith("outcome:")), None)
    situation_line = next((l for l in lines if l.lower().startswith("situation:")), None)
    options_line = next((l for l in lines if l.lower().startswith("options:")), None)

    return outcome_line, situation_line, options_line


def generate_story_step(history: str) -> str:
    """
    Generate the next step using Groq's LLM.
    """
    prompt = f"""
Continue this 3-word text adventure coherently. The story will start quite mondane but it is your task to lead it into a fantasy direction quickly with dragons, goblins, forests, towers and such. 

Rules:
- Use only this format:
Outcome: <one word>
Situation: <one word>
Options: <option1> | <option2>
- Keep it connected to previous events.
- No explanations or extra text.

Story so far:
{history.strip()}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_PRIMARY,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
    except Exception:
        # fallback model if primary fails
        response = client.chat.completions.create(
            model=MODEL_FALLBACK,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )

    return response.choices[0].message.content.strip()


# --- MAIN GAME LOOP ---

def main():
    clear_screen()
    print("🎮 Welcome to the Three-Word Adventure!\n")
    situation = "Situation: Alarm"
    options = "Options: Snooze | Wake"
    history = f"{situation}\n{options}"

    print(f"{situation}\n{options}")

    while True:
        choice = input("\nYour choice: ").strip().capitalize()

        if choice.lower() in ["quit", "exit"]:
            print("\n👋 Thanks for playing!")
            break

        history += f"\nChoice: {choice}\n"

        try:
            step = generate_story_step(history)
        except Exception as e:
            print(f"\n⚠️ Error generating story: {e}")
            break

        outcome, new_situation, new_options = parse_story_step(step)

        # Show only the outcome first
        if outcome:
            print(f"\n{outcome}")
        else:
            print("\n(Outcome missing from model response)")

        # Wait before clearing and showing next part
        time.sleep(DISPLAY_DELAY)
        clear_screen()

        # Then show the next situation and options
        if new_situation and new_options:
            print(f"{new_situation}\n{new_options}")
        else:
            print("(Incomplete response from model)")
            print(step)

        # Add to history
        history += "\n" + step

        # Trim history
        if len(history.splitlines()) > MAX_HISTORY_LINES:
            history = "\n".join(history.splitlines()[-MAX_HISTORY_LINES:])


# --- ENTRY POINT ---

if __name__ == "__main__":
    main()
