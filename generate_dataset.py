import json
import random

questions = [
    ("What is AI?", "AI stands for Artificial Intelligence."),
    ("What is Python?", "Python is a popular programming language."),
    ("Who is the Prime Minister of India?", "The Prime Minister of India is Narendra Modi."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is 2 + 2?", "2 + 2 equals 4."),
    ("What is the boiling point of water?", "The boiling point of water is 100 degrees Celsius."),
    ("Who wrote Harry Potter?", "J.K. Rowling wrote Harry Potter."),
    ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
    ("What is the speed of light?", "The speed of light is approximately 299,792 kilometers per second."),
    ("What is the currency of Japan?", "The currency of Japan is Yen."),
    ("What is the tallest mountain?", "Mount Everest is the tallest mountain in the world."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
    ("What is the chemical symbol for water?", "The chemical symbol for water is H2O."),
    ("What is the smallest prime number?", "The smallest prime number is 2."),
    ("Who discovered gravity?", "Sir Isaac Newton discovered gravity."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is the freezing point of water?", "The freezing point of water is 0 degrees Celsius."),
    ("Who is known as the father of computers?", "Charles Babbage is known as the father of computers."),
    ("What is the largest ocean?", "The Pacific Ocean is the largest ocean on Earth."),
    ("What is the national animal of India?", "The national animal of India is the Bengal Tiger."),
]

with open("dataset.jsonl", "w", encoding="utf-8") as f:
    for i in range(1000):
        q, a = random.choice(questions)
        entry = {"user": q, "assistant": a}
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print("Dataset generated: 1000 lines written to dataset.jsonl")