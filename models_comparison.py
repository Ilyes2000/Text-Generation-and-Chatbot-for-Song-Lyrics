
from transformers import pipeline
from transformers.utils import logging
import torch
from random import shuffle
import csv

#List of models
trained_models_list = [
    'petkopetkov/Qwen2.5-0.5B-song-lyrics-generation',
    'petkopetkov/SmolLM2-135M-song-lyrics-generation',
    'petkopetkov/SmolLM2-135M-Instruct-song-lyrics-generation',
    'petkopetkov/SmolLM2-360M-song-lyrics-generation',
    'petkopetkov/SmolLM2-360M-Instruct-song-lyrics-generation'
]

#Stuff for saving score into csv
filename = 'models_score/score.csv'

def read_score_file(filename):
    data = {}
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 2:
                data[row[0]] = int(row[1])
    return data

def update_score_file(filename, data):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for key, value in data.items():
            writer.writerow([key, value])

#Initialize our score dict and print it
score = read_score_file(filename)
print("Current scores : \n")
for key, value in score.items():
    print(f"{key} : {value}\n")

#making the output cleaner by making the model less verbose
logging.set_verbosity_error()

def compare(model_list, prompt):
    for model_name, i in zip(model_list, range(len(trained_models_list))):
        generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            max_new_tokens=1024,
            repetition_penalty=1.5,
            # temperature=0.6,
            # do_sample=True
        )

        output = generator(
            prompt,
            # do_sample=True,
            # top_k=50,
            # top_p=0.95
        )
        
        # Print the model name and the generated text
        print(f"\nModel number {i}\n")
        print(f"{output[0]['generated_text']}\n")
        print("*" * 50)

#How many times do the user want to compare models
n_try = int(input("How many times do you want to redo it ? (Enter a number >0)\n"))

#main loop
for i in range(n_try):
    print("*" * 100)
    #prompt = "Generate song lyrics based on the description: " + input("Enter key words for the song you want to generate : ") + "\nSong lyrics:\n"
    prompt = "Generate song lyrics based on the description: happy, joyful, and carefree, with a positive and uplifting vibe\nSong lyrics:\n"

    #shuffling the order of the models for less biased pick
    shuffle(trained_models_list)
    compare(trained_models_list, prompt)
    print("*" * 100)
    favorite = int(input("So, which one did you prefer ? (Enter the number of the model)\n"))
    score[trained_models_list[favorite]] += 1

#updating the score file before ending the program
update_score_file(filename, score)


