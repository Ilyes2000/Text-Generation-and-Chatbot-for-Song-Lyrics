# Text-Generation-and-Chatbot-for-Song-Lyrics
The objective of this project is to create a text generation model for generating song lyrics given descriptions.

The main part of the code is in the `Song_lyrics_generation.ipynb` notebook.

Some parts of the code require a Hugging Face token. It can be set as a `HF_TOKEN=YOUR_TOKEN` environment variable.

Explanation of the files:

- `plots` directory - includes the plots generated during the training process.

- `results` directory - includes all collected user evaluations

- `Song_lyrics_generation.ipynb` notebook - main part of the code (dataset exploration, generating descriptions, visualizing results, etc)

- `generate_responses.py` - generate responses on 50 prompts (defined in the `generate_responses.py` file) with base and fine-tuned models (the responses are saved in the json files (`*-song-lyrics-generation.json`))

- `NLP_report.pdf` - the final report in PDF format

- `app.py` - web-based interface to perform the evaluations (Gradio application) 

Install requirements:

```
pip install torch datasets pandas tqdm transformers trl matplotlib seaborn ipykernel
```

## Evaluating/ranking the models:

The interface is deployed on Hugging Face:

https://huggingface.co/spaces/petkopetkov/LLM-song-lyrics-generation-ranking

Run it locally:

```
python app.py
```

## Generating fine-tuned/base models responses on 50 prompts:

The responses for these 50 prompts (defined in the `generate_responses.py` file) are already generated and store in the json files (`*-song-lyrics-generation.json`).

Generate the responses locally:

```
python generate_responses.py
```

## Hugging Face dataset and trained models:

- dataset - https://huggingface.co/datasets/petkopetkov/spotify-million-song-dataset-descriptions

- Qwen2.5-0.5B - https://huggingface.co/petkopetkov/Qwen2.5-0.5B-song-lyrics-generation

- SmolLM2-135M - https://huggingface.co/petkopetkov/SmolLM2-135M-song-lyrics-generation

- SmolLM2-135M-Instruct - https://huggingface.co/petkopetkov/SmolLM2-135M-Instruct-song-lyrics-generation

- SmolLM2-360M - https://huggingface.co/petkopetkov/SmolLM2-360M-song-lyrics-generation

- SmolLM2-360M-Instruct - https://huggingface.co/petkopetkov/SmolLM2-360M-Instruct-song-lyrics-generation