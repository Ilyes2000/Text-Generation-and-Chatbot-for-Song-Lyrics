from transformers import pipeline
from tqdm import tqdm
import json

models = [
    {
        "finetuned_model_name": "petkopetkov/SmolLM2-135M-song-lyrics-generation",
        "base_model_name": "HuggingFaceTB/SmolLM2-135M",
    },
    {
        "finetuned_model_name": "petkopetkov/SmolLM2-135M-Instruct-song-lyrics-generation",
        "base_model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
    },
    {
        "finetuned_model_name": "petkopetkov/SmolLM2-360M-song-lyrics-generation",
        "base_model_name": "HuggingFaceTB/SmolLM2-360M",
    },
    {
        "finetuned_model_name": "petkopetkov/SmolLM2-360M-Instruct-song-lyrics-generation",
        "base_model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
    },
    {
        "finetuned_model_name": "petkopetkov/Qwen2.5-0.5B-song-lyrics-generation",
        "base_model_name": "Qwen/Qwen2.5-0.5B",
    },
]

prompts = [
    "melancholic, dreamy, ethereal, nostalgic, poetic, airy, bittersweet, intimate, storytelling, mystical, Fleetwood Mac",
    "raw, gritty, defiant, bluesy, rebellious, urgent, intense, soulful, electrifying, passionate, visceral, The Rolling Stones",
    "sultry, dark, moody, hypnotic, brooding, mysterious, sensual, evocative, cinematic, smooth, jazzy, Lana Del Rey",
    "playful, quirky, satirical, upbeat, theatrical, ironic, witty, punchy, lively, humorous, cabaret, Queen",
    "haunting, poetic, gothic, mystical, ethereal, sorrowful, enchanting, dreamlike, introspective, dark romance, Florence + The Machine",
    "sentimental, heartfelt, acoustic, nostalgic, warm, raw, sincere, storytelling, comforting, emotional, Paul Simon",
    "aggressive, fast-paced, high-energy, punk, anti-establishment, rebellious, punchy, straightforward, sarcastic, gritty, The Ramones",
    "smooth, sensual, groovy, jazzy, sophisticated, elegant, soulful, warm, intimate, laid-back, Norah Jones",
    "dark, industrial, mechanical, menacing, raw, aggressive, intense, futuristic, dystopian, pulsating, Nine Inch Nails",
    "joyful, funky, celebratory, groovy, high-energy, playful, uplifting, smooth, stylish, charismatic, Earth, Wind & Fire",
    "nostalgic, dreamy, shoegaze, surreal, hazy, meditative, otherworldly, romantic, melancholic, introspective, My Bloody Valentine",
    "poignant, sorrowful, heartbreaking, emotional, cinematic, deeply personal, confessional, intimate, lyrical, Damien Rice",
    "euphoric, electronic, uplifting, anthemic, pulsating, danceable, atmospheric, modern, exhilarating, dreamy, Calvin Harris",
    "explosive, rebellious, aggressive, high-energy, politically charged, confrontational, raw, powerful, uncompromising, Rage Against The Machine",
    "mellow, chilled, jazzy, warm, laid-back, intimate, groovy, smooth, nighttime vibes, escapist, Tom Misch",
    "ethereal, uplifting, majestic, orchestral, grandiose, cinematic, dreamlike, floating, celestial, Coldplay",
    "fierce, confident, powerful, sassy, anthemic, empowering, bold, danceable, charismatic, Rihanna",
    "mystical, progressive, cinematic, epic, dramatic, storytelling, ethereal, poetic, surreal, Tool",
    "raw, storytelling, gritty, emotional, melancholic, observational, confessional, lonely, folk-inspired, Bruce Springsteen",
    "hypnotic, dark, electronic, industrial, brooding, mysterious, pulsating, futuristic, immersive, The Weeknd",
    "nostalgic, hopeful, country, storytelling, heartfelt, acoustic, warm, sentimental, down-to-earth, sincere, Taylor Swift",
    "edgy, intense, high-energy, heavy, chaotic, distorted, anthemic, aggressive, dynamic, Linkin Park",
    "sunny, carefree, acoustic, simple, breezy, beachy, laid-back, nostalgic, storytelling, warm, Jack Johnson",
    "spiritual, anthemic, uplifting, gospel-influenced, warm, celebratory, grand, soulful, reverent, Kanye West",
    "dark, cinematic, suspenseful, mysterious, poetic, gothic, haunting, theatrical, surreal, Nick Cave & The Bad Seeds",
    "bittersweet, nostalgic, romantic, emotional, warm, orchestral, poetic, timeless, storytelling, Elton John",
    "eccentric, bizarre, experimental, unpredictable, theatrical, chaotic, colorful, surreal, unpredictable, David Bowie",
    "tense, edgy, alternative, cryptic, moody, brooding, experimental, unsettling, layered, Radiohead",
    "uplifting, funky, vibrant, celebratory, high-energy, soulful, stylish, groovy, dynamic, Bruno Mars",
    "gritty, southern, bluesy, raw, rebellious, foot-stomping, storytelling, authentic, earthy, The Black Keys",
    "psychedelic, colorful, whimsical, surreal, dreamy, storytelling, imaginative, intricate, The Beatles (Sgt. Pepper era)",
    "ominous, aggressive, eerie, doom-laden, heavy, raw, dark, apocalyptic, relentless, Black Sabbath",
    "mellow, jazzy, poetic, romantic, introspective, wistful, lounge-inspired, warm, cinematic, Chet Baker",
    "soulful, passionate, heart-wrenching, confessional, gospel-tinged, deeply personal, powerful, Adele",
    "brooding, darkwave, gothic, atmospheric, melancholic, mysterious, ethereal, synth-heavy, Depeche Mode",
    "exuberant, bombastic, theatrical, outrageous, playful, rebellious, larger-than-life, Meat Loaf",
    "emotive, grungy, raw, introspective, angsty, visceral, poetic, haunting, Kurt Cobain/Nirvana",
    "sun-drenched, warm, nostalgic, storytelling, folky, reflective, easy-going, wistful, John Mayer",
    "menacing, relentless, rapid-fire, lyrical, aggressive, intense, braggadocious, gritty, Eminem",
    "psychedelic, experimental, otherworldly, transcendental, layered, mystical, avant-garde, Pink Floyd",
    "soulful, bluesy, smoky, passionate, dramatic, heart-on-sleeve, old-school, Amy Winehouse",
    "dark, cinematic, haunting, orchestral, operatic, storytelling, gothic, grand, Evanescence",
    "romantic, dreamy, cinematic, nostalgic, orchestral, delicate, floating, timeless, Michael Bublé",
    "heavy, dynamic, aggressive, emotional, epic, intense, dark, theatrical, Muse",
    "laid-back, retro, funky, cool, stylish, groovy, effortless, danceable, Anderson .Paak",
    "sentimental, acoustic, poetic, heartfelt, nostalgic, understated, warm, confessional, Tracy Chapman",
    "enigmatic, electronic, hypnotic, spacey, futuristic, cinematic, melancholic, The xx",
    "cheerful, retro, surf rock, fun, simple, carefree, lighthearted, beachy, youthful, The Beach Boys",
    "heartfelt, singer-songwriter, indie, introspective, soft-spoken, poetic, reflective, sad but hopeful, Phoebe Bridgers",
    "boisterous, loud, anthemic, rebellious, arena rock, high-energy, dramatic, stadium-filling, Bon Jovi"
]

for models_pair in models:
    finetuned_model_name = models_pair["finetuned_model_name"]
    base_model_name = models_pair["base_model_name"]
    finetuned_pipe = pipeline("text-generation", model=finetuned_model_name, tokenizer=finetuned_model_name)
    base_pipe = pipeline("text-generation", model=base_model_name, tokenizer=base_model_name)

    results = []
    for p in tqdm(prompts):
        finetuned_resp = finetuned_pipe(p, max_new_tokens=256, do_sample=True, truncation=True, batch_size=8)[0]["generated_text"]
        base_resp = base_pipe(p, max_new_tokens=256, do_sample=True, truncation=True, batch_size=8)[0]["generated_text"]

        results.append({
            "prompt": "Generate song lyrics based on the description: " + p + "\nSong lyrics:",
            "finetuned_output": finetuned_resp,
            "base_output": base_resp
        })
        with open(f"{finetuned_model_name.split('/')[1]}.json", "w") as outfile: 
            json.dump(results, outfile)
