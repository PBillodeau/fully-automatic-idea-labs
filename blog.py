from gpt4all import GPT4All
import subprocess

model = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf", device='gpu')

config = dict(
    max_tokens=800, # (int, default: 200 ) – The maximum number of tokens to generate.
    temp=0.7, # (float, default: 0.7 ) – The model temperature. Larger values increase creativity but decrease factuality.
    top_k=40, # (int, default: 40 ) – Randomly sample from the top_k most likely tokens at each generation step. Set this to 1 for greedy decoding.
    top_p=0.4, # (float, default: 0.4 ) – Randomly sample at each generation step from the top most likely tokens whose probabilities add up to top_p.
    min_p=0.0, # (float, default: 0.0 ) – Randomly sample at each generation step from the top most likely tokens whose probabilities are at least min_p.
    repeat_penalty=1.18, # (float, default: 1.18 ) – Penalize the model for repetition. Higher values result in less repetition.
    repeat_last_n=64, # (int, default: 64 ) – How far in the models generation history to apply the
    n_batch=16, # (int, default: 8 ) – Number of prompt tokens processed in parallel. Larger values decrease latency but increase resource requirements.
    n_predict=None, # (int | None, default: None ) – Equivalent to max_tokens, exists for backwards compatibility.
    streaming=True # (bool, default: False ) – If True, this method will instead return a generator that yields tokens as the model generates them.
)

with model.chat_session():
    prompt = input("Title: ")
    response = model.generate(prompt="You are a brilliant tech leader with over 40 years of experience. Create a blog post for Fully Automatic Idea Labs with the title: " + prompt, **config)

    with open('./docs/' + '-'.join(prompt.split()) + '.html', 'w') as outfile:
        outfile.write(''.join(response))

    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", "Create " + prompt])
    subprocess.run(["git", "push", "origin", "master"])
