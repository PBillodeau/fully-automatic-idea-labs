from gpt4all import GPT4All
import subprocess

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

def create_content(prompt):
    model = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf", device='gpu')

    with model.chat_session():
        response = model.generate(prompt="You are a brilliant tech leader with over 40 years of experience. Create a blog post with the title: " + prompt, **config)

    print("Writing article...")
    with open('./articles/' + '-'.join(prompt.split()).lower() + '.html', 'w') as outfile:
        outfile.write('<pre class="container" style="white-space: break-spaces;">' + ''.join(response).strip() + '</pre>')

def build_page(prompt):
    print("Building page...")
    with open('./docs/' + '-'.join(prompt.split()).lower() + '.html', 'w') as outfile:
        with open('./templates/header.html', 'r') as header:
            for line in header:
                outfile.write(line)

        with open('./articles/' + '-'.join(prompt.split()).lower() + '.html', 'r') as content:
            for line in content:
                outfile.write(line)

        with open('./templates/footer.html', 'r') as footer:
            for line in footer:
                outfile.write(line)

def deploy(message):
    print("Deploying...")
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", message])
    subprocess.run(["git", "push", "origin", "master"])

def main():
    prompt = input("Title: ")
    create_content(prompt)
    build_page(prompt)
    deploy("Create " + prompt)

if __name__ == "__main__":
    main()
