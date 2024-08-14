import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_tree_vectors_gpt2(branching_factors, steps, initial_prompt, model, tokenizer):
    full_paths = []  # List to store all completed paths

    # Tokenize the initial prompt and get embeddings for each token
    tokens = tokenizer.tokenize(initial_prompt)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    inputs = tokenizer(initial_prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        initial_embeddings = outputs.hidden_states[-1][0].cpu().numpy()

    # Create the initial path with all tokens and embeddings from the initial prompt
    initial_path = [(token_ids[i], initial_embeddings[i]) for i in range(len(tokens))]
    current_level = [initial_path]

    for idx, d in enumerate(branching_factors):
        next_level = []
        for path in current_level:
            parent_token_ids = [token_id for token_id, _ in path]  # Extract token IDs
            for _ in range(d):
                new_token_ids = parent_token_ids
                new_path = path.copy()  # Start a new path based on the current one

                # Generate a new token and update the path
                for _ in range(steps):
                    inputs = torch.tensor([new_token_ids])
                    outputs = model.generate(inputs, max_new_tokens=1, do_sample=True, top_k=50, temperature=0.7)
                    next_token_id = outputs[0, -1].item()
                    new_token_ids.append(next_token_id)

                    # Get the output embedding of the new token
                    inputs = torch.tensor([new_token_ids])
                    with torch.no_grad():
                        outputs = model(inputs, output_hidden_states=True)
                        new_vector = outputs.hidden_states[-1][0, -1].cpu().numpy()

                    new_path.append((next_token_id, new_vector))

                # If it's the last branching factor, consider it a complete path
                if idx == len(branching_factors) - 1:
                    # Decode the full text from token IDs
                    full_text = tokenizer.decode([token_id for (token_id, _) in new_path])
                    full_paths.append((full_text, new_path))
                else:
                    next_level.append(new_path)
        current_level = next_level

    return full_paths


if __name__ == "__main__":
    # Initialize GPT-2 model and tokenizer
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Example parameters
    branching_factors = [2, 4]  # Branching pattern
    steps = 3                   # Number of steps to move in each direction before next branching
    initial_prompt = "It is"  # Initial prompt token

    # Generate tree vectors and paths
    full_paths = generate_tree_vectors_gpt2(branching_factors, steps, initial_prompt, model, tokenizer)


    print(full_paths)
