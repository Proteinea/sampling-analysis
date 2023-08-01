import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained('transformers/examples/pytorch/language-modeling/output/checkpoint-23000')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a padding token
model = AutoModelForCausalLM.from_pretrained('transformers/examples/pytorch/language-modeling/output/checkpoint-23000').to(device)
model.config.max_length = 512  # Set the maximum length to 512

input_file = "transformers/examples/pytorch/language-modeling/validation_seqs_50_1.txt"  # Replace with the path to your input file
output_file = "protgpt2_finetune_gfp_generated/exp0/exp02.txt"  # Replace with the path to your output file

with open(input_file, "r") as file:
    input_text = file.read()

input_sequences = input_text.split("\n")  # Split input file into sequences

generated_sequences = []
with open(output_file, "w") as file:
    for sequence in input_sequences:
        # Skip empty sequences
        if not sequence:
            continue

        # Tokenize the input sequence
        inputs = tokenizer.encode_plus(
            sequence,
            padding="longest",
            truncation=True,
            max_length=512,  # Set the maximum length to 512
            return_tensors="pt"
        ).to(device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Generate multiple sequences
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            top_k=950,
            repetition_penalty=1.2,
            eos_token_id=0,
            do_sample=True,
            num_return_sequences=5
        )

        # Decode and store the generated sequences
        for generated_output in outputs:
            generated_text = tokenizer.decode(generated_output, skip_special_tokens=True)
            generated_text = generated_text.replace("|endoftext|>", "")  # Remove the header
            file.write(generated_text + "\n")
            generated_sequences.append(generated_text)
