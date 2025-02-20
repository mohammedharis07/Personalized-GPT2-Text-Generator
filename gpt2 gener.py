from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# ✅ Set the absolute path to your dataset file
file_path = r"C:\Users\Mohammed Haris\OneDrive\Desktop\progidy\gpt-2\data.txt"

# ✅ Load dataset from text file
dataset = load_dataset('text', data_files={'train': file_path})

# ✅ Split dataset: 90% train, 10% validation
split_dataset = dataset["train"].train_test_split(test_size=0.1)

# Get train and validation datasets
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# ✅ Load tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# ✅ Fix padding issue
tokenizer.pad_token = tokenizer.eos_token  # ✅ Set pad_token explicitly

# ✅ Tokenization function (modified to include truncation and labels)
def tokenize_function(examples):
    tokenized_output = tokenizer(
        examples["text"],
        truncation=True,  # ✅ Explicitly set truncation
        padding="max_length",
        max_length=512
    )
    
    # ✅ Set labels to match input_ids (important for computing loss)
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    
    return tokenized_output

# ✅ Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# ✅ Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",  # ✅ Evaluate after each epoch
    save_strategy="epoch",
    num_train_epochs=3,
    logging_dir="./logs"
)

# ✅ Define trainer with train and eval datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval  # ✅ Now evaluation works properly
)

# ✅ Train the model
trainer.train()

# ✅ Save the fine-tuned model
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

# ✅ Test the fine-tuned model
from transformers import pipeline

generator = pipeline("text-generation", model="./gpt2-finetuned", tokenizer="./gpt2-finetuned")
prompt = "i am a legend"
result = generator(prompt, max_length=100, num_return_sequences=1)

print(result[0]['generated_text'])
