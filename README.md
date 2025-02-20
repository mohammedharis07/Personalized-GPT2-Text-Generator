# Personalized-GPT2-Text-Generator
# GPT-2 Fine-Tuning for Custom Text Generation

This project fine-tunes OpenAI's GPT-2 model on a custom dataset to generate coherent and contextually relevant text based on a given prompt.

## 🚀 Features
- Fine-tunes GPT-2 on any custom text dataset.
- Supports training and evaluation with `Trainer` API.
- Generates high-quality text based on prompts.
- Saves and loads fine-tuned models for future use.

## 📂 Project Structure
📂 GPT2-Fine-Tuning ├── 📜 data.txt # Custom training dataset ├── 📜 train_gpt2.py # Main script for fine-tuning GPT-2 ├── 📂 gpt2-finetuned # Folder to store fine-tuned model ├── 📂 logs # Training logs ├── 📜 README.md # Project documentation └── 📜 requirements.txt # Required dependencies

## 🛠 Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/GPT2-Fine-Tuning.git
cd GPT2-Fine-Tuning
2️⃣ Install Dependencies
Ensure you have Python 3.8+ installed, then install dependencies:
pip install -r requirements.txt
3️⃣ Download GPT-2 Model
The script automatically downloads GPT-2 from Hugging Face, but you can manually download it:
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
🏋️ Fine-Tune GPT-2
Run the following command to train the model on your dataset:
python train_gpt2.py

🔮 Generate Text with the Fine-Tuned Model
After training, test the model using:
from transformers import pipeline

generator = pipeline("text-generation", model="./gpt2-finetuned", tokenizer="./gpt2-finetuned")
prompt = "Once upon a time"
result = generator(prompt, max_length=100, num_return_sequences=1)
print(result[0]['generated_text'])

📝 Customizing the Dataset
Modify data.txt with your own text to train GPT-2 on different writing styles.

🛠 Troubleshooting
FileNotFoundError: Ensure data.txt exists in the correct path.
Tokenizer Padding Error: Run tokenizer.pad_token = tokenizer.eos_token before training.
CUDA Memory Issues: Try reducing per_device_train_batch_size.
📜 License
This project is licensed under the MIT License.

⭐ Acknowledgments
Hugging Face Transformers
OpenAI's GPT-2
🔥 Contribute & Feedback
Want to improve this project? Feel free to fork, open an issue, or submit a pull request!

If you found this useful, give it a star ⭐ on GitHub!
