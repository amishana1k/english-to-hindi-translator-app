import tkinter as tk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from datetime import datetime

# Load the trained model and tokenizers
model = load_model("english_to_hindi_model.h5")

with open("english_tokenizer.pkl", "rb") as f:
    english_tokenizer = pickle.load(f)

with open("hindi_tokenizer.pkl", "rb") as f:
    hindi_tokenizer = pickle.load(f)

# Create a reverse mapping for Hindi tokenizer
reverse_hindi_tokenizer = {v: k for k, v in hindi_tokenizer.word_index.items()}

# Function to check if the word starts with a vowel
def starts_with_vowel(word):
    return word[0].lower() in 'aeiou'

# GUI function for translation
def translate_word():
    input_word = entry.get().strip()
    
    # Check the current time
    current_hour = datetime.now().hour
    
    # Handle words that start with a vowel
    if starts_with_vowel(input_word):
        if not (21 <= current_hour < 22):  # 9 PM to 10 PM logic
            output_label.config(text="This word starts with a vowel. Please provide another word.")
            return
    
    # Convert input word to sequence
    input_sequence = english_tokenizer.texts_to_sequences([input_word])
    input_padded = pad_sequences(input_sequence, maxlen=model.input_shape[1], padding='post')
    
    # Predict using the model
    prediction = model.predict(input_padded)
    hindi_indices = np.argmax(prediction, axis=-1)
    
    # Convert indices to Hindi words
    hindi_word = ''.join([reverse_hindi_tokenizer.get(idx, '') for idx in hindi_indices.flatten()])
    if not hindi_word.strip():
        hindi_word = "Translation not available."
    
    output_label.config(text=f"Hindi Translation: {hindi_word}")

# GUI setup
root = tk.Tk()
root.title("English to Hindi Translator")

tk.Label(root, text="Enter an English word:").pack()
entry = tk.Entry(root)
entry.pack()

translate_button = tk.Button(root, text="Translate", command=translate_word)
translate_button.pack()

output_label = tk.Label(root, text="")
output_label.pack()

root.mainloop()
