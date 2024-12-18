import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:True'

import torch
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from evaluate import load
import nltk
from nltk.translate.bleu_score import sentence_bleu
import shutil
import json
import gc

# CUDA setup
print("Checking CUDA availability...")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% GPU memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_dataset(dataset):
    """
    Preprocessing dataset with strict memory constraints
    """
    df = pd.DataFrame(dataset['train'])[['question_1', 'question_2']]
    
    # Limit initial dataset size
    df = df.head(300)  # Further reduced dataset size
    
    def is_valid_pair(row):
        q = row['question_1'].lower()
        a = row['question_2'].lower()
        
        # Basic validation
        if not q.endswith('?') or a.endswith('?'):
            return False
        if len(q) < 10 or len(a) < 20 or len(a) > 150:  # Reduced max length
            return False
            
        # Check for medical terms
        medical_terms = ['medical', 'health', 'doctor', 'symptom', 'treatment',
                        'disease', 'patient', 'medicine', 'hospital', 'pain']
        if not any(term in q + ' ' + a for term in medical_terms):
            return False
            
        return True
    
    # Apply validation
    df = df[df.apply(is_valid_pair, axis=1)]
    df = df.drop_duplicates(subset=['question_1'])
    
    # Simple format
    df['text'] = df.apply(lambda x: (
        f"Q: {x['question_1']}\nA: {x['question_2']}\n---\n"
    ), axis=1)
    
    df = df[['text']]
    gc.collect()
    
    return df

class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=96):  # Further reduced max_length
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        return {
            key: val[idx].clone().detach()
            for key, val in self.encodings.items()
        }

    def __len__(self):
        return len(self.encodings['input_ids'])

def setup_model_tokenizer():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Minimal tokens
    special_tokens = {
        'additional_special_tokens': ['[Q]', '[A]']
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def train_medical_model(train_dataset, val_dataset, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=128,
        eval_steps=100,          # Reduced evaluation frequency
        save_steps=100,          # Reduced saving frequency
        warmup_steps=200,        # Reduced warmup steps
        learning_rate=2e-5,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=1,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        fp16=False,
        optim="adamw_torch_fused",
        max_grad_norm=0.5,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    return trainer

def generate_medical_response(question, model, tokenizer, max_length=150):
    try:
        # Simple prompt format
        prompt = f"Q: {question}\nA:"

        inputs = tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=64,  # Limit input length
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("A:")[-1].strip()
        
        # Clean up memory
        del inputs, outputs
        torch.cuda.empty_cache()
        
        return response
    
    except Exception as e:
        print(f"Error in response generation: {str(e)}")
        return "I apologize, but I'm unable to generate a response at the moment."

def verify_dataset_quality(df):
    print("\nDataset Quality Report:")
    print("-----------------------")
    print(f"Total samples: {len(df)}")
    print(f"Average answer length: {df['text'].str.len().mean():.0f} characters")
    print("\nSample Q&A Pair:")
    sample = df.sample(1).iloc[0]
    print(f"\n{sample['text']}")
    return df

def use_medical_chatbot(model_dir):
    print("\nLoading model for chat...")
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    
    model.to(device)
    model.eval()

    print("\nMedical Chatbot Ready! Type 'exit' to end the conversation.")
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == 'exit':
            break
            
        if len(question) < 5:
            print("Please ask a more detailed question.")
            continue
            
        try:
            torch.cuda.empty_cache()  # Clear cache before generation
            answer = generate_medical_response(question, model, tokenizer)
            print(f"\nResponse: {answer}")
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try rephrasing your question.")

def main():
    try:
        # Initial cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        output_dir = "./medical_model"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        print("Loading dataset...")
        dataset = load_dataset("medical_questions_pairs")
        
        print("Preprocessing dataset...")
        df = preprocess_dataset(dataset)
        
        print("Verifying dataset quality...")
        df = verify_dataset_quality(df)
        
        df.to_csv('medical_qa_processed.csv', index=False)
        print("Dataset saved to medical_qa_processed.csv")

        print("Setting up model...")
        global model, tokenizer
        model, tokenizer = setup_model_tokenizer()
        
        # Enable memory optimizations
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        
        model.to(device)

        print("Preparing training data...")
        train_texts, val_texts = train_test_split(
            df['text'].tolist(),
            test_size=0.1,
            random_state=42
        )

        train_dataset = MedicalDataset(train_texts, tokenizer)
        val_dataset = MedicalDataset(val_texts, tokenizer)

        print("Starting training...")
        trainer = train_medical_model(train_dataset, val_dataset, output_dir)

        # Clear memory before training
        torch.cuda.empty_cache()
        gc.collect()

        print("Training model...")
        try:
            trainer.train()
            print("Training completed successfully!")
            
            print("Saving model...")
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            print("Starting chatbot mode...")
            use_medical_chatbot(output_dir)
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise e

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main()