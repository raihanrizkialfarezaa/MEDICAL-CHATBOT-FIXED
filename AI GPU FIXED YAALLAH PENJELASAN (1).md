## 1. PERSIAPAN DASAR DAN KONFIGURASI LINGKUNGAN

### 1.1 Konfigurasi Memori GPU
```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:True'
```
Baris kode ini melakukan pengaturan awal untuk manajemen memori GPU:
- `PYTORCH_CUDA_ALLOC_CONF`: Variabel lingkungan yang mengatur alokasi CUDA untuk PyTorch
- `max_split_size_mb:32`: Membatasi ukuran alokasi memori maksimum menjadi 32 megabyte, mencegah penggunaan memori yang berlebihan
- `expandable_segments:True`: Mengizinkan segmen memori untuk berkembang secara dinamis sesuai kebutuhan
- Pengaturan ini penting untuk mencegah crash program akibat kehabisan memori GPU

### 1.2 Import Library
```python
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
```

Setiap library memiliki fungsi spesifik:
1. `torch`: Framework utama untuk deep learning, menyediakan komputasi tensor dan pemrosesan neural network
2. `datasets`: Library untuk mengelola dan memuat dataset dengan efisien
3. `pandas`: Library untuk manipulasi dan analisis data dalam format tabel (DataFrame)
4. `numpy`: Library untuk komputasi numerik dan operasi array
5. `sklearn.model_selection`: Menyediakan fungsi untuk membagi dataset menjadi data training dan validasi
6. `transformers`: Library utama yang menyediakan model-model transformers seperti GPT-2
7. `nltk`: Natural Language Toolkit untuk pemrosesan bahasa alami
8. `shutil`: Untuk operasi file sistem seperti copy dan delete
9. `json`: Untuk membaca dan menulis file JSON
10. `gc`: Garbage Collector untuk manajemen memori

### 1.3 Konfigurasi dan Pengecekan CUDA
```python
print("Checking CUDA availability...")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.set_per_process_memory_fraction(0.7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Bagian ini melakukan serangkaian pengecekan dan pengaturan GPU:
1. `torch.cuda.is_available()`: Memeriksa apakah sistem memiliki GPU NVIDIA dengan CUDA
2. `torch.cuda.get_device_name(0)`: Mendapatkan nama GPU yang terdeteksi (indeks 0 untuk GPU pertama)
3. `torch.cuda.empty_cache()`: Membersihkan cache GPU untuk membebaskan memori
4. `gc.collect()`: Menjalankan garbage collector untuk membersihkan objek Python yang tidak terpakai
5. `torch.cuda.set_per_process_memory_fraction(0.7)`: Membatasi penggunaan memori GPU hingga 70%
6. `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`: Menentukan perangkat yang akan digunakan (GPU jika tersedia, CPU jika tidak)

### 1.4 Fungsi Preprocessing Dataset
```python
def preprocess_dataset(dataset):
    """
    Preprocessing dataset with strict memory constraints
    """
    # Mengambil kolom yang diperlukan
    df = pd.DataFrame(dataset['train'])[['question_1', 'question_2']]
    
    # Membatasi ukuran dataset untuk efisiensi
    df = df.head(300)
    
    def is_valid_pair(row):
        q = row['question_1'].lower()
        a = row['question_2'].lower()
        
        # Validasi dasar
        if not q.endswith('?') or a.endswith('?'):
            return False
        if len(q) < 10 or len(a) < 20 or len(a) > 150:
            return False
            
        # Memeriksa keberadaan istilah medis
        medical_terms = ['medical', 'health', 'doctor', 'symptom', 'treatment',
                        'disease', 'patient', 'medicine', 'hospital', 'pain']
        if not any(term in q + ' ' + a for term in medical_terms):
            return False
            
        return True
```

Fungsi preprocessing ini memiliki beberapa tahap penting:
1. Ekstraksi Data:
   - Mengambil hanya kolom 'question_1' (pertanyaan) dan 'question_2' (jawaban)
   - Membatasi dataset menjadi 300 baris untuk menghindari masalah memori

2. Validasi Pasangan Data:
   - Mengkonversi teks ke lowercase untuk konsistensi
   - Memeriksa format pertanyaan (harus diakhiri tanda tanya)
   - Memeriksa format jawaban (tidak boleh diakhiri tanda tanya)
   - Memvalidasi panjang teks:
     * Pertanyaan minimal 10 karakter
     * Jawaban antara 20-150 karakter
   - Memeriksa relevansi medis dengan mencari istilah-istilah medis

## 2. IMPLEMENTASI DATASET DAN MODEL

### 2.1 Kelas MedicalDataset
```python
class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=96):
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
```

Kelas ini merupakan implementasi custom dataset untuk data medis:
1. Constructor (`__init__`):
   - Menerima parameter:
     * `texts`: List teks pertanyaan dan jawaban
     * `tokenizer`: Objek tokenizer untuk mengubah teks menjadi token
     * `max_length`: Panjang maksimum sequence (default: 96 token)
   - Melakukan tokenisasi dengan parameter:
     * `padding=True`: Menambahkan padding agar semua sequence sama panjang
     * `truncation=True`: Memotong teks yang terlalu panjang
     * `return_tensors="pt"`: Mengembalikan hasil dalam format PyTorch tensor

2. Method `__getitem__`:
   - Mengambil data pada indeks tertentu
   - Menggunakan `clone().detach()` untuk memastikan tensor independen dari sumber
   - Mengembalikan dictionary berisi token-token untuk model

3. Method `__len__`:
   - Mengembalikan jumlah total data dalam dataset

### 2.2 Setup Model dan Tokenizer
```python
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
```

Fungsi ini melakukan inisialisasi model dan tokenizer:
1. Pemilihan Model:
   - Menggunakan model dasar "gpt2"
   - `GPT2Tokenizer`: Untuk mengubah teks menjadi token yang dipahami model
   - `GPT2LMHeadModel`: Model GPT-2 dengan language modeling head

2. Konfigurasi Token Khusus:
   - Menambahkan token spesial '[Q]' untuk pertanyaan dan '[A]' untuk jawaban
   - `resize_token_embeddings`: Menyesuaikan ukuran embedding layer untuk token baru
   - Menetapkan padding token sama dengan end-of-sequence token

### 2.3 Fungsi Training Model
```python
def train_medical_model(train_dataset, val_dataset, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=128,
        eval_steps=100,
        save_steps=100,
        warmup_steps=200,
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
```

Parameter training dikonfigurasi dengan sangat hati-hati:
1. Parameter Dasar:
   - `num_train_epochs`: 5 epoch training
   - `per_device_train_batch_size=1`: Batch size kecil untuk menghemat memori
   - `gradient_accumulation_steps=128`: Mengakumulasi gradien untuk simulasi batch besar

2. Parameter Optimisasi:
   - `learning_rate=2e-5`: Learning rate yang relatif kecil untuk fine-tuning
   - `warmup_steps=200`: Pemanasan gradual untuk stabilitas training
   - `max_grad_norm=0.5`: Clipping gradien untuk mencegah eksplosif

3. Parameter Evaluasi dan Penyimpanan:
   - `eval_steps=100`: Evaluasi setiap 100 langkah
   - `save_steps=100`: Menyimpan checkpoint setiap 100 langkah
   - `save_total_limit=1`: Menyimpan hanya 1 checkpoint terbaik


## 3. GENERASI RESPONS DAN EVALUASI MODEL

### 3.1 Fungsi Generasi Respons Medis
```python
def generate_medical_response(question, model, tokenizer, max_length=150):
    try:
        # Format prompt sederhana
        prompt = f"Q: {question}\nA:"

        inputs = tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=64,
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
        
        # Pembersihan memori
        del inputs, outputs
        torch.cuda.empty_cache()
        
        return response
    
    except Exception as e:
        print(f"Error in response generation: {str(e)}")
        return "I apologize, but I'm unable to generate a response at the moment."
```

Fungsi ini memiliki beberapa komponen penting:
1. Persiapan Input:
   - Membuat format prompt dengan struktur "Q: [pertanyaan]\nA:"
   - Mengkodekan prompt menjadi token dengan batasan panjang 64
   - Memindahkan input ke device yang sesuai (GPU/CPU)

2. Konfigurasi Generasi:
   - `max_length=150`: Panjang maksimal respons
   - `num_return_sequences=1`: Menghasilkan 1 respons
   - `no_repeat_ngram_size=3`: Mencegah pengulangan frasa 3-gram
   - `do_sample=True`: Menggunakan sampling probabilistik
   - `top_p=0.9`: Nucleus sampling dengan probabilitas 0.9
   - `top_k=50`: Membatasi 50 token dengan probabilitas tertinggi
   - `temperature=0.7`: Mengatur tingkat kreativitas output

3. Manajemen Memori:
   - Menggunakan `torch.no_grad()` untuk efisiensi memori
   - Menghapus variabel tidak terpakai
   - Membersihkan cache GPU setelah generasi

### 3.2 Fungsi Verifikasi Kualitas Dataset
```python
def verify_dataset_quality(df):
    print("\nDataset Quality Report:")
    print("-----------------------")
    print(f"Total samples: {len(df)}")
    print(f"Average answer length: {df['text'].str.len().mean():.0f} characters")
    print("\nSample Q&A Pair:")
    sample = df.sample(1).iloc[0]
    print(f"\n{sample['text']}")
    return df
```

Fungsi ini melakukan:
1. Pelaporan Statistik:
   - Menghitung jumlah total sampel
   - Menghitung rata-rata panjang jawaban
2. Sampling Data:
   - Menampilkan contoh pasangan Q&A secara acak
3. Validasi Visual:
   - Memungkinkan pemeriksaan manual kualitas data

### 3.3 Fungsi Chatbot Medis
```python
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
            torch.cuda.empty_cache()
            answer = generate_medical_response(question, model, tokenizer)
            print(f"\nResponse: {answer}")
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try rephrasing your question.")
```

Fungsi ini mengimplementasikan interface chatbot:
1. Inisialisasi:
   - Memuat model dan tokenizer dari direktori yang ditentukan
   - Memindahkan model ke device yang sesuai
   - Mengatur model ke mode evaluasi

2. Loop Interaksi:
   - Menerima input pertanyaan dari pengguna
   - Memvalidasi panjang pertanyaan
   - Membersihkan cache GPU sebelum generasi
   - Menangani error dengan graceful error handling


## 3. GENERASI RESPONS DAN EVALUASI MODEL

### 3.1 Fungsi Generasi Respons Medis
```python
def generate_medical_response(question, model, tokenizer, max_length=150):
    try:
        # Format prompt sederhana
        prompt = f"Q: {question}\nA:"

        inputs = tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=64,
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
        
        # Pembersihan memori
        del inputs, outputs
        torch.cuda.empty_cache()
        
        return response
    
    except Exception as e:
        print(f"Error in response generation: {str(e)}")
        return "I apologize, but I'm unable to generate a response at the moment."
```

Fungsi ini memiliki beberapa komponen penting:
1. Persiapan Input:
   - Membuat format prompt dengan struktur "Q: [pertanyaan]\nA:"
   - Mengkodekan prompt menjadi token dengan batasan panjang 64
   - Memindahkan input ke device yang sesuai (GPU/CPU)

2. Konfigurasi Generasi:
   - `max_length=150`: Panjang maksimal respons
   - `num_return_sequences=1`: Menghasilkan 1 respons
   - `no_repeat_ngram_size=3`: Mencegah pengulangan frasa 3-gram
   - `do_sample=True`: Menggunakan sampling probabilistik
   - `top_p=0.9`: Nucleus sampling dengan probabilitas 0.9
   - `top_k=50`: Membatasi 50 token dengan probabilitas tertinggi
   - `temperature=0.7`: Mengatur tingkat kreativitas output

3. Manajemen Memori:
   - Menggunakan `torch.no_grad()` untuk efisiensi memori
   - Menghapus variabel tidak terpakai
   - Membersihkan cache GPU setelah generasi

### 3.2 Fungsi Verifikasi Kualitas Dataset
```python
def verify_dataset_quality(df):
    print("\nDataset Quality Report:")
    print("-----------------------")
    print(f"Total samples: {len(df)}")
    print(f"Average answer length: {df['text'].str.len().mean():.0f} characters")
    print("\nSample Q&A Pair:")
    sample = df.sample(1).iloc[0]
    print(f"\n{sample['text']}")
    return df
```

Fungsi ini melakukan:
1. Pelaporan Statistik:
   - Menghitung jumlah total sampel
   - Menghitung rata-rata panjang jawaban
2. Sampling Data:
   - Menampilkan contoh pasangan Q&A secara acak
3. Validasi Visual:
   - Memungkinkan pemeriksaan manual kualitas data

### 3.3 Fungsi Chatbot Medis
```python
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
            torch.cuda.empty_cache()
            answer = generate_medical_response(question, model, tokenizer)
            print(f"\nResponse: {answer}")
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try rephrasing your question.")
```

Fungsi ini mengimplementasikan interface chatbot:
1. Inisialisasi:
   - Memuat model dan tokenizer dari direktori yang ditentukan
   - Memindahkan model ke device yang sesuai
   - Mengatur model ke mode evaluasi

2. Loop Interaksi:
   - Menerima input pertanyaan dari pengguna
   - Memvalidasi panjang pertanyaan
   - Membersihkan cache GPU sebelum generasi
   - Menangani error dengan graceful error handling


## 4. FUNGSI UTAMA DAN ALUR EKSEKUSI PROGRAM

### 4.1 Fungsi Main
```python
def main():
    try:
        # Pembersihan awal
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
```

Bagian pertama fungsi main melakukan:
1. Inisialisasi Lingkungan:
   - Membersihkan cache GPU dan memori
   - Menghapus dan membuat ulang direktori output
   
2. Persiapan Dataset:
   - Memuat dataset medis
   - Melakukan preprocessing
   - Memverifikasi kualitas data
   - Menyimpan dataset yang telah diproses

```python
        print("Setting up model...")
        global model, tokenizer
        model, tokenizer = setup_model_tokenizer()
        
        # Mengaktifkan optimasi memori
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
```

Bagian kedua melakukan:
1. Persiapan Model:
   - Inisialisasi model dan tokenizer
   - Mengaktifkan optimasi memori dengan gradient checkpointing
   - Menonaktifkan cache untuk penghematan memori
   
2. Persiapan Data Training:
   - Membagi dataset menjadi data training (90%) dan validasi (10%)
   - Membuat objek dataset untuk training dan validasi

```python
        print("Starting training...")
        trainer = train_medical_model(train_dataset, val_dataset, output_dir)

        # Membersihkan memori sebelum training
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
```

Bagian terakhir melakukan:
1. Proses Training:
   - Inisialisasi trainer
   - Membersihkan memori sebelum training
   - Melakukan proses training model
   
2. Penyimpanan dan Deployment:
   - Menyimpan model yang telah dilatih
   - Memulai mode chatbot untuk interaksi
   
3. Penanganan Error:
   - Menangkap dan menampilkan error yang mungkin terjadi
   - Membersihkan memori jika terjadi error

### 4.2 Entry Point Program
```python
if __name__ == "__main__":
    main()
```
- Memastikan fungsi main() hanya dijalankan jika file dieksekusi langsung
- Tidak akan dijalankan jika file diimpor sebagai modul

## 5. FITUR KEAMANAN DAN OPTIMASI

1. Manajemen Memori yang Ketat:
   - Pembersihan cache GPU secara regular
   - Penggunaan gradient checkpointing
   - Batasan ukuran batch dan dataset

2. Penanganan Error yang Komprehensif:
   - Try-catch block di setiap operasi kritis
   - Pesan error yang informatif
   - Pembersihan memori otomatis saat error

3. Validasi Input:
   - Pemeriksaan panjang pertanyaan
   - Validasi format data
   - Pembatasan ukuran output

Program ini dirancang dengan mempertimbangkan aspek performa, keamanan, dan kehandalan, sambil tetap menjaga efisiensi penggunaan sumber daya komputasi.
