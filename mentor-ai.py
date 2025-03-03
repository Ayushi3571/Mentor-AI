# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from gensim import corpora, models, similarities
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#import urllib.request
import requests
#import urllib
#from urllib.request import urlopen
#import urllib3
import html
#import matplotlib.pyplot as plt
#from nltk.tokenize import sent_tokenize
#from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure necessary NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class CalculusMentorRAG:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.chunks = []
        self.chunk_sources = []  # To store section/page information
        self.dictionary = None
        self.tfidf = None
        self.index = None
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        print(f"Extracting text from {pdf_path}...")
        reader = PdfReader(pdf_path)
        text = ""
        
        # Extract text along with page numbers
        page_texts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                page_texts.append((i+1, page_text))
                text += page_text + "\n"
                
        print(f"Extracted {len(page_texts)} pages from PDF.")
        return text, page_texts
    
    def clean_textbook(self, txt_file_url, output_dir=None):
        """
        Clean and structure a textbook from a txt file.
        Returns a structured dictionary with chapters, sections, and content.
        Optionally saves cleaned chapters to separate files.
        """
        
        print(f"Cleaning textbook from {txt_file_url}...")
        
        # Read the file
        response = requests.get(txt_file_url)
        if (response.status_code):
            content = response.text
        
        # Replace special characters - 
        content = html.unescape(content)
        content = content.replace('\r', '')  # Remove carriage returns
        
        # Split by lines
        lines = content.split('\n')
        
        # Remove empty lines and standardize spacing
        lines = [line.strip() for line in lines if line.strip()]
        
        # Structure for the textbook
        textbook = {
            'title': 'Calculus Textbook',
            'chapters': []
        }
        
        chapter_content = ""
        chapter_num = 0
        chapter_title = ""
        
        # Process each line
        for line in lines[2407:]:
            # Check if this is a chapter heading
            if line == "CHAPTER":
                nxt = True
                continue
            if ("CHAPTER" in line) or nxt:   
                # add chapter content from last chapter:
                nxt = False
                if chapter_content:
                    textbook['chapters'][chapter_num-1]['content'] = chapter_content
                    chapter_content = ""
                # Create new chapter
                chapter_num += 1
                chapter_title = line
                current_chapter = {
                    'number': chapter_num,
                    'title': chapter_title,
                    'content': ""
                }
                textbook['chapters'].append(current_chapter)
                continue
            
            else:
                chapter_content += line
            
        '''
        # If no chapters were found, create a single chapter with all content
        if not textbook['chapters']:
            textbook['chapters'] = [{
                'number': '1',
                'title': 'Entire Textbook',
                'sections': [{
                    'title': 'Main Content',
                    'page_start': 1,
                    'page_end': current_page,
                    'content': content
                }]
            }]'''
        
        # Optionally save cleaned chapters to files
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            for chapter in textbook['chapters']:
                chapter_content = f"# {chapter['title']}\n\n"
                
                for section in chapter['sections']:
                    chapter_content += f"## {section['title']} (Pages {section['page_start']}-{section['page_end'] or '?'})\n\n"
                    chapter_content += section['content'] + "\n\n"
                
                filename = os.path.join(output_dir, f"Chapter_{chapter['number']}.txt")
                with open(filename, 'w', encoding='utf-8') as file:
                    file.write(chapter_content)
                print(f"Saved {filename}")
        
        print(f"Textbook cleaned and structured into {len(textbook['chapters'])} chapters")
        return textbook
    
    def preprocess_text(self, text):
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokenize and lemmatize
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return ' '.join(words)
    
    def chunk_text(self, textbook, chunk_size=500, overlap=100):
        """Split text into overlapping chunks and associate with page numbers."""
        self.chunks = []
        self.chunk_sources = []
        
        for chapter in textbook["chapters"]:
            # Create chunks from this chapter
            words = chapter['content'].split()
            for i in range(0, len(words), chunk_size - overlap):
                if i + chunk_size < len(words):
                    chunk = ' '.join(words[i:i+chunk_size])
                else:
                    chunk = ' '.join(words[i:])
                
                # Skip very short chunks
                if len(chunk.split()) < 20:
                    continue
                    
                self.chunks.append(chunk)
                self.chunk_sources.append({
                    'chapter': chapter['title'],
                    'chapter_number': chapter['number'],
                    'chunk_index': len(self.chunks) - 1
                })
        
        print(f"Created {len(self.chunks)} chunks from the textbook.")
        return self.chunks, self.chunk_sources
    
    def build_similarity_model(self):
        """Build Gensim similarity model from chunks."""
        print("Building similarity model...")
        
        # Preprocess chunks
        processed_chunks = [self.preprocess_text(chunk) for chunk in self.chunks]
        
        # Create dictionary and corpus
        self.dictionary = corpora.Dictionary([text.split() for text in processed_chunks])
        corpus = [self.dictionary.doc2bow(text.split()) for text in processed_chunks]
        
        # Create TF-IDF model
        self.tfidf = models.TfidfModel(corpus)
        corpus_tfidf = self.tfidf[corpus]
        
        # Create similarity index
        self.index = similarities.MatrixSimilarity(corpus_tfidf)
        
        print("Similarity model built successfully!")
        return self.dictionary, self.tfidf, self.index
    
    def retrieve_relevant_chunks(self, query, top_n=3):
        """Find chunks most relevant to the query."""
        if not self.dictionary or not self.tfidf or not self.index:
            raise ValueError("Similarity model not built yet. Call build_similarity_model() first.")
        
        # Preprocess the query
        processed_query = self.preprocess_text(query)
        
        # Convert query to bag-of-words format
        query_bow = self.dictionary.doc2bow(processed_query.split())
        
        # Transform using TF-IDF model
        query_tfidf = self.tfidf[query_bow]
        
        # Get similarity scores
        similarities_to_query = self.index[query_tfidf]
        
        # Get top N chunks
        top_indices = np.argsort(similarities_to_query)[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            # Check if the similarity is too low to be relevant
            similarity = similarities_to_query[idx]
            if similarity < 0.1:  # Threshold can be adjusted
                continue
                
            results.append({
                'chunk': self.chunks[idx],
                'similarity': float(similarity),
                'source': self.chunk_sources[idx],
            })
            
        return results
    
    def answer_question(self, question, top_n=3):
        """Generate an answer to a calculus question using retrieved chunks."""
        # Retrieve relevant passages
        relevant_chunks = self.retrieve_relevant_chunks(question, top_n=top_n)
        
        if not relevant_chunks:
            return "I couldn't find information in the textbook to answer this question."
        
        # Format the retrieved information
        answer = f"Here are the most relevant sections from the calculus textbook:\n\n"
        
        for i, chunk in enumerate(relevant_chunks):
            answer += f"Excerpt {i+1} (Page {chunk['source']['chapter_number'], chunk['source']['chapter']}, {chunk['source']['chunk_index']}):\n"
            answer += f"{chunk['chunk'][:500]}...\n\n"
            
        answer += "Based on these sections from the textbook, this should help address your question."
        return answer
    
    def train_from_txt(self, link, chunk_size=500, overlap=100):
        """One-step function to train the model from a PDF file."""
        
        textbook = self.clean_textbook(link)
        
        # Chunk the text
        self.chunk_text(textbook)
        
        # Build the similarity model
        self.build_similarity_model()
        
        print(f"Training complete. Ready to answer questions about calculus!")
        return True
    
    def save_model(self, directory="model"):
        """Save the model to disk."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save dictionary
        self.dictionary.save(os.path.join(directory, "dictionary.gensim"))
        
        # Save TF-IDF model
        self.tfidf.save(os.path.join(directory, "tfidf.gensim"))
        
        # Save index
        self.index.save(os.path.join(directory, "index.gensim"))
        
        # Save chunks and sources
        pd.DataFrame({
            'chunk': self.chunks,
            #'page': [source['page'] for source in self.chunk_sources],
            'section': [source['chapter'] for source in self.chunk_sources],
            'chunk_index': [source['chunk_index'] for source in self.chunk_sources]
        }).to_csv(os.path.join(directory, "chunks.csv"), index=False)
        
        print(f"Model saved to {directory}")
        
    def load_model(self, directory="model"):
        """Load the model from disk."""
        # Load dictionary
        self.dictionary = corpora.Dictionary.load(os.path.join(directory, "dictionary.gensim"))
        
        # Load TF-IDF model
        self.tfidf = models.TfidfModel.load(os.path.join(directory, "tfidf.gensim"))
        
        # Load index
        self.index = similarities.MatrixSimilarity.load(os.path.join(directory, "index.gensim"))
        
        # Load chunks and sources
        chunks_df = pd.read_csv(os.path.join(directory, "chunks.csv"))
        self.chunks = chunks_df['chunk'].tolist()
        
        self.chunk_sources = []
        for _, row in chunks_df.iterrows():
            self.chunk_sources.append({
                'page': row['page'],
                'section': row['section'],
                'chunk_index': row['chunk_index']
            })
            
        print(f"Model loaded from {directory}")
        return True

# Example usage
if __name__ == "__main__":
    
    mentor = CalculusMentorRAG()
    link = "https://archive.org/stream/CalculusSpivak/Calculus%20-%20Spivak_djvu.txt"
    
    # Train or load the model
    if os.path.exists("model/dictionary.gensim"):
        print("Loading existing model...")
        mentor.load_model()
    else:
        print("Training new model...")
        mentor.train_from_txt(link)
        mentor.save_model()

    # Interactive question answering loop
    print("\nCalculus Mentor is ready! Type 'quit' to exit.")
    while True:
        question = input("\nAsk a calculus question: ")
        if question.lower() == 'quit':
            break
            
        answer = mentor.answer_question(question)
        print("\n" + answer)
        
'''
1. The txt source is messing up symbols, we need official textbook
2. llama path: "C:/Users/ayush/.ollama/models/blobs/sha256-00e1317cbf74d901080d7100f57580ba8dd8de57203072dc6f668324ba545f29"

'''