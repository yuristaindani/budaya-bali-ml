# KODE TIGA BAGUS BANGET
# import os
# from dotenv import load_dotenv
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI 
# from translate_func import translate_to_english, translate_from_english, detect_language, is_indonesian, translate_to_target_language
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# load_dotenv()

# class DualLanguageRAG:
#     def __init__(self):
#         # Initialize embedding model
#         self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
#         # Load both FAISS indexes
#         try:
#             self.vectordb_indonesia = FAISS.load_local(
#                 "faiss_index_indonesia", 
#                 self.embedding, 
#                 allow_dangerous_deserialization=True
#             )
#             self.retriever_indonesia = self.vectordb_indonesia.as_retriever(
#                 search_type="similarity", 
#                 search_kwargs={"k": 4}
#             )
#             print("✅ Indonesian FAISS index loaded successfully")
#         except Exception as e:
#             print(f"❌ Error loading Indonesian index: {e}")
#             self.vectordb_indonesia = None
#             self.retriever_indonesia = None
        
#         try:
#             self.vectordb_english = FAISS.load_local(
#                 "faiss_index_english", 
#                 self.embedding, 
#                 allow_dangerous_deserialization=True
#             )
#             self.retriever_english = self.vectordb_english.as_retriever(
#                 search_type="similarity", 
#                 search_kwargs={"k": 4}
#             )
#             print("✅ English FAISS index loaded successfully")
#         except Exception as e:
#             print(f"❌ Error loading English index: {e}")
#             self.vectordb_english = None
#             self.retriever_english = None
        
#         # Initialize LLM
#         self.llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)
        
#         # Prompt templates for different languages
#         self.template_indonesian = """
# Anda adalah asisten yang membantu. Silakan jawab pertanyaan berdasarkan HANYA pada konteks yang diberikan.
# Jika jawaban tidak ada dalam konteks, jangan membuat jawaban. Berikan jawaban yang ringkas dan sopan dalam bahasa Indonesia.

# Setelah jawaban, tambahkan tag:
# - Jika Anda menggunakan konteks, akhiri dengan [SOURCES USED]
# - Jika konteks tidak berguna, akhiri dengan [NO SOURCE USED]

# Konteks:
# {context}

# Pertanyaan:
# {question}

# Jawaban:
# """
        
#         self.template_english = """
# You are a helpful assistant. Please answer the question based ONLY on the provided context.
# If the answer is not in the context, don't make up an answer. Be concise and polite.

# After the answer, add a tag:
# - If you used the context, end with [SOURCES USED]
# - If the context was not useful, end with [NO SOURCE USED]

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
        
#         # Create prompt templates
#         self.prompt_indonesian = PromptTemplate(
#             template=self.template_indonesian, 
#             input_variables=["context", "question"]
#         )
#         self.prompt_english = PromptTemplate(
#             template=self.template_english, 
#             input_variables=["context", "question"]
#         )
        
#         # Create QA chains
#         if self.retriever_indonesia:
#             self.qa_chain_indonesia = RetrievalQA.from_chain_type(
#                 llm=self.llm,
#                 retriever=self.retriever_indonesia,
#                 chain_type="stuff",
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": self.prompt_indonesian},
#             )
        
#         if self.retriever_english:
#             self.qa_chain_english = RetrievalQA.from_chain_type(
#                 llm=self.llm,
#                 retriever=self.retriever_english,
#                 chain_type="stuff",
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": self.prompt_english},
#             )

#     def _calculate_cosine_similarity(self, text1, text2):
#         """Calculate cosine similarity between two texts using the embedding model"""
#         # Get embeddings for both texts
#         emb1 = self.embedding.embed_query(text1)
#         emb2 = self.embedding.embed_query(text2)
        
#         # Reshape for cosine similarity calculation
#         emb1 = np.array(emb1).reshape(1, -1)
#         emb2 = np.array(emb2).reshape(1, -1)
        
#         # Calculate and return cosine similarity
#         return cosine_similarity(emb1, emb2)[0][0]

#     def query_rag_dual_language(self, user_question: str) -> dict:
#         """Enhanced RAG function with dual language support"""
        
#         # Detect input language
#         input_lang = detect_language(user_question)
#         print(f"\n🌐 Detected Language: {input_lang}")
        
#         # Determine which dataset to use
#         use_indonesian_dataset = is_indonesian(user_question)
#         print(f"📚 Using dataset: {'Indonesian' if use_indonesian_dataset else 'English'}")
        
#         if use_indonesian_dataset and self.retriever_indonesia:
#             # Use Indonesian dataset for Indonesian questions
#             print("🔍 Searching in Indonesian dataset...")
#             result = self.qa_chain_indonesia.invoke(user_question)
#             full_answer = result['result'].strip()
            
#         elif not use_indonesian_dataset and self.retriever_english:
#             # Use English dataset for non-Indonesian questions
#             print("🔍 Searching in English dataset...")
            
#             # Translate question to English for search
#             question_en = translate_to_english(user_question, input_lang)
#             print(f"🔄 Translated question: {question_en}")
            
#             result = self.qa_chain_english.invoke(question_en)
#             full_answer_en = result['result'].strip()
            
#             # Translate answer back to user's language if needed
#             if input_lang != 'en':
#                 # Extract clean answer without tags
#                 if "[SOURCES USED]" in full_answer_en:
#                     clean_answer_en = full_answer_en.replace("[SOURCES USED]", "").strip()
#                     tag = "[SOURCES USED]"
#                 elif "[NO SOURCE USED]" in full_answer_en:
#                     clean_answer_en = full_answer_en.replace("[NO SOURCE USED]", "").strip()
#                     tag = "[NO SOURCE USED]"
#                 else:
#                     clean_answer_en = full_answer_en
#                     tag = ""
                
#                 # Translate clean answer
#                 translated_answer = translate_from_english(clean_answer_en, input_lang)
#                 full_answer = translated_answer + (" " + tag if tag else "")
#                 print(f"🔄 Translated answer: {translated_answer}")
#             else:
#                 full_answer = full_answer_en
                
#         else:
#             # Fallback if appropriate dataset is not available
#             error_msg = "Maaf, dataset yang sesuai tidak tersedia." if use_indonesian_dataset else "Sorry, the appropriate dataset is not available."
#             return {
#                 "input_language": input_lang,
#                 "question": user_question,
#                 "answer": error_msg,
#                 "sources": [],
#                 "dataset_used": "none"
#             }
        
#         # Check if model used sources
#         if "[SOURCES USED]" in full_answer:
#             use_sources = True
#             clean_answer = full_answer.replace("[SOURCES USED]", "").strip()
#         elif "[NO SOURCE USED]" in full_answer:
#             use_sources = False
#             clean_answer = full_answer.replace("[NO SOURCE USED]", "").strip()
#         else:
#             use_sources = False
#             clean_answer = full_answer
        
#         # Process source documents
#         sources = []
#         if use_sources and result.get("source_documents"):
#             print("\n=== DEBUGGING: SOURCE DOCUMENTS ===")
#             for i, doc in enumerate(result["source_documents"], 1):
#                 print(f"[{i}] TITLE:", doc.metadata.get("title"))
#                 print("     URL  :", doc.metadata.get("url"))
#                 print("     IMAGE:", doc.metadata.get("image"))
#                 print("     SAMPLE:", doc.page_content[:80].replace("\n", " "), "...\n")
            
#             # Filter and prepare sources with cosine similarity
#             sources = self._filter_relevant_sources(
#                 result["source_documents"], 
#                 clean_answer, 
#                 user_question if use_indonesian_dataset else question_en if 'question_en' in locals() else user_question
#             )
        
#         return {
#             "input_language": input_lang,
#             "question": user_question,
#             "answer": clean_answer,
#             "sources": sources,
#             "dataset_used": "indonesian" if use_indonesian_dataset else "english"
#         }
    
#     def _filter_relevant_sources(self, source_documents, answer, question):
#         """Filter and rank sources that are actually relevant to the answer"""
#         answer_lower = answer.lower()
#         question_lower = question.lower()
#         question_keywords = set(question_lower.split())
        
#         # Extract key phrases from answer (sentences or important phrases)
#         answer_phrases = []
#         sentences = answer.split('.')
#         for sentence in sentences:
#             if len(sentence.strip()) > 10:  # Skip very short sentences
#                 answer_phrases.append(sentence.strip().lower())
        
#         url_to_doc = {}
        
#         # Get best version of each document per URL
#         for doc in source_documents:
#             url = doc.metadata.get("url", "No URL")
#             image = doc.metadata.get("image", "")
#             if url not in url_to_doc or (not url_to_doc[url].metadata.get("image") and image):
#                 url_to_doc[url] = doc
        
#         # Calculate relevance scores for each document
#         document_scores = []
        
#         for url, doc in url_to_doc.items():
#             snippet = doc.page_content.strip().lower()
#             title = doc.metadata.get("title", "").lower()
#             image = doc.metadata.get("image", "")
            
#             # Calculate different relevance scores
#             relevance_score = self._calculate_document_relevance(
#                 doc, answer_lower, answer_phrases, question_lower, question_keywords
#             )
            
#             # Calculate cosine similarity between question and document content
#             cosine_sim = self._calculate_cosine_similarity(question, doc.page_content)
            
#             # Combine scores (you can adjust weights as needed)
#             combined_score = (relevance_score * 0.7) + (cosine_sim * 0.3)
            
#             # Only include documents with minimum relevance
#             if combined_score > 0.1:  # Threshold for relevance
#                 document_scores.append({
#                     "doc": doc,
#                     "score": combined_score,
#                     "relevance_score": relevance_score,
#                     "cosine_similarity": cosine_sim,
#                     "title": doc.metadata.get("title", "No title"),
#                     "url": url,
#                     "image": image
#                 })
        
#         # Sort by combined score (highest first)
#         document_scores.sort(key=lambda x: x['score'], reverse=True)
        
#         # Convert to final source format
#         sources = []
#         for doc_info in document_scores:
#             sources.append({
#                 "title": doc_info["title"],
#                 "url": doc_info["url"],
#                 "image": doc_info["image"],
#                 "relevance_score": doc_info["relevance_score"],
#                 "cosine_similarity": doc_info["cosine_similarity"]
#             })
            
#         print(f"\n=== SOURCE RANKING DEBUG ===")
#         for i, doc_info in enumerate(document_scores[:3], 1):
#             print(f"[{i}] SCORE: {doc_info['score']:.3f} (Relevance: {doc_info['relevance_score']:.3f}, Cosine: {doc_info['cosine_similarity']:.3f})")
#             print(f"    TITLE: {doc_info['title']}")
#             print(f"    URL: {doc_info['url']}\n")
        
#         return sources
    
#     def _calculate_document_relevance(self, doc, answer_lower, answer_phrases, question_lower, question_keywords):
#         """Calculate how relevant a document is to the answer and question"""
#         snippet = doc.page_content.strip().lower()
#         title = doc.metadata.get("title", "").lower()
        
#         total_score = 0.0
        
#         # 1. Direct content overlap with answer (highest weight)
#         content_overlap_score = 0.0
#         for phrase in answer_phrases:
#             if len(phrase) > 5:  # Only consider meaningful phrases
#                 # Check for exact phrase matches
#                 if phrase in snippet:
#                     content_overlap_score += 3.0
#                 # Check for partial matches (word overlap)
#                 phrase_words = set(phrase.split())
#                 snippet_words = set(snippet.split())
#                 overlap_ratio = len(phrase_words & snippet_words) / len(phrase_words) if phrase_words else 0
#                 if overlap_ratio > 0.5:  # More than 50% word overlap
#                     content_overlap_score += overlap_ratio * 2.0
        
#         # 2. Title relevance to question (high weight)
#         title_score = 0.0
#         title_words = set(title.split())
#         title_question_overlap = len(question_keywords & title_words) / len(question_keywords) if question_keywords else 0
#         title_score = title_question_overlap * 2.5
        
#         # 3. Content relevance to question (medium weight)
#         question_score = 0.0
#         snippet_words = set(snippet.split())
#         question_content_overlap = len(question_keywords & snippet_words) / len(question_keywords) if question_keywords else 0
#         question_score = question_content_overlap * 1.5
        
#         # 4. Specific keyword matching (medium weight)
#         keyword_score = 0.0
#         for keyword in question_keywords:
#             if len(keyword) > 2:  # Skip very short words
#                 if keyword in title:
#                     keyword_score += 1.0  # Title match is more important
#                 elif keyword in snippet:
#                     keyword_score += 0.5
        
#         # 5. Answer-specific content matching (highest weight)
#         answer_content_score = 0.0
#         # Check if document content appears in the answer
#         doc_sentences = snippet.split('.')
#         for sentence in doc_sentences:
#             if len(sentence.strip()) > 10:
#                 sentence_clean = sentence.strip().lower()
#                 if sentence_clean in answer_lower:
#                     answer_content_score += 4.0  # Very high score for direct usage
#                 else:
#                     # Check for word overlap with answer
#                     sentence_words = set(sentence_clean.split())
#                     answer_words = set(answer_lower.split())
#                     overlap = len(sentence_words & answer_words)
#                     if overlap > 3:  # Significant word overlap
#                         answer_content_score += (overlap / len(sentence_words)) * 2.0
        
#         # Combine all scores
#         total_score = (
#             answer_content_score * 0.4 +    # 40% weight - most important
#             content_overlap_score * 0.25 +  # 25% weight
#             title_score * 0.2 +             # 20% weight
#             question_score * 0.1 +          # 10% weight
#             keyword_score * 0.05            # 5% weight
#         )
        
#         return total_score

# # Initialize global RAG instance
# rag_system = DualLanguageRAG()

# def query_rag_multilang(user_question: str) -> dict:
#     """Main function to query the dual language RAG system"""
#     return rag_system.query_rag_dual_language(user_question)

# def run_multilang_rag_pipeline():
#     """Interactive CLI"""
#     print("🧠 Budaya Bali RAG Dual Language System")
#     print("Sistem ini akan:")
#     print("- Menggunakan dataset Indonesia untuk pertanyaan berbahasa Indonesia")
#     print("- Menggunakan dataset English untuk pertanyaan berbahasa lain")
#     print("Type 'exit' atau 'quit' untuk keluar.\n")

#     while True:
#         user_input = input("Pertanyaan Anda / Your question:\n> ").strip()

#         if user_input.lower() in ["exit", "quit", "keluar"]:
#             print("Goodbye! / Selamat tinggal!")
#             break

#         try:
#             result = query_rag_multilang(user_input)

#             print(f"\n📌 Answer ({result['dataset_used']} dataset):\n{result['answer']}")
            
#             if result['sources']:
#                 print(f"\n📚 Sources Used:")
#                 for i, source in enumerate(result['sources'], 1):
#                     print(f"{i}. {source['title']}")
#                     print(f"   {source['url']}")
#                     if source.get('cosine_similarity') is not None:
#                         print(f"   🔍 Similarity Score: {source['cosine_similarity']:.3f}")
#                     if source['image']:
#                         print(f"   📷 Image: {source['image']}")
#             else:
#                 print("\n⚠ No sources included.")

#             print("-" * 60)

#         except Exception as e:
#             print(f"❌ Error: {e}")
#             print("Please try again.")

# if __name__ == "__main__":
#     run_multilang_rag_pipeline()


# # KODE JIKA USER BERTANYA DENGAN BAHASA INDOENSIA DAN JAWABAN TIDAK DITEMUKAN DI DOKUMEN BAHASA INDONESIA AKAN DICARI PADA DOKUMEN BAHASA INGGRIS SEBELUM TTS
# import os
# from dotenv import load_dotenv
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI 
# from translate_func import translate_to_english, translate_from_english, detect_language, is_indonesian, translate_to_target_language
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# load_dotenv()

# class DualLanguageRAG:
#     def __init__(self):
#         # Initialize embedding model
#         self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
#         # Load both FAISS indexes
#         try:
#             self.vectordb_indonesia = FAISS.load_local(
#                 "faiss_index_indonesia", 
#                 self.embedding, 
#                 allow_dangerous_deserialization=True
#             )
#             self.retriever_indonesia = self.vectordb_indonesia.as_retriever(
#                 search_type="similarity", 
#                 search_kwargs={"k": 4}
#             )
#             print("✅ Indonesian FAISS index loaded successfully")
#         except Exception as e:
#             print(f"❌ Error loading Indonesian index: {e}")
#             self.vectordb_indonesia = None
#             self.retriever_indonesia = None
        
#         try:
#             self.vectordb_english = FAISS.load_local(
#                 "faiss_index_english", 
#                 self.embedding, 
#                 allow_dangerous_deserialization=True
#             )
#             self.retriever_english = self.vectordb_english.as_retriever(
#                 search_type="similarity", 
#                 search_kwargs={"k": 4}
#             )
#             print("✅ English FAISS index loaded successfully")
#         except Exception as e:
#             print(f"❌ Error loading English index: {e}")
#             self.vectordb_english = None
#             self.retriever_english = None
        
#         # Initialize LLM
#         self.llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)
        
#         # Prompt templates for different languages
#         self.template_indonesian = """
# Anda adalah asisten yang membantu. Silakan jawab pertanyaan berdasarkan HANYA pada konteks yang diberikan.
# Jika jawaban tidak ada dalam konteks, jangan membuat jawaban. Berikan jawaban yang ringkas dan sopan dalam bahasa Indonesia.

# Setelah jawaban, tambahkan tag:
# - Jika Anda menggunakan konteks, akhiri dengan [SOURCES USED]
# - Jika konteks tidak berguna, akhiri dengan [NO SOURCE USED]

# Konteks:
# {context}

# Pertanyaan:
# {question}

# Jawaban:
# """
        
#         self.template_english = """
# You are a helpful assistant. Please answer the question based ONLY on the provided context.
# If the answer is not in the context, don't make up an answer. Be concise and polite.

# After the answer, add a tag:
# - If you used the context, end with [SOURCES USED]
# - If the context was not useful, end with [NO SOURCE USED]

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
        
#         # Create prompt templates
#         self.prompt_indonesian = PromptTemplate(
#             template=self.template_indonesian, 
#             input_variables=["context", "question"]
#         )
#         self.prompt_english = PromptTemplate(
#             template=self.template_english, 
#             input_variables=["context", "question"]
#         )
        
#         # Create QA chains
#         if self.retriever_indonesia:
#             self.qa_chain_indonesia = RetrievalQA.from_chain_type(
#                 llm=self.llm,
#                 retriever=self.retriever_indonesia,
#                 chain_type="stuff",
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": self.prompt_indonesian},
#             )
        
#         if self.retriever_english:
#             self.qa_chain_english = RetrievalQA.from_chain_type(
#                 llm=self.llm,
#                 retriever=self.retriever_english,
#                 chain_type="stuff",
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": self.prompt_english},
#             )

#     def _calculate_cosine_similarity(self, text1, text2):
#         """Calculate cosine similarity between two texts using the embedding model"""
#         # Get embeddings for both texts
#         emb1 = self.embedding.embed_query(text1)
#         emb2 = self.embedding.embed_query(text2)
        
#         # Reshape for cosine similarity calculation
#         emb1 = np.array(emb1).reshape(1, -1)
#         emb2 = np.array(emb2).reshape(1, -1)
        
#         # Calculate and return cosine similarity
#         return cosine_similarity(emb1, emb2)[0][0]

#     def query_rag_dual_language(self, user_question: str) -> dict:
#         """Enhanced RAG function with dual language support with fallback mechanism"""
    
#         # Detect input language
#         input_lang = detect_language(user_question)
#         print(f"\n🌐 Detected Language: {input_lang}")
    
#         # Determine which dataset to try first
#         use_indonesian_dataset = is_indonesian(user_question)
#         print(f"📚 Primary dataset: {'Indonesian' if use_indonesian_dataset else 'English'}")
    
#         result = None
#         dataset_used = "none"
#         question_en = None
    
#         # Try with Indonesian dataset first if applicable
#         if use_indonesian_dataset and self.retriever_indonesia:
#             print("🔍 Searching in Indonesian dataset...")
#             result = self.qa_chain_indonesia.invoke(user_question)
#             full_answer = result['result'].strip()
        
#             # Check if answer indicates no source was found
#             if "[NO SOURCE USED]" in full_answer:
#                 print("⚠ No useful results in Indonesian dataset, trying English...")
#                 result = None
#             else:
#                 dataset_used = "indonesian"
    
#         # If no results from Indonesian or question is in other language
#         if result is None and self.retriever_english:
#             print("🔍 Searching in English dataset...")
        
#             # Translate question to English if needed
#             if input_lang != 'en':
#                 question_en = translate_to_english(user_question, input_lang)
#                 print(f"🔄 Translated question: {question_en}")
#             else:
#                 question_en = user_question
            
#             result = self.qa_chain_english.invoke(question_en)
#             full_answer_en = result['result'].strip()
#             dataset_used = "english"
        
#             # Translate answer back to user's language if needed
#             if input_lang != 'en':
#                 # Extract clean answer without tags
#                 if "[SOURCES USED]" in full_answer_en:
#                     clean_answer_en = full_answer_en.replace("[SOURCES USED]", "").strip()
#                     tag = "[SOURCES USED]"
#                 elif "[NO SOURCE USED]" in full_answer_en:
#                     clean_answer_en = full_answer_en.replace("[NO SOURCE USED]", "").strip()
#                     tag = "[NO SOURCE USED]"
#                 else:
#                     clean_answer_en = full_answer_en
#                     tag = ""
            
#                 # Translate clean answer
#                 translated_answer = translate_from_english(clean_answer_en, input_lang)
#                 full_answer = translated_answer + (" " + tag if tag else "")
#                 print(f"🔄 Translated answer: {translated_answer}")
#             else:
#                 full_answer = full_answer_en
    
#         # Fallback if no results found in either dataset
#         if result is None:
#             error_msg = "Maaf, informasi tidak ditemukan dalam dataset kami." if use_indonesian_dataset else "Sorry, the information was not found in our dataset."
#             return {
#                 "input_language": input_lang,
#                 "question": user_question,
#                 "answer": error_msg,
#                 "sources": [],
#                 "dataset_used": "none"
#             }
    
#         # Check if model used sources
#         if "[SOURCES USED]" in full_answer:
#             use_sources = True
#             clean_answer = full_answer.replace("[SOURCES USED]", "").strip()
#         elif "[NO SOURCE USED]" in full_answer:
#             use_sources = False
#             clean_answer = full_answer.replace("[NO SOURCE USED]", "").strip()
#         else:
#             use_sources = False
#             clean_answer = full_answer
    
#         # Process source documents
#         sources = []
#         if use_sources and result.get("source_documents"):
#             print("\n=== DEBUGGING: SOURCE DOCUMENTS ===")
#             for i, doc in enumerate(result["source_documents"], 1):
#                 print(f"[{i}] TITLE:", doc.metadata.get("title"))
#                 print("     URL  :", doc.metadata.get("url"))
#                 print("     IMAGE:", doc.metadata.get("image"))
#                 print("     SAMPLE:", doc.page_content[:80].replace("\n", " "), "...\n")
        
#             # Filter and prepare sources
#             search_query = user_question if dataset_used == "indonesian" else question_en if question_en else user_question
#             sources = self._filter_relevant_sources(
#                 result["source_documents"], 
#                 clean_answer, 
#                 search_query
#             )
    
#         return {
#             "input_language": input_lang,
#             "question": user_question,
#             "answer": clean_answer,
#             "sources": sources,
#             "dataset_used": dataset_used
#         }
    
#     def _filter_relevant_sources(self, source_documents, answer, question):
#         """Filter and rank sources that are actually relevant to the answer"""
#         answer_lower = answer.lower()
#         question_lower = question.lower()
#         question_keywords = set(question_lower.split())
        
#         # Extract key phrases from answer (sentences or important phrases)
#         answer_phrases = []
#         sentences = answer.split('.')
#         for sentence in sentences:
#             if len(sentence.strip()) > 10:  # Skip very short sentences
#                 answer_phrases.append(sentence.strip().lower())
        
#         url_to_doc = {}
        
#         # Get best version of each document per URL
#         for doc in source_documents:
#             url = doc.metadata.get("url", "No URL")
#             image = doc.metadata.get("image", "")
#             if url not in url_to_doc or (not url_to_doc[url].metadata.get("image") and image):
#                 url_to_doc[url] = doc
        
#         # Calculate relevance scores for each document
#         document_scores = []
        
#         for url, doc in url_to_doc.items():
#             snippet = doc.page_content.strip().lower()
#             title = doc.metadata.get("title", "").lower()
#             image = doc.metadata.get("image", "")
            
#             # Calculate different relevance scores
#             relevance_score = self._calculate_document_relevance(
#                 doc, answer_lower, answer_phrases, question_lower, question_keywords
#             )
            
#             # Calculate cosine similarity between question and document content
#             cosine_sim = self._calculate_cosine_similarity(question, doc.page_content)
            
#             # Combine scores (you can adjust weights as needed)
#             combined_score = (relevance_score * 0.7) + (cosine_sim * 0.3)
            
#             # Only include documents with minimum relevance
#             if combined_score > 0.1:  # Threshold for relevance
#                 document_scores.append({
#                     "doc": doc,
#                     "score": combined_score,
#                     "relevance_score": relevance_score,
#                     "cosine_similarity": cosine_sim,
#                     "title": doc.metadata.get("title", "No title"),
#                     "url": url,
#                     "image": image
#                 })
        
#         # Sort by combined score (highest first)
#         document_scores.sort(key=lambda x: x['score'], reverse=True)
        
#         # Convert to final source format
#         sources = []
#         for doc_info in document_scores:
#             sources.append({
#                 "title": doc_info["title"],
#                 "url": doc_info["url"],
#                 "image": doc_info["image"],
#                 "relevance_score": doc_info["relevance_score"],
#                 "cosine_similarity": doc_info["cosine_similarity"]
#             })
            
#         print(f"\n=== SOURCE RANKING DEBUG ===")
#         for i, doc_info in enumerate(document_scores[:3], 1):
#             print(f"[{i}] SCORE: {doc_info['score']:.3f} (Relevance: {doc_info['relevance_score']:.3f}, Cosine: {doc_info['cosine_similarity']:.3f})")
#             print(f"    TITLE: {doc_info['title']}")
#             print(f"    URL: {doc_info['url']}\n")
        
#         return sources
    
#     def _calculate_document_relevance(self, doc, answer_lower, answer_phrases, question_lower, question_keywords):
#         """Calculate how relevant a document is to the answer and question"""
#         snippet = doc.page_content.strip().lower()
#         title = doc.metadata.get("title", "").lower()
        
#         total_score = 0.0
        
#         # 1. Direct content overlap with answer (highest weight)
#         content_overlap_score = 0.0
#         for phrase in answer_phrases:
#             if len(phrase) > 5:  # Only consider meaningful phrases
#                 # Check for exact phrase matches
#                 if phrase in snippet:
#                     content_overlap_score += 3.0
#                 # Check for partial matches (word overlap)
#                 phrase_words = set(phrase.split())
#                 snippet_words = set(snippet.split())
#                 overlap_ratio = len(phrase_words & snippet_words) / len(phrase_words) if phrase_words else 0
#                 if overlap_ratio > 0.5:  # More than 50% word overlap
#                     content_overlap_score += overlap_ratio * 2.0
        
#         # 2. Title relevance to question (high weight)
#         title_score = 0.0
#         title_words = set(title.split())
#         title_question_overlap = len(question_keywords & title_words) / len(question_keywords) if question_keywords else 0
#         title_score = title_question_overlap * 2.5
        
#         # 3. Content relevance to question (medium weight)
#         question_score = 0.0
#         snippet_words = set(snippet.split())
#         question_content_overlap = len(question_keywords & snippet_words) / len(question_keywords) if question_keywords else 0
#         question_score = question_content_overlap * 1.5
        
#         # 4. Specific keyword matching (medium weight)
#         keyword_score = 0.0
#         for keyword in question_keywords:
#             if len(keyword) > 2:  # Skip very short words
#                 if keyword in title:
#                     keyword_score += 1.0  # Title match is more important
#                 elif keyword in snippet:
#                     keyword_score += 0.5
        
#         # 5. Answer-specific content matching (highest weight)
#         answer_content_score = 0.0
#         # Check if document content appears in the answer
#         doc_sentences = snippet.split('.')
#         for sentence in doc_sentences:
#             if len(sentence.strip()) > 10:
#                 sentence_clean = sentence.strip().lower()
#                 if sentence_clean in answer_lower:
#                     answer_content_score += 4.0  # Very high score for direct usage
#                 else:
#                     # Check for word overlap with answer
#                     sentence_words = set(sentence_clean.split())
#                     answer_words = set(answer_lower.split())
#                     overlap = len(sentence_words & answer_words)
#                     if overlap > 3:  # Significant word overlap
#                         answer_content_score += (overlap / len(sentence_words)) * 2.0
        
#         # Combine all scores
#         total_score = (
#             answer_content_score * 0.4 +    # 40% weight - most important
#             content_overlap_score * 0.25 +  # 25% weight
#             title_score * 0.2 +             # 20% weight
#             question_score * 0.1 +          # 10% weight
#             keyword_score * 0.05            # 5% weight
#         )
        
#         return total_score

# # Initialize global RAG instance
# rag_system = DualLanguageRAG()

# def query_rag_multilang(user_question: str) -> dict:
#     """Main function to query the dual language RAG system"""
#     return rag_system.query_rag_dual_language(user_question)

# def run_multilang_rag_pipeline():
#     """Interactive CLI"""
#     print("🧠 Budaya Bali RAG Dual Language System")
#     print("Sistem ini akan:")
#     print("- Menggunakan dataset Indonesia untuk pertanyaan berbahasa Indonesia")
#     print("- Menggunakan dataset English untuk pertanyaan berbahasa lain")
#     print("Type 'exit' atau 'quit' untuk keluar.\n")

#     while True:
#         user_input = input("Pertanyaan Anda / Your question:\n> ").strip()

#         if user_input.lower() in ["exit", "quit", "keluar"]:
#             print("Goodbye! / Selamat tinggal!")
#             break

#         try:
#             result = query_rag_multilang(user_input)

#             print(f"\n📌 Answer ({result['dataset_used']} dataset):\n{result['answer']}")
            
#             if result['sources']:
#                 print(f"\n📚 Sources Used:")
#                 for i, source in enumerate(result['sources'], 1):
#                     print(f"{i}. {source['title']}")
#                     print(f"   {source['url']}")
#                     if source.get('cosine_similarity') is not None:
#                         print(f"   🔍 Similarity Score: {source['cosine_similarity']:.3f}")
#                     if source['image']:
#                         print(f"   📷 Image: {source['image']}")
#             else:
#                 print("\n⚠ No sources included.")

#             print("-" * 60)

#         except Exception as e:
#             print(f"❌ Error: {e}")
#             print("Please try again.")

# if __name__ == "__main__":
#     run_multilang_rag_pipeline()


# # KODE JIKA USER BERTANYA DENGAN BAHASA INDOENSIA DAN JAWABAN TIDAK DITEMUKAN DI DOKUMEN BAHASA INDONESIA AKAN DICARI PADA DOKUMEN BAHASA INGGRIS DAN AKAN DITAMPILKAN 1 SUMBER SAJA
# import os
# from dotenv import load_dotenv
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI 
# from translate_func import translate_to_english, translate_from_english, detect_language, is_indonesian, translate_to_target_language
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# load_dotenv()

# class DualLanguageRAG:
#     def __init__(self):
#         # Initialize embedding model
#         self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
#         # Load both FAISS indexes
#         try:
#             self.vectordb_indonesia = FAISS.load_local(
#                 "faiss_index_indonesia", 
#                 self.embedding, 
#                 allow_dangerous_deserialization=True
#             )
#             self.retriever_indonesia = self.vectordb_indonesia.as_retriever(
#                 search_type="similarity", 
#                 search_kwargs={"k": 4}
#             )
#             print("✅ Indonesian FAISS index loaded successfully")
#         except Exception as e:
#             print(f"❌ Error loading Indonesian index: {e}")
#             self.vectordb_indonesia = None
#             self.retriever_indonesia = None
        
#         try:
#             self.vectordb_english = FAISS.load_local(
#                 "faiss_index_english", 
#                 self.embedding, 
#                 allow_dangerous_deserialization=True
#             )
#             self.retriever_english = self.vectordb_english.as_retriever(
#                 search_type="similarity", 
#                 search_kwargs={"k": 4}
#             )
#             print("✅ English FAISS index loaded successfully")
#         except Exception as e:
#             print(f"❌ Error loading English index: {e}")
#             self.vectordb_english = None
#             self.retriever_english = None
        
#         # Initialize LLM
#         self.llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)
        
#         # Prompt templates for different languages
#         self.template_indonesian = """
# Anda adalah asisten yang membantu. Silakan jawab pertanyaan berdasarkan HANYA pada konteks yang diberikan.
# Jika jawaban tidak ada dalam konteks, jangan membuat jawaban. Berikan jawaban yang ringkas dan sopan dalam bahasa Indonesia.

# Setelah jawaban, tambahkan tag:
# - Jika Anda menggunakan konteks, akhiri dengan [SOURCES USED]
# - Jika konteks tidak berguna, akhiri dengan [NO SOURCE USED]

# Konteks:
# {context}

# Pertanyaan:
# {question}

# Jawaban:
# """
        
#         self.template_english = """
# You are a helpful assistant. Please answer the question based ONLY on the provided context.
# If the answer is not in the context, don't make up an answer. Be concise and polite.

# After the answer, add a tag:
# - If you used the context, end with [SOURCES USED]
# - If the context was not useful, end with [NO SOURCE USED]

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
        
#         # Create prompt templates
#         self.prompt_indonesian = PromptTemplate(
#             template=self.template_indonesian, 
#             input_variables=["context", "question"]
#         )
#         self.prompt_english = PromptTemplate(
#             template=self.template_english, 
#             input_variables=["context", "question"]
#         )
        
#         # Create QA chains
#         if self.retriever_indonesia:
#             self.qa_chain_indonesia = RetrievalQA.from_chain_type(
#                 llm=self.llm,
#                 retriever=self.retriever_indonesia,
#                 chain_type="stuff",
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": self.prompt_indonesian},
#             )
        
#         if self.retriever_english:
#             self.qa_chain_english = RetrievalQA.from_chain_type(
#                 llm=self.llm,
#                 retriever=self.retriever_english,
#                 chain_type="stuff",
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": self.prompt_english},
#             )

#     def _calculate_cosine_similarity(self, text1, text2):
#         """Calculate cosine similarity between two texts using the embedding model"""
#         # Get embeddings for both texts
#         emb1 = self.embedding.embed_query(text1)
#         emb2 = self.embedding.embed_query(text2)
        
#         # Reshape for cosine similarity calculation
#         emb1 = np.array(emb1).reshape(1, -1)
#         emb2 = np.array(emb2).reshape(1, -1)
        
#         # Calculate and return cosine similarity
#         return cosine_similarity(emb1, emb2)[0][0]

#     def query_rag_dual_language(self, user_question: str) -> dict:
#         """Enhanced RAG function with dual language support with fallback mechanism"""
    
#         # Detect input language
#         input_lang = detect_language(user_question)
#         print(f"\n🌐 Detected Language: {input_lang}")
    
#         # Determine which dataset to try first
#         use_indonesian_dataset = is_indonesian(user_question)
#         print(f"📚 Primary dataset: {'Indonesian' if use_indonesian_dataset else 'English'}")
    
#         result = None
#         dataset_used = "none"
#         question_en = None
    
#         # Try with Indonesian dataset first if applicable
#         if use_indonesian_dataset and self.retriever_indonesia:
#             print("🔍 Searching in Indonesian dataset...")
#             result = self.qa_chain_indonesia.invoke(user_question)
#             full_answer = result['result'].strip()
        
#             # Check if answer indicates no source was found
#             if "[NO SOURCE USED]" in full_answer:
#                 print("⚠ No useful results in Indonesian dataset, trying English...")
#                 result = None
#             else:
#                 dataset_used = "indonesian"
    
#         # If no results from Indonesian or question is in other language
#         if result is None and self.retriever_english:
#             print("🔍 Searching in English dataset...")
        
#             # Translate question to English if needed
#             if input_lang != 'en':
#                 question_en = translate_to_english(user_question, input_lang)
#                 print(f"🔄 Translated question: {question_en}")
#             else:
#                 question_en = user_question
            
#             result = self.qa_chain_english.invoke(question_en)
#             full_answer_en = result['result'].strip()
#             dataset_used = "english"
        
#             # Translate answer back to user's language if needed
#             if input_lang != 'en':
#                 # Extract clean answer without tags
#                 if "[SOURCES USED]" in full_answer_en:
#                     clean_answer_en = full_answer_en.replace("[SOURCES USED]", "").strip()
#                     tag = "[SOURCES USED]"
#                 elif "[NO SOURCE USED]" in full_answer_en:
#                     clean_answer_en = full_answer_en.replace("[NO SOURCE USED]", "").strip()
#                     tag = "[NO SOURCE USED]"
#                 else:
#                     clean_answer_en = full_answer_en
#                     tag = ""
            
#                 # Translate clean answer
#                 translated_answer = translate_from_english(clean_answer_en, input_lang)
#                 full_answer = translated_answer + (" " + tag if tag else "")
#                 print(f"🔄 Translated answer: {translated_answer}")
#             else:
#                 full_answer = full_answer_en
    
#         # Fallback if no results found in either dataset
#         if result is None:
#             error_msg = "Maaf, informasi tidak ditemukan dalam dataset kami." if use_indonesian_dataset else "Sorry, the information was not found in our dataset."
#             return {
#                 "input_language": input_lang,
#                 "question": user_question,
#                 "answer": error_msg,
#                 "sources": [],
#                 "dataset_used": "none"
#             }
    
#         # Check if model used sources
#         if "[SOURCES USED]" in full_answer:
#             use_sources = True
#             clean_answer = full_answer.replace("[SOURCES USED]", "").strip()
#         elif "[NO SOURCE USED]" in full_answer:
#             use_sources = False
#             clean_answer = full_answer.replace("[NO SOURCE USED]", "").strip()
#         else:
#             use_sources = False
#             clean_answer = full_answer
    
#         # Process source documents
#         sources = []
#         if use_sources and result.get("source_documents"):
#             print("\n=== DEBUGGING: SOURCE DOCUMENTS ===")
#             for i, doc in enumerate(result["source_documents"], 1):
#                 print(f"[{i}] TITLE:", doc.metadata.get("title"))
#                 print("     URL  :", doc.metadata.get("url"))
#                 print("     IMAGE:", doc.metadata.get("image"))
#                 print("     SAMPLE:", doc.page_content[:80].replace("\n", " "), "...\n")
        
#             # Filter and prepare sources
#             search_query = user_question if dataset_used == "indonesian" else question_en if question_en else user_question
#             sources = self._filter_relevant_sources(
#                 result["source_documents"], 
#                 clean_answer, 
#                 search_query
#             )
    
#         return {
#             "input_language": input_lang,
#             "question": user_question,
#             "answer": clean_answer,
#             "sources": sources,
#             "dataset_used": dataset_used
#         }
    
#     def _filter_relevant_sources(self, source_documents, answer, question):
#         """Filter and rank sources that are actually relevant to the answer"""
#         answer_lower = answer.lower()
#         question_lower = question.lower()
#         question_keywords = set(question_lower.split())
        
#         # Extract key phrases from answer (sentences or important phrases)
#         answer_phrases = []
#         sentences = answer.split('.')
#         for sentence in sentences:
#             if len(sentence.strip()) > 10:  # Skip very short sentences
#                 answer_phrases.append(sentence.strip().lower())
        
#         url_to_doc = {}
        
#         # Get best version of each document per URL
#         for doc in source_documents:
#             url = doc.metadata.get("url", "No URL")
#             image = doc.metadata.get("image", "")
#             if url not in url_to_doc or (not url_to_doc[url].metadata.get("image") and image):
#                 url_to_doc[url] = doc
        
#         # Calculate relevance scores for each document
#         document_scores = []
        
#         for url, doc in url_to_doc.items():
#             snippet = doc.page_content.strip().lower()
#             title = doc.metadata.get("title", "").lower()
#             image = doc.metadata.get("image", "")
            
#             # Calculate different relevance scores
#             relevance_score = self._calculate_document_relevance(
#                 doc, answer_lower, answer_phrases, question_lower, question_keywords
#             )
            
#             # Calculate cosine similarity between question and document content
#             cosine_sim = self._calculate_cosine_similarity(question, doc.page_content)
            
#             # Combine scores (you can adjust weights as needed)
#             combined_score = (relevance_score * 0.7) + (cosine_sim * 0.3)
            
#             # Only include documents with minimum relevance
#             if combined_score > 0.1:  # Threshold for relevance
#                 document_scores.append({
#                     "doc": doc,
#                     "score": combined_score,
#                     "relevance_score": relevance_score,
#                     "cosine_similarity": cosine_sim,
#                     "title": doc.metadata.get("title", "No title"),
#                     "url": url,
#                     "image": image
#                 })
        
#         # Sort by combined score (highest first)
#         document_scores.sort(key=lambda x: x['score'], reverse=True)
        
#         # Convert to final source format
#         sources = []
#         if document_scores:
#             doc_info = document_scores[0]
#             sources.append({
#                 "title": doc_info["title"],
#                 "url": doc_info["url"],
#                 "image": doc_info["image"],
#                 "relevance_score": doc_info["relevance_score"],
#                 "cosine_similarity": doc_info["cosine_similarity"]
#             })
        
#         print(f"\n=== SOURCE RANKING DEBUG ===")
#         for i, doc_info in enumerate(document_scores[:3], 1):
#             print(f"[{i}] SCORE: {doc_info['score']:.3f} (Relevance: {doc_info['relevance_score']:.3f}, Cosine: {doc_info['cosine_similarity']:.3f})")
#             print(f"    TITLE: {doc_info['title']}")
#             print(f"    URL: {doc_info['url']}\n")
        
#         return sources
    
#     def _calculate_document_relevance(self, doc, answer_lower, answer_phrases, question_lower, question_keywords):
#         """Calculate how relevant a document is to the answer and question"""
#         snippet = doc.page_content.strip().lower()
#         title = doc.metadata.get("title", "").lower()
        
#         total_score = 0.0
        
#         # 1. Direct content overlap with answer (highest weight)
#         content_overlap_score = 0.0
#         for phrase in answer_phrases:
#             if len(phrase) > 5:  # Only consider meaningful phrases
#                 # Check for exact phrase matches
#                 if phrase in snippet:
#                     content_overlap_score += 3.0
#                 # Check for partial matches (word overlap)
#                 phrase_words = set(phrase.split())
#                 snippet_words = set(snippet.split())
#                 overlap_ratio = len(phrase_words & snippet_words) / len(phrase_words) if phrase_words else 0
#                 if overlap_ratio > 0.5:  # More than 50% word overlap
#                     content_overlap_score += overlap_ratio * 2.0
        
#         # 2. Title relevance to question (high weight)
#         title_score = 0.0
#         title_words = set(title.split())
#         title_question_overlap = len(question_keywords & title_words) / len(question_keywords) if question_keywords else 0
#         title_score = title_question_overlap * 2.5
        
#         # 3. Content relevance to question (medium weight)
#         question_score = 0.0
#         snippet_words = set(snippet.split())
#         question_content_overlap = len(question_keywords & snippet_words) / len(question_keywords) if question_keywords else 0
#         question_score = question_content_overlap * 1.5
        
#         # 4. Specific keyword matching (medium weight)
#         keyword_score = 0.0
#         for keyword in question_keywords:
#             if len(keyword) > 2:  # Skip very short words
#                 if keyword in title:
#                     keyword_score += 1.0  # Title match is more important
#                 elif keyword in snippet:
#                     keyword_score += 0.5
        
#         # 5. Answer-specific content matching (highest weight)
#         answer_content_score = 0.0
#         # Check if document content appears in the answer
#         doc_sentences = snippet.split('.')
#         for sentence in doc_sentences:
#             if len(sentence.strip()) > 10:
#                 sentence_clean = sentence.strip().lower()
#                 if sentence_clean in answer_lower:
#                     answer_content_score += 4.0  # Very high score for direct usage
#                 else:
#                     # Check for word overlap with answer
#                     sentence_words = set(sentence_clean.split())
#                     answer_words = set(answer_lower.split())
#                     overlap = len(sentence_words & answer_words)
#                     if overlap > 3:  # Significant word overlap
#                         answer_content_score += (overlap / len(sentence_words)) * 2.0
        
#         # Combine all scores
#         total_score = (
#             answer_content_score * 0.4 +    # 40% weight - most important
#             content_overlap_score * 0.25 +  # 25% weight
#             title_score * 0.2 +             # 20% weight
#             question_score * 0.1 +          # 10% weight
#             keyword_score * 0.05            # 5% weight
#         )
        
#         return total_score

# # Initialize global RAG instance
# rag_system = DualLanguageRAG()

# def query_rag_multilang(user_question: str) -> dict:
#     """Main function to query the dual language RAG system"""
#     return rag_system.query_rag_dual_language(user_question)

# def run_multilang_rag_pipeline():
#     """Interactive CLI"""
#     print("🧠 Budaya Bali RAG Dual Language System")
#     print("Sistem ini akan:")
#     print("- Menggunakan dataset Indonesia untuk pertanyaan berbahasa Indonesia")
#     print("- Menggunakan dataset English untuk pertanyaan berbahasa lain")
#     print("Type 'exit' atau 'quit' untuk keluar.\n")

#     while True:
#         user_input = input("Pertanyaan Anda / Your question:\n> ").strip()

#         if user_input.lower() in ["exit", "quit", "keluar"]:
#             print("Goodbye! / Selamat tinggal!")
#             break

#         try:
#             result = query_rag_multilang(user_input)

#             print(f"\n📌 Answer ({result['dataset_used']} dataset):\n{result['answer']}")
            
#             if result['sources']:
#                 print(f"\n📚 Sources Used:")
#                 for i, source in enumerate(result['sources'], 1):
#                     print(f"{i}. {source['title']}")
#                     print(f"   {source['url']}")
#                     if source.get('cosine_similarity') is not None:
#                         print(f"   🔍 Similarity Score: {source['cosine_similarity']:.3f}")
#                     if source['image']:
#                         print(f"   📷 Image: {source['image']}")
#             else:
#                 print("\n⚠ No sources included.")

#             print("-" * 60)

#         except Exception as e:
#             print(f"❌ Error: {e}")
#             print("Please try again.")

# if __name__ == "__main__":
#     run_multilang_rag_pipeline()


# KODE JIKA USER BERTANYA DENGAN BAHASA INDOENSIA DAN JAWABAN TIDAK DITEMUKAN DI DOKUMEN BAHASA INDONESIA AKAN DICARI PADA DOKUMEN BAHASA INGGRIS 
# DAN AKAN DITAMPILKAN 1 SUMBER DAN BEBERAPA SUMBER SESUAI PERTANYAAN

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from translate_func import (
    translate_to_english,
    translate_from_english,
    detect_language,
    is_indonesian,
    translate_to_target_language,
)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

class DualLanguageRAG:
    def __init__(self):
        # Initialize embedding model
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load both FAISS indexes
        try:
            self.vectordb_indonesia = FAISS.load_local(
                "faiss_index_indonesia", 
                self.embedding, 
                allow_dangerous_deserialization=True
            )
            self.retriever_indonesia = self.vectordb_indonesia.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 4}
            )
            print("✅ Indonesian FAISS index loaded successfully")
        except Exception as e:
            print(f"❌ Error loading Indonesian index: {e}")
            self.vectordb_indonesia = None
            self.retriever_indonesia = None
        
        try:
            self.vectordb_english = FAISS.load_local(
                "faiss_index_english", 
                self.embedding, 
                allow_dangerous_deserialization=True
            )
            self.retriever_english = self.vectordb_english.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 4}
            )
            print("✅ English FAISS index loaded successfully")
        except Exception as e:
            print(f"❌ Error loading English index: {e}")
            self.vectordb_english = None
            self.retriever_english = None
        
        # Initialize LLM
        self.llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)
        
        # Prompt templates for different languages
        self.template_indonesian = """
Anda adalah asisten yang membantu. Silakan jawab pertanyaan berdasarkan HANYA pada konteks yang diberikan.
Jika jawaban tidak ada dalam konteks, jangan membuat jawaban. Berikan jawaban yang ringkas dan sopan dalam bahasa Indonesia.

Setelah jawaban, tambahkan tag:
- Jika Anda menggunakan konteks, akhiri dengan [SOURCES USED]
- Jika konteks tidak berguna, akhiri dengan [NO SOURCE USED]

Konteks:
{context}

Pertanyaan:
{question}

Jawaban:
"""
        
        self.template_english = """
You are a helpful assistant. Please answer the question based ONLY on the provided context.
If the answer is not in the context, don't make up an answer. Be concise and polite.

After the answer, add a tag:
- If you used the context, end with [SOURCES USED]
- If the context was not useful, end with [NO SOURCE USED]

Context:
{context}

Question:
{question}

Answer:
"""
        
        # Create prompt templates
        self.prompt_indonesian = PromptTemplate(
            template=self.template_indonesian, 
            input_variables=["context", "question"]
        )
        self.prompt_english = PromptTemplate(
            template=self.template_english, 
            input_variables=["context", "question"]
        )
        
        # Create QA chains
        if self.retriever_indonesia:
            self.qa_chain_indonesia = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever_indonesia,
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt_indonesian},
            )
        
        if self.retriever_english:
            self.qa_chain_english = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever_english,
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt_english},
            )

    def _calculate_cosine_similarity(self, text1, text2):
        """Calculate cosine similarity between two texts using the embedding model"""
        emb1 = self.embedding.embed_query(text1)
        emb2 = self.embedding.embed_query(text2)
        emb1 = np.array(emb1).reshape(1, -1)
        emb2 = np.array(emb2).reshape(1, -1)
        return cosine_similarity(emb1, emb2)[0][0]

    def _is_list_question(self, question: str) -> bool:
        """Deteksi apakah pertanyaan meminta daftar/banyak item."""
        keywords = [
            "sebutkan", "daftar", "apa saja", "siapa saja", "contoh", "jenis-jenis",
            "jelaskan beberapa", "berikan beberapa", "tuliskan beberapa", "list", "perbedaan", "rekomendasi",
            "mention", "list", "any", "anyone", "example", "types","explain some", "give some",
            "write some", "list", "differences", "recommendation"
        ]
        q_lower = question.lower()
        return any(kw in q_lower for kw in keywords)

    def _filter_relevant_sources(self, source_documents, answer, question):
        """Filter and rank sources, tampilkan 1 atau beberapa tergantung intent pertanyaan."""
        answer_lower = answer.lower()
        question_lower = question.lower()
        question_keywords = set(question_lower.split())
        
        # Extract key phrases from answer (sentences or important phrases)
        answer_phrases = []
        sentences = answer.split('.')
        for sentence in sentences:
            if len(sentence.strip()) > 10:  # Skip very short sentences
                answer_phrases.append(sentence.strip().lower())
        
        url_to_doc = {}
        
        # Get best version of each document per URL
        for doc in source_documents:
            url = doc.metadata.get("url", "No URL")
            image = doc.metadata.get("image", "")
            if url not in url_to_doc or (not url_to_doc[url].metadata.get("image") and image):
                url_to_doc[url] = doc
        
        # Calculate relevance scores for each document
        document_scores = []
        
        for url, doc in url_to_doc.items():
            snippet = doc.page_content.strip().lower()
            title = doc.metadata.get("title", "").lower()
            image = doc.metadata.get("image", "")
            
            # Calculate different relevance scores
            relevance_score = self._calculate_document_relevance(
                doc, answer_lower, answer_phrases, question_lower, question_keywords
            )
            
            # Calculate cosine similarity between question and document content
            cosine_sim = self._calculate_cosine_similarity(question, doc.page_content)
            
            # Combine scores (you can adjust weights as needed)
            combined_score = (relevance_score * 0.7) + (cosine_sim * 0.3)
            
            # Only include documents with minimum relevance
            if combined_score > 0.1:  # Threshold for relevance
                document_scores.append({
                    "doc": doc,
                    "score": combined_score,
                    "relevance_score": relevance_score,
                    "cosine_similarity": cosine_sim,
                    "title": doc.metadata.get("title", "No title"),
                    "url": url,
                    "image": image
                })
        
        # Sort by combined score (highest first)
        document_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # === MODIFIKASI: Pilih jumlah sumber berdasarkan intent pertanyaan ===
        sources = []
        if document_scores:
            if self._is_list_question(question):
                # Tampilkan beberapa dokumen (misal: top 3)
                for doc_info in document_scores[:3]:
                    sources.append({
                        "title": doc_info["title"],
                        "url": doc_info["url"],
                        "image": doc_info["image"],
                        "relevance_score": doc_info["relevance_score"],
                        "cosine_similarity": doc_info["cosine_similarity"]
                    })
            else:
                # Tampilkan hanya 1 dokumen paling relevan
                doc_info = document_scores[0]
                sources.append({
                    "title": doc_info["title"],
                    "url": doc_info["url"],
                    "image": doc_info["image"],
                    "relevance_score": doc_info["relevance_score"],
                    "cosine_similarity": doc_info["cosine_similarity"]
                })
        
        print(f"\n=== SOURCE RANKING DEBUG ===")
        for i, doc_info in enumerate(document_scores[:3], 1):
            print(f"[{i}] SCORE: {doc_info['score']:.3f} (Relevance: {doc_info['relevance_score']:.3f}, Cosine: {doc_info['cosine_similarity']:.3f})")
            print(f"    TITLE: {doc_info['title']}")
            print(f"    URL: {doc_info['url']}\n")
        
        return sources

    def _calculate_document_relevance(self, doc, answer_lower, answer_phrases, question_lower, question_keywords):
        """Calculate how relevant a document is to the answer and question"""
        snippet = doc.page_content.strip().lower()
        title = doc.metadata.get("title", "").lower()
        
        total_score = 0.0
        
        # 1. Direct content overlap with answer (highest weight)
        content_overlap_score = 0.0
        for phrase in answer_phrases:
            if len(phrase) > 5:  # Only consider meaningful phrases
                # Check for exact phrase matches
                if phrase in snippet:
                    content_overlap_score += 3.0
                # Check for partial matches (word overlap)
                phrase_words = set(phrase.split())
                snippet_words = set(snippet.split())
                overlap_ratio = len(phrase_words & snippet_words) / len(phrase_words) if phrase_words else 0
                if overlap_ratio > 0.5:  # More than 50% word overlap
                    content_overlap_score += overlap_ratio * 2.0
        
        # 2. Title relevance to question (high weight)
        title_score = 0.0
        title_words = set(title.split())
        title_question_overlap = len(question_keywords & title_words) / len(question_keywords) if question_keywords else 0
        title_score = title_question_overlap * 2.5
        
        # 3. Content relevance to question (medium weight)
        question_score = 0.0
        snippet_words = set(snippet.split())
        question_content_overlap = len(question_keywords & snippet_words) / len(question_keywords) if question_keywords else 0
        question_score = question_content_overlap * 1.5
        
        # 4. Specific keyword matching (medium weight)
        keyword_score = 0.0
        for keyword in question_keywords:
            if len(keyword) > 2:  # Skip very short words
                if keyword in title:
                    keyword_score += 1.0  # Title match is more important
                elif keyword in snippet:
                    keyword_score += 0.5
        
        # 5. Answer-specific content matching (highest weight)
        answer_content_score = 0.0
        # Check if document content appears in the answer
        doc_sentences = snippet.split('.')
        for sentence in doc_sentences:
            if len(sentence.strip()) > 10:
                sentence_clean = sentence.strip().lower()
                if sentence_clean in answer_lower:
                    answer_content_score += 4.0  # Very high score for direct usage
                else:
                    # Check for word overlap with answer
                    sentence_words = set(sentence_clean.split())
                    answer_words = set(answer_lower.split())
                    overlap = len(sentence_words & answer_words)
                    if overlap > 3:  # Significant word overlap
                        answer_content_score += (overlap / len(sentence_words)) * 2.0
        
        # Combine all scores
        total_score = (
            answer_content_score * 0.4 +    # 40% weight - most important
            content_overlap_score * 0.25 +  # 25% weight
            title_score * 0.2 +             # 20% weight
            question_score * 0.1 +          # 10% weight
            keyword_score * 0.05            # 5% weight
        )
        
        return total_score

    def query_rag_dual_language(self, user_question: str) -> dict:
        """Enhanced RAG function with dual language support with flexible source listing"""
    
        # Detect input language
        input_lang = detect_language(user_question)
        print(f"\n🌐 Detected Language: {input_lang}")
    
        # Determine which dataset to try first
        use_indonesian_dataset = is_indonesian(user_question)
        print(f"📚 Primary dataset: {'Indonesian' if use_indonesian_dataset else 'English'}")
    
        result = None
        dataset_used = "none"
        question_en = None
    
        # Try with Indonesian dataset first if applicable
        if use_indonesian_dataset and self.retriever_indonesia:
            print("🔍 Searching in Indonesian dataset...")
            result = self.qa_chain_indonesia.invoke(user_question)
            full_answer = result['result'].strip()
        
            # Check if answer indicates no source was found
            if "[NO SOURCE USED]" in full_answer:
                print("⚠ No useful results in Indonesian dataset, trying English...")
                result = None
            else:
                dataset_used = "indonesian"
    
        # If no results from Indonesian or question is in other language
        if result is None and self.retriever_english:
            print("🔍 Searching in English dataset...")
        
            # Translate question to English if needed
            if input_lang != 'en':
                question_en = translate_to_english(user_question, input_lang)
                print(f"🔄 Translated question: {question_en}")
            else:
                question_en = user_question
            
            result = self.qa_chain_english.invoke(question_en)
            full_answer_en = result['result'].strip()
            dataset_used = "english"
        
            # Translate answer back to user's language if needed
            if input_lang != 'en':
                # Extract clean answer without tags
                if "[SOURCES USED]" in full_answer_en:
                    clean_answer_en = full_answer_en.replace("[SOURCES USED]", "").strip()
                    tag = "[SOURCES USED]"
                elif "[NO SOURCE USED]" in full_answer_en:
                    clean_answer_en = full_answer_en.replace("[NO SOURCE USED]", "").strip()
                    tag = "[NO SOURCE USED]"
                else:
                    clean_answer_en = full_answer_en
                    tag = ""
            
                # Translate clean answer
                translated_answer = translate_from_english(clean_answer_en, input_lang)
                full_answer = translated_answer + (" " + tag if tag else "")
                print(f"🔄 Translated answer: {translated_answer}")
            else:
                full_answer = full_answer_en
    
        # Fallback if no results found in either dataset
        if result is None:
            error_msg = "Maaf, informasi tidak ditemukan dalam dataset kami." if use_indonesian_dataset else "Sorry, the information was not found in our dataset."
            return {
                "input_language": input_lang,
                "question": user_question,
                "answer": error_msg,
                "sources": [],
                "dataset_used": "none"
            }
    
        # Check if model used sources
        if "[SOURCES USED]" in full_answer:
            use_sources = True
            clean_answer = full_answer.replace("[SOURCES USED]", "").strip()
        elif "[NO SOURCE USED]" in full_answer:
            use_sources = False
            clean_answer = full_answer.replace("[NO SOURCE USED]", "").strip()
        else:
            use_sources = False
            clean_answer = full_answer
    
        # Process source documents
        sources = []
        if use_sources and result.get("source_documents"):
            print("\n=== DEBUGGING: SOURCE DOCUMENTS ===")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"[{i}] TITLE:", doc.metadata.get("title"))
                print("     URL  :", doc.metadata.get("url"))
                print("     IMAGE:", doc.metadata.get("image"))
                print("     SAMPLE:", doc.page_content[:80].replace("\n", " "), "...\n")
        
            # Filter and prepare sources
            search_query = user_question if dataset_used == "indonesian" else question_en if question_en else user_question
            sources = self._filter_relevant_sources(
                result["source_documents"], 
                clean_answer, 
                search_query
            )
    
        return {
            "input_language": input_lang,
            "question": user_question,
            "answer": clean_answer,
            "sources": sources,
            "dataset_used": dataset_used
        }

# Initialize global RAG instance
rag_system = DualLanguageRAG()

def query_rag_multilang(user_question: str) -> dict:
    """Main function to query the dual language RAG system"""
    return rag_system.query_rag_dual_language(user_question)

def run_multilang_rag_pipeline():
    """Interactive CLI"""
    print("🧠 Budaya Bali RAG Dual Language System")
    print("Sistem ini akan:")
    print("- Menggunakan dataset Indonesia untuk pertanyaan berbahasa Indonesia")
    print("- Menggunakan dataset English untuk pertanyaan berbahasa lain")
    print("Type 'exit' atau 'quit' untuk keluar.\n")

    while True:
        user_input = input("Pertanyaan Anda / Your question:\n> ").strip()

        if user_input.lower() in ["exit", "quit", "keluar"]:
            print("Goodbye! / Selamat tinggal!")
            break

        try:
            result = query_rag_multilang(user_input)

            print(f"\n📌 Answer ({result['dataset_used']} dataset):\n{result['answer']}")
            
            if result['sources']:
                print(f"\n📚 Sources Used:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['title']}")
                    print(f"   {source['url']}")
                    if source.get('cosine_similarity') is not None:
                        print(f"   🔍 Similarity Score: {source['cosine_similarity']:.3f}")
                    if source['image']:
                        print(f"   📷 Image: {source['image']}")
            else:
                print("\n⚠ No sources included.")

            print("-" * 60)

        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    run_multilang_rag_pipeline()
