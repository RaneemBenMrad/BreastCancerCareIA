import os
import tempfile
import shutil
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import re
import random
import numpy as np
import pandas as pd
import joblib
import requests
from bs4 import BeautifulSoup
from collections import Counter
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from langdetect import detect
from groq import Groq
import logging
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import io
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Configuration initiale
load_dotenv()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = 'votre_cle_secrete_ici'

# Chargement des modèles
try:
    svm_model = joblib.load('./svm_model_10features.pkl')
    scaler = joblib.load('./scaler_10features.pkl')
    print("SVM model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

try:
    cnn_model = load_model('./backup_model_augmented.h5')
    print("CNN model loaded successfully")
except Exception as e:
    print(f"Error loading CNN model: {e}")
    cnn_model = None

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("La clé GROQ_API_KEY est manquante dans le fichier .env")

try:
    groq_client = Groq(api_key=groq_api_key)
    print("Groq client initialized successfully")
except Exception as e:
    print(f"Error initializing Groq client: {e}")

users_db = {}
embedding_model = None  # Global variable to avoid redundant loading

# Validation du mot de passe
def validate_password(password):
    if len(password) < 12:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'[0-9]', password):
        return False
    if not re.search(r'[^A-Za-z0-9]', password):
        return False
    return True

# Mots-clés liés au cancer du sein (avec variantes pour tolérer les erreurs)
BREAST_CANCER_KEYWORDS = {
    'en': ['breast cancer', 'breast', 'tumor', 'mammogram', 'chemotherapy', 'radiation', 'lumpectomy', 'mastectomy',
           'detection', 'symptoms', 'treatment'],
    'fr': ['cancer du sein', 'sein', 'tumeur', 'mammographie', 'chimiothérapie', 'radiothérapie', 'tumorectomie',
           'mastectomie', 'détection', 'symptômes', 'traitement', 'cancer desien', 'cancer de sient'],
    'ar': ['سرطان الثدي', 'الثدي', 'ورم', 'تصوير الثدي', 'العلاج الكيميائي', 'العلاج الإشعاعي', 'استئصال الورم',
           'استئصال الثدي', 'الكشف', 'الأعراض', 'العلاج']
}

# Vérification du contenu lié au cancer du sein
def is_breast_cancer_related(text):
    text = text.lower()
    print(f"Checking if text is related to breast cancer: {text[:100]}...")
    try:
        lang = detect(text)
    except:
        lang = 'fr'  # Default to French for this document
    keywords = BREAST_CANCER_KEYWORDS.get(lang, BREAST_CANCER_KEYWORDS['en'])
    if any(keyword.lower() in text for keyword in keywords):
        print(f"Found breast cancer related keyword in lang {lang}")
        return True
    print("No breast cancer related keywords found.")
    return False

def is_breast_cancer_related_chat(message, lang):
    message = message.lower().replace("cest", "c'est").replace("sien", "sein").replace("sient", "sein")
    keywords = BREAST_CANCER_KEYWORDS.get(lang, BREAST_CANCER_KEYWORDS['en'])
    return any(keyword.lower() in message for keyword in keywords)

# Chargement du PDF
def pdf_loader(uploaded_file):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs or not any(doc.page_content.strip() for doc in docs):
            raise ValueError("Le PDF est vide ou ne peut pas être chargé.")

        full_text = " ".join(doc.page_content for doc in docs)
        print(f"Extracted text from PDF: {full_text[:100]}...")
        if not is_breast_cancer_related(full_text):
            raise ValueError("Le contenu du PDF ne semble pas être lié au cancer du sein.")

        return docs, None
    except Exception as e:
        return None, f"Erreur lors du chargement du PDF : {str(e)}"
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Erreur lors de la suppression de {temp_path}: {e}")

# Traitement des documents
def split_document(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    return splitter.split_documents(documents)

def vector_db(chunks):
    global embedding_model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
    return FAISS.from_documents(chunks, embedding_model)

# Chaîne RAG
def rag(vdb, question):
    retriever = vdb.as_retriever()
    llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return rag_chain.invoke({'query': question})

# Stopwords arabes pour le résumé
arabic_stopwords = [
    'و', 'في', 'من', 'على', 'إلى', 'عن', 'هذا', 'هذه', 'الذي', 'التي', 'ما', 'كان', 'كانت',
    'بعد', 'قبل', 'مع', 'أو', 'لكن', 'إذا', 'حتى', 'أن', 'لا', 'لم', 'لن', 'له', 'لها',
    'هم', 'هن', 'هو', 'هي', 'كل', 'بعض', 'أي', 'بين', 'خلال', 'حول', 'فيه', 'عليه', 'منه'
]

# Résumé de texte
def summarize_text(text, num_sentences=3):
    sentences = re.split(r'[.!?؟]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < num_sentences:
        return text
    words = [word.lower() for word in re.findall(r'\w+', text, re.UNICODE) if word.lower() not in arabic_stopwords]
    word_freq = Counter(words)
    sentence_scores = {}
    for sentence in sentences:
        sentence_words = [word.lower() for word in re.findall(r'\w+', sentence, re.UNICODE)]
        score = sum(word_freq.get(word, 0) for word in sentence_words)
        sentence_scores[sentence] = score
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    summary = [sentence for sentence in sentences if sentence in top_sentences]
    return ' '.join(summary)

# Génération de réponse
def generate_response(message, lang):
    try:
        print(f"Processing question: {message} (lang: {lang})")
        normalized_message = message.lower().replace('cest', "c'est").replace('sien', 'sein').replace('sient', 'sein')\
            .replace('haw', 'how').replace('knaw', 'know')
        if not is_breast_cancer_related_chat(normalized_message, lang):
            return {
                'en': 'Please ask a question related to breast cancer.',
                'fr': 'Veuillez poser une question liée au cancer du sein.',
                'ar': 'يرجى طرح سؤال متعلق بسرطان الثدي.'
            }.get(lang, 'Please ask a question related to breast cancer.')

        system_prompts = {
            'en': """
            You are a helpful assistant specializing in breast cancer information. Provide accurate, concise answers related to breast cancer symptoms, detection, treatments, risk factors, and support. Answer only in English unless specified. If the question is unclear, ask for clarification.
            """,
            'fr': """
            Vous êtes un assistant utile spécialisé dans les informations sur le cancer du sein. Fournissez des réponses précises et concises sur les symptômes, la détection, les traitements, les facteurs de risque et le soutien. Répondez uniquement en français. Si la question est floue, demandez des précisions.
            """,
            'ar': """
            أنت مساعد مفيد متخصص في معلومات سرطان الثدي. قدم إجابات دقيقة وموجزة تتعلق بأعراض سرطان الثدي، الكشف عنه، العلاجات، عوامل الخطر، والدعم. أجب فقط بالعربية. إذا كان السؤال غير واضح، اطلب توضيحًا.
            """
        }
        system_prompt = system_prompts.get(lang, system_prompts['en'])
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": normalized_message}
            ],
            temperature=0.7,
            max_tokens=200
        )
        answer = response.choices[0].message.content.strip()
        if len(answer) < 10 or not is_breast_cancer_related_chat(answer, lang):
            answer = {
                'en': 'I couldn’t find a precise answer. Please rephrase your question or ask something specific about breast cancer.',
                'fr': 'Je n’ai pas trouvé de réponse précise. Veuillez reformuler votre question ou poser une question spécifique sur le cancer du sein.',
                'ar': 'لم أتمكن من العثور على إجابة دقيقة. من فضلك، أعد صياغة سؤالك أو اسأل شيئًا محددًا عن سرطان الثدي.'
            }.get(lang)
        return answer
    except Exception as e:
        print(f"Groq error: {e}")
        return {
            'en': 'An error occurred while processing your question. Please check your API key or try again.',
            'fr': 'Une erreur s’est produite lors du traitement de votre question. Vérifiez votre clé API ou réessayez.',
            'ar': 'حدث خطأ أثناء معالجة سؤالك. تحقق من مفتاح API الخاص بك أو حاول مرة أخرى.'
        }.get(lang)

# Prétraitement des images
def preprocess_image(image_file):
    try:
        img = Image.open(image_file)
        img.verify()
        image_file.seek(0)
        img = Image.open(image_file)
        img = img.resize((256, 256))
        img_array = img_to_array(img)
        if img_array.shape[-1] != 3:
            img = img.convert('RGB')
            img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Gestion des index FAISS
def save_faiss_index(vdb, user_id):
    os.makedirs("faiss_indices", exist_ok=True)
    index_path = os.path.join("faiss_indices", f"faiss_{user_id}.index")
    vdb.save_local(index_path)
    return index_path

def load_faiss_index(index_path):
    global embedding_model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
    return FAISS.load_local(index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)

# Routes
@app.route('/chatbot', methods=['POST'])
def chatbot():
    if 'chat_history' not in session:
        session['chat_history'] = []
    data = request.get_json()
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'response': 'Please enter a valid question.'})
    try:
        lang = detect(message)
    except:
        lang = 'fr'  # Default to French
    answer = generate_response(message, lang)
    session['chat_history'].append({'message': message, 'response': answer})
    return jsonify({'response': answer})

@app.route('/chatbot/history', methods=['GET'])
def get_chatbot_history():
    return jsonify({'history': session.get('chat_history', [])})

@app.route('/pdf_chat', methods=['GET', 'POST'])
def pdf_chat():
    user_id = session.get('email', 'guest')
    if 'pdf_vdb_path' not in session:
        session['pdf_vdb_path'] = None
    if 'pdf_uploaded' not in session:
        session['pdf_uploaded'] = False

    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            flash("No file uploaded.", 'error')
            return render_template('pdf_chat.html', error="No file uploaded.")

        pdf_file = request.files['pdf_file']
        if pdf_file.filename == '':
            flash("No file selected.", 'error')
            return render_template('pdf_chat.html', error="No file selected.")

        documents, error_message = pdf_loader(pdf_file)
        if error_message:
            flash(error_message, 'error')
            return render_template('pdf_chat.html', error=error_message)

        chunks = split_document(documents)
        vdb = vector_db(chunks)
        index_path = save_faiss_index(vdb, user_id)
        session['pdf_vdb_path'] = index_path
        session['pdf_uploaded'] = True  # Set flag to indicate successful upload
        flash("PDF processed successfully. You can now ask questions.", 'success')
        return render_template('pdf_chat.html', success="PDF processed successfully. You can now ask questions.", pdf_uploaded=True)

    if request.method == 'GET':
        question = request.args.get('question', '').strip()
        if not question:
            # Reset session state on initial page load (no question provided)
            session['pdf_uploaded'] = False
            session.pop('pdf_vdb_path', None)
            return render_template('pdf_chat.html', pdf_uploaded=False)

        if session.get('pdf_vdb_path') and session.get('pdf_uploaded'):
            try:
                vdb = load_faiss_index(session['pdf_vdb_path'])
                try:
                    lang = detect(question)
                except:
                    lang = 'fr'  # Default to French
                print(f"Processing question: {question} (lang: {lang})")
                if not is_breast_cancer_related_chat(question, lang):
                    flash("Please ask a question related to breast cancer.", 'error')
                    return render_template('pdf_chat.html', error="Please ask a question related to breast cancer.", pdf_uploaded=True)
                result = rag(vdb, question)
                print(f"RAG result: {result['result']}")
                print(f"Sources: {[doc.page_content for doc in result['source_documents']]}")  # Debug sources
                return render_template('pdf_chat.html',
                                      response=result['result'],
                                      sources=[doc.page_content for doc in result['source_documents']],
                                      question=question,
                                      pdf_uploaded=True,
                                      debug_response=result['result'])  # Debug response
            except Exception as e:
                flash(f"Error loading FAISS index: {str(e)}", 'error')
                session.pop('pdf_vdb_path', None)
                session['pdf_uploaded'] = False
                return render_template('pdf_chat.html', error=f"Error loading FAISS index: {str(e)}", pdf_uploaded=False)
        else:
            return render_template('pdf_chat.html', pdf_uploaded=False)

@app.route('/')
@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        mock_user = {
            'first_name': 'Test',
            'last_name': 'User',
            'email': 'test@example.com'
        }
        return render_template('dashboard.html', user=mock_user)
    return render_template('dashboard.html', user=users_db[session['email']])

@app.route('/api/news')
def get_news():
    news = [
        {"title": "Many People With Cancer Wait Too Long to Start"},
        {"title": "New Poll Shows That People Don't Trust AI-Generated"},
        {"title": "Black Women Less Likely to Receive Certain Targeted"},
        {"title": "Walking More May Help You Live Longer After Cancer"}
    ]
    return jsonify(news)

@app.route('/api/testimonials')
def get_testimonials():
    testimonials = [
        {
            "name": "VANESSA",
            "since": "2023",
            "photo": f"https://randomuser.me/api/portraits/women/{random.randint(1, 100)}.jpg",
            "quote": "It's so comforting to relate to other people within your age range walking through the world with the same diagnosis as you..."
        }
    ]
    return jsonify(testimonials)

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

@app.route('/donate', methods=['GET', 'POST'])
def donate():
    if request.method == 'POST':
        amount = request.form.get('amount')
        name = request.form.get('name')
        email = request.form.get('email')
        flash('Thank you for your donation! (This is a placeholder response)', 'success')
        return redirect(url_for('donate'))
    return render_template('donate.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email not in users_db:
            flash('Email non trouvé. Veuillez vous inscrire.', 'error')
            return redirect(url_for('signup'))
        if not check_password_hash(users_db[email]['password'], password):
            flash('Mot de passe incorrect.', 'error')
            return redirect(url_for('login'))
        session['email'] = email
        flash('Connexion réussie!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        password = request.form.get('password')
        if email in users_db:
            flash('Cet email est déjà enregistré.', 'error')
            return redirect(url_for('signup'))
        if not validate_password(password):
            flash(
                'Le mot de passe doit contenir au moins 12 caractères, incluant une majuscule, une minuscule, un chiffre et un caractère spécial.',
                'error')
            return redirect(url_for('signup'))
        users_db[email] = {
            'first_name': first_name,
            'last_name': last_name,
            'password': generate_password_hash(password)
        }
        flash('Inscription réussie! Veuillez vous connecter.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    user_id = session.get('email', 'guest')
    index_path = os.path.join("faiss_indices", f"faiss_{user_id}.index")
    if os.path.exists(index_path):
        try:
            shutil.rmtree(index_path)
        except Exception as e:
            print(f"Erreur lors du nettoyage de l'index FAISS: {e}")
    session.pop('email', None)
    session.pop('pdf_vdb_path', None)
    session.pop('pdf_uploaded', None)
    flash('Déconnexion réussie.', 'info')
    return redirect(url_for('dashboard'))

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    return render_template('analyze.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    return render_template('chat.html')

@app.route('/tumor-details-prediction', methods=['GET', 'POST'])
def tumor_details_prediction():
    if request.method == 'POST':
        try:
            input_data = {
                'radius_mean': float(request.form['radius_mean']),
                'texture_mean': float(request.form['texture_mean']),
                'perimeter_mean': float(request.form['perimeter_mean']),
                'area_mean': float(request.form['area_mean']),
                'smoothness_mean': float(request.form['smoothness_mean']),
                'concavity_mean': float(request.form['concavity_mean']),
                'concave points_mean': float(request.form['concave points_mean']),
                'radius_worst': float(request.form['radius_worst']),
                'texture_worst': float(request.form['texture_worst']),
                'concave points_worst': float(request.form['concave points_worst'])
            }
            for key, value in input_data.items():
                if value < 0:
                    flash(f"The value of {key} must be positive.", 'error')
                    return render_template('tumor_details.html')
            feature_columns = [
                'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                'concavity_mean', 'concave points_mean', 'radius_worst', 'texture_worst', 'concave points_worst'
            ]
            input_df = pd.DataFrame([input_data], columns=feature_columns)
            input_scaled = scaler.transform(input_df)
            prediction = svm_model.predict(input_scaled)[0]
            prediction_result = "Benign (B)" if prediction == 'B' else "Malignant (M)"
            return render_template('result.html', prediction=prediction_result)
        except ValueError:
            flash("Please enter valid numeric values.", 'error')
            return render_template('tumor_details.html')
        except Exception as e:
            flash(f"Error in prediction: {str(e)}", 'error')
            return render_template('tumor_details.html')
    return render_template('tumor_details.html')

@app.route('/ultrasound-image-prediction', methods=['GET', 'POST'])
def ultrasound_image_prediction():
    if request.method == 'POST':
        if 'ultrasound_image' not in request.files:
            flash("No images uploaded.", 'error')
            return render_template('ultrasound_image.html')

        image_file = request.files['ultrasound_image']
        if image_file.filename == '':
            flash("No files selected.", 'error')
            return render_template('ultrasound_image.html')

        if not cnn_model:
            flash("Error: CNN model not loaded.", 'error')
            return render_template('ultrasound_image.html')

        os.makedirs('uploads', exist_ok=True)
        filepath = os.path.join('uploads', image_file.filename)
        image_file.save(filepath)

        try:
            img_array = preprocess_image(open(filepath, 'rb'))
            if img_array is None:
                flash("Error: Invalid image format or corrupt image.", 'error')
                os.remove(filepath)
                return render_template('ultrasound_image.html')

            prediction = cnn_model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = prediction[0][predicted_class] * 100

            class_mapping = {0: 'Normal', 1: 'Benign', 2: 'Malignant'}
            prediction_result = (
                f"{class_mapping[predicted_class]} ({confidence:.2f}% confidence).<br>"
                "<strong>Warning:</strong> This prediction is experimental and does not replace a medical diagnosis."
                "Consult a radiologist or physician for a professional evaluation."
            )

            os.remove(filepath)
            return render_template('ultrasound_image.html', prediction=prediction_result)

        except Exception as e:
            print(f"Prediction error: {e}")
            flash(f"Error during prediction: {str(e)}", 'error')
            if os.path.exists(filepath):
                os.remove(filepath)
            return render_template('ultrasound_image.html')

    return render_template('ultrasound_image.html')

@app.route('/article-summarizer', methods=['GET', 'POST'])
def article_summarizer():
    if request.method == 'POST':
        try:
            article_url = request.form.get('article_url')
            article_text = request.form.get('article_text')
            if not article_text and not article_url:
                flash("Veuillez fournir un texte ou une URL d'article.", 'error')
                return render_template('article_summarizer.html')
            if article_url:
                try:
                    session = requests.Session()
                    retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
                    session.mount('https://', HTTPAdapter(max_retries=retries))
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = session.get(article_url, headers=headers, timeout=120)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    main_content = (soup.find('main') or
                                    soup.find('article') or
                                    soup.find('div', class_=re.compile('content|article|main|body|post|page', re.I)) or
                                    soup.find('div', id=re.compile('content|article|main|body', re.I)) or
                                    soup.find('section'))
                    if main_content:
                        text_elements = main_content.find_all(
                            ['p', 'div', 'span', 'h1', 'h2', 'h3', 'li', 'article', 'section'])
                    else:
                        text_elements = soup.find_all(
                            ['p', 'div', 'span', 'h1', 'h2', 'h3', 'li', 'article', 'section'])
                    article_text = ' '.join(
                        [elem.get_text(strip=True) for elem in text_elements if elem.get_text(strip=True)])
                    if not article_text.strip():
                        flash("Impossible d'extraire le texte de cette URL.", 'error')
                        return render_template('article_summarizer.html')
                except requests.RequestException as e:
                    flash(f"Erreur lors de la récupération de l'URL : {str(e)}", 'error')
                    return render_template('article_summarizer.html')
            if len(article_text.strip()) < 30:
                flash("Le texte fourni est trop court pour être résumé.", 'error')
                return render_template('article_summarizer.html')
            normalized_text = re.sub(r'[-_\s]+', ' ', article_text.lower())
            keywords = [
                "cancer du sein", "carcinome mammaire", "tumeur du sein",
                "breast cancer", "breast tumor", "mammary carcinoma",
                "سرطان الثدي", "ورم الثدي", "سرطان الثدي الخبيث"
            ]
            if not any(keyword.lower() in normalized_text for keyword in keywords):
                flash("L'article ne semble pas traiter du cancer du sein.", 'error')
                return render_template('article_summarizer.html')
            summary = summarize_text(article_text, num_sentences=3)
            return render_template('article_summarizer.html', summary=summary)
        except Exception as e:
            flash(f"Erreur lors du résumé : {str(e)}", 'error')
            return render_template('article_summarizer.html')
    return render_template('article_summarizer.html')

if __name__ == '__main__':
    if not users_db:
        users_db['test@example.com'] = {
            'first_name': 'Test',
            'last_name': 'User',
            'password': generate_password_hash('TestPassword123!')
        }
    print("\nRoutes disponibles:")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule}")
    app.run(host='127.0.0.1', port=5001, debug=False)