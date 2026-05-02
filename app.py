import os
import sqlite3
import urllib.parse
from math import radians, sin, cos, sqrt, atan2

import pandas as pd
from flask import Flask, request, jsonify, render_template, abort, redirect, url_for, flash, session
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

# Import the hybrid model function
# Ensure hybrid_recommender.py is in the same folder
from hybrid_recommender import predict_hybrid_rating 

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_and_secure_key_change_me' 
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' 

DB_FILE_PATH = 'hospital_reviews.db'

# --- User Class ---
class User(UserMixin):
    def __init__(self, id, username, email, password, last_symptom=None, last_dept=None):
        self.id = id
        self.username = username
        self.email = email
        self.password = password
        self.last_symptom = last_symptom
        self.last_dept = last_dept

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    u = cursor.fetchone()
    conn.close()
    if u:
        # This handles cases where the tuple might be 4 or 6 elements during migration
        # u[0]=id, u[1]=user, u[2]=email, u[3]=pass, u[4]=symptom, u[5]=dept
        return User(u[0], u[1], u[2], u[3], 
                    u[4] if len(u) > 4 else None, 
                    u[5] if len(u) > 5 else None)
    return None

# --- Database Initialization & Migration ---
def init_db():
    conn = sqlite3.connect(DB_FILE_PATH)
    # 1. Create tables if they don't exist
    conn.execute('''CREATE TABLE IF NOT EXISTS users 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                     username TEXT UNIQUE, email TEXT UNIQUE, password TEXT,
                     last_symptom TEXT, last_dept TEXT)''')
    conn.execute('CREATE TABLE IF NOT EXISTS bookmarks (user_id INTEGER, hospital_name TEXT, PRIMARY KEY(user_id, hospital_name))')
    
    # 2. MIGRATION: Check if old database needs new columns
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'last_symptom' not in columns:
        print("Migrating database: Adding last_symptom column...")
        conn.execute("ALTER TABLE users ADD COLUMN last_symptom TEXT")
    if 'last_dept' not in columns:
        print("Migrating database: Adding last_dept column...")
        conn.execute("ALTER TABLE users ADD COLUMN last_dept TEXT")
        
    conn.commit()
    conn.close()

init_db()

# --- Load Dataset ---
def load_review_data():
    global df
    try:
        conn = sqlite3.connect(DB_FILE_PATH)
        query = """
        SELECT h.hospital_name AS "Hospital_Name", r.star_rating AS "Star rating", 
               r.review_text_raw AS "Review content", r.sentiment_score AS "Sentiment Score"
        FROM reviews r JOIN hospitals h ON r.hospital_id = h.hospital_id;
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        df['Sentiment Score'] = pd.to_numeric(df['Sentiment Score'], errors='coerce')
        df.dropna(subset=['Review content', 'Hospital_Name', 'Star rating'], inplace=True)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"DATABASE ERROR: {e}")
        df = None

load_review_data()

# --- Hospital Database with Verified Insurance ---
HOSPITAL_INFO_DB = {
    'Sunway Medical Centre Penang': {'location': '3106, Lebuh Tenggiri 2 Pusat Bandar Seberang Jaya, 13700 Perai, Pulau Pinang', 'hours': 'Open 24 Hours', 'days': 'Monday - Sunday', 'phone': '+60 4-373 9191', 'lat': 5.3905, 'lon': 100.3986, 'category': 'Private', 'insurance': ['AIA', 'Allianz', 'Etiqa', 'Great Eastern', 'Hong Leong Assurance', 'Manulife', 'Prudential', 'Prudential BSN', 'Income Insurance', 'Zurich']},
    'LohGuanLye Specialists Centre': {'location': '238, Jln Macalister, 10400 George Town, Pulau Pinang', 'hours': 'Open 24 Hours', 'days': 'Monday - Sunday', 'phone': '+60 4-238 8888', 'lat': 5.4162, 'lon': 100.3204, 'category': 'Private', 'insurance': ['AIA', 'Allianz', 'Generali (AXA)', 'Etiqa', 'Great Eastern', 'Manulife', 'Prudential', 'Tokio Marine', 'Zurich']},
    'Gleneagles Hospital Penang': {'location': '1, Jalan Pangkor, 10050 George Town, Pulau Pinang', 'hours': 'Open 24 Hours', 'days': 'Monday - Sunday', 'phone': '+60 4-222 9111', 'lat': 5.4262, 'lon': 100.3159, 'category': 'Private', 'insurance': ['AIA', 'Allianz', 'Great Eastern', 'Prudential', 'HSBC Amanah', 'Zurich', 'AIG', 'Etiqa']},
    'Pantai Hospital Penang': {'location': '82, Jalan Tengah, Bayan Baru, 11900 Bayan Lepas, Pulau Pinang', 'hours': 'Open 24 Hours', 'days': 'Monday - Sunday', 'phone': '+60 4-643 3888', 'lat': 5.3275, 'lon': 100.2855, 'category': 'Private', 'insurance': ['AIA', 'Allianz', 'Prudential', 'Great Eastern', 'Manulife', 'Etiqa', 'Liberty Insurance', 'Generali']},
    'Island Hospital Penang': {'location': '308, Jln Macalister, 10450 George Town, Pulau Pinang', 'hours': '8:30am - 5pm', 'days': 'Monday - Saturday', 'phone': '+60 4-238 3388', 'lat': 5.4224, 'lon': 100.314, 'category': 'Private', 'insurance': ['AIA', 'Allianz', 'Great Eastern', 'Prudential', 'Manulife', 'Generali', 'FWD Takaful']},
    'KPJ Penang Specialist Hospital': {'location': '570, Jln Perda Utama, Bandar Perda, 14000 Bukit Mertajam, Pulau Pinang', 'hours': 'Open 24 Hours', 'days': 'Monday - Sunday', 'phone': '+60 4-548 6688', 'lat': 5.3703, 'lon': 100.4357, 'category': 'Private', 'insurance': ['AIA', 'Allianz', 'Prudential', 'Etiqa', 'Takaful Ikhlas', 'Great Eastern Takaful']},
    'Bagan Specialist Centre': {'location': 'Jalan Bagan 1, Taman Bagan, 13400 Butterworth, Pulau Pinang', 'hours': 'Open 24 Hours', 'days': 'Monday - Sunday', 'phone': '+60 4-371 0000', 'lat': 5.4091, 'lon': 100.3848, 'category': 'Private', 'insurance': ['AIA', 'Allianz', 'Great Eastern', 'Prudential', 'Etiqa', 'Manulife']},
    'Lam Wah Ee Hospital': {'location': '141, Jln Tan Sri Teh Ewe Lim, 11600 Jelutong, Pulau Pinang', 'hours': 'Open 24 Hours', 'days': 'Monday - Sunday', 'phone': '+60 4-652 8888', 'lat': 5.3922, 'lon': 100.3039, 'category': 'Not-for-profit Private', 'insurance': ['AIA', 'Allianz', 'Prudential', 'Great Eastern', 'Manulife', 'Etiqa']},
    'Penang Adventist Hospital': {'location': '465, Jalan Burma, 10350 George Town, Pulau Pinang', 'hours': 'Open 24 Hours', 'days': 'Monday - Sunday', 'phone': '+60 4-222 7200', 'lat': 5.4328, 'lon': 100.3053, 'category': 'Private', 'insurance': ['AIA', 'Allianz', 'Great Eastern', 'Prudential', 'Etiqa', 'Manulife', 'Zurich']},
    'MountMiriam Cancer Hospital': {'location': '23, Jalan Bulan, Fettes Park, 11200 Tanjung Bungah, Pulau Pinang', 'hours': 'Open 24 Hours', 'days': 'Monday - Sunday', 'phone': '+60 4-892 3999', 'lat': 5.4591, 'lon': 100.301, 'category': 'Specialist Hospital', 'insurance': ['AIA', 'Allianz', 'Prudential', 'Great Eastern', 'Etiqa']},
    'Penang General Hospital': {'location': 'Jalan Residensi, 10450 George Town, Pulau Pinang', 'hours': 'Open 24 Hours', 'days': 'Monday - Sunday', 'phone': '+60 4-222 5333', 'lat': 5.4165, 'lon': 100.3111, 'category': 'Government (MOH)', 'insurance': ['Government GL', 'Self-Pay (Subsidized)', 'Private Reimbursement']},
    'Hospital Seberang Jaya': {'location': 'Jln Tun Hussein Onn, Seberang Jaya, 13700 Perai, Pulau Pinang', 'hours': 'Open 24 Hours', 'days': 'Monday - Sunday', 'phone': '+60 4-382 7333', 'lat': 5.3942, 'lon': 100.4082, 'category': 'Government (MOH)', 'insurance': ['Government GL', 'Self-Pay (Subsidized)', 'Private Reimbursement']},
    'Bukit Mertajam Hospital': {'location': 'Jln Kulim, 14000 Bukit Mertajam, Pulau Pinang', 'hours': 'Open 24 Hours', 'days': 'Monday - Sunday', 'phone': '+60 4-549 7333', 'lat': 5.36, 'lon': 100.4645, 'category': 'Government (MOH)', 'insurance': ['Government GL', 'Self-Pay (Subsidized)', 'Private Reimbursement']},
    'Kepala Batas Hospital': {'location': 'Jalan Bertam 2, 13200 Kepala Batas, Pulau Pinang', 'hours': 'Open 24 Hours', 'days': 'Monday - Sunday', 'phone': '+60 4-579 3333', 'lat': 5.5167, 'lon': 100.4286, 'category': 'Government (MOH)', 'insurance': ['Government GL', 'Self-Pay (Subsidized)', 'Private Reimbursement']},
    'Balik Pulau Hospital': {'location': 'Jalan Balik Pulau, 11000 Balik Pulau, Pulau Pinang', 'hours': 'Open 24 Hours', 'days': 'Monday - Sunday', 'phone': '+60 4-866 9333', 'lat': 5.3508, 'lon': 100.233, 'category': 'Government (MOH)', 'insurance': ['Government GL', 'Self-Pay (Subsidized)', 'Private Reimbursement']}
}

DEPARTMENT_DATA = {
    'Ophthalmology': ['eye', 'vision', 'blurry', 'glaucoma', 'cataract'], 
    'Dermatology': ['skin', 'rash', 'acne', 'itch'],
    'Cardiology': ['heart', 'chest pain', 'blood pressure'], 
    'Orthopedics': ['bone', 'joint', 'fracture', 'knee'],
    'Gastroenterology': ['stomach', 'digestive', 'endoscopy'], 
    'Pediatrics': ['child', 'pediatrician', 'baby'],
    'ENT (Otolaryngology)': ['ear', 'nose', 'throat', 'sinus'], 
    'Neurology': ['nerve', 'headache', 'migraine', 'seizure'],
    'Obstetrics & Gynecology (OB-GYN)': ['pregnancy', 'pregnant', 'ob-gyn']
}

POSITIVE_SENTIMENT_KEYWORDS = ['good', 'great', 'excellent', 'professional', 'friendly', 'clean', 'helpful', 'best', 'fast', 'recommended', 'care', 'efficient', 'quick']

# --- Helper Functions ---
def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_department_from_symptoms(symptom_text):
    symptom_text = symptom_text.lower()
    for dept, keywords in DEPARTMENT_DATA.items():
        for kw in keywords:
            if kw in symptom_text: return dept, kw
    return "General Medicine", None

def recommend_hospitals_for_department(department, user_lat=None, user_lon=None, top_n=3):
    if df is None: return []
    keywords = DEPARTMENT_DATA.get(department, [])
    hospital_scores = []
    
    for hospital_name in HOSPITAL_INFO_DB:
        hospital_reviews = df[df['Hospital_Name'] == hospital_name]
        search_pattern = '|'.join(keywords) if keywords else ''
        relevant_reviews = hospital_reviews[hospital_reviews['Review content'].str.contains(search_pattern, case=False, na=False)]
        
        if not relevant_reviews.empty:
            all_text = " ".join(relevant_reviews['Review content'].astype(str)).lower()
            top_pos_keywords = list(set([w for w in POSITIVE_SENTIMENT_KEYWORDS if w in all_text]))[:3]
            avg_rating = relevant_reviews['Star rating'].mean()
            avg_sentiment = (relevant_reviews['Sentiment Score'].mean() + 1) / 2 # Normalize 0 to 1
            
            rating_norm = avg_rating / 5.0
            distance_km = None
            
            if user_lat is None or user_lon is None:
                rating_cont = rating_norm * 70
                sentiment_cont = avg_sentiment * 30
                distance_cont = 0
            else:
                h = HOSPITAL_INFO_DB[hospital_name]
                distance_km = calculate_haversine_distance(user_lat, user_lon, h['lat'], h['lon'])
                dist_score = max(0, 1 - (distance_km / 25))
                # Hybrid Weights: 70% Quality (split 70/30) + 30% Distance
                rating_cont = (rating_norm * 0.7 * 0.7) * 100
                sentiment_cont = (avg_sentiment * 0.3 * 0.7) * 100
                distance_cont = (dist_score * 0.3) * 100
            
            final_score = (rating_cont + sentiment_cont + distance_cont) / 100
            hospital_scores.append({
                'hospital_name': hospital_name, 'score': final_score, 
                'avg_rating': avg_rating, 'review_count': len(relevant_reviews),
                'distance_km': round(distance_km, 1) if distance_km is not None else "N/A",
                'positive_keywords': top_pos_keywords,
                'breakdown': {'rating': round(rating_cont, 1), 'sentiment': round(sentiment_cont, 1), 'distance': round(distance_cont, 1)}
            })
    return sorted(hospital_scores, key=lambda x: x['score'], reverse=True)[:top_n]

# --- ROUTES ---

@app.route('/')
def home():
    if not current_user.is_authenticated and 'guest' not in session: 
        return redirect(url_for('welcome'))
    return render_template('homepage.html')

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

@app.route('/guest')
def guest_login():
    session['guest'] = True
    return redirect(url_for('home'))

@app.route('/symptom_recommend', methods=['POST'])
def handle_symptom_recommendation():
    data = request.get_json()
    dept, matched_kw = get_department_from_symptoms(data['symptoms'])
    recommendations = recommend_hospitals_for_department(dept, data.get('lat'), data.get('lon'))
    
    if current_user.is_authenticated:
        conn = sqlite3.connect(DB_FILE_PATH)
        conn.execute("UPDATE users SET last_symptom = ?, last_dept = ? WHERE id = ?", 
                     (data['symptoms'], dept, current_user.id))
        conn.commit(); conn.close()
    
    return jsonify({'department': dept, 'matched_keyword': matched_kw, 'recommendations': recommendations})

@app.route('/hybrid')
def hybrid_page():
    if not current_user.is_authenticated and 'guest' not in session: return redirect(url_for('login'))
    return render_template('hybrid_recommender.html', is_guest=('guest' in session))

@app.route('/get_recommendation', methods=['POST'])
def handle_hybrid_recommendation():
    if 'guest' in session: return jsonify({"error": "Member-only feature"}), 403
    data = request.get_json()
    try:
        prediction_result = predict_hybrid_rating(current_user.id, data['hospital_id'], data['review_text'])
        return jsonify(prediction_result)
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/hospital/<hospital_name>')
def hospital_details(hospital_name):
    info = HOSPITAL_INFO_DB.get(hospital_name)
    if not info: abort(404)
    h_data = info.copy(); h_data['name'] = hospital_name
    q = urllib.parse.quote_plus(f"{hospital_name} {h_data['location']}")
    h_data['google_maps_link'] = f"https://www.google.com/maps/search/?api=1&query={q}"
    is_bookmarked = False
    if current_user.is_authenticated:
        conn = sqlite3.connect(DB_FILE_PATH); cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM bookmarks WHERE user_id = ? AND hospital_name = ?", (current_user.id, hospital_name))
        is_bookmarked = cursor.fetchone() is not None; conn.close()
    return render_template('hospital_detail.html', hospital=h_data, is_bookmarked=is_bookmarked)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('home'))
    if request.method == 'POST':
        hashed = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        try:
            conn = sqlite3.connect(DB_FILE_PATH)
            conn.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                         (request.form['username'], request.form['email'], hashed))
            conn.commit(); conn.close()
            flash('Account created!', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError: flash('User exists.', 'error')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('home'))
    if request.method == 'POST':
        conn = sqlite3.connect(DB_FILE_PATH); cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (request.form['email'],))
        u = cursor.fetchone(); conn.close()
        if u and bcrypt.check_password_hash(u[3], request.form['password']):
            # Handles 4 or 6 column users
            user = User(u[0], u[1], u[2], u[3], u[4] if len(u)>4 else None, u[5] if len(u)>5 else None)
            login_user(user, remember=True)
            session.pop('guest', None)
            return redirect(url_for('home'))
        flash('Login failed.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user(); session.pop('guest', None)
    return redirect(url_for('welcome'))

@app.route('/bookmarks')
@login_required
def bookmarks_page():
    conn = sqlite3.connect(DB_FILE_PATH); cursor = conn.cursor()
    cursor.execute("SELECT hospital_name FROM bookmarks WHERE user_id = ?", (current_user.id,))
    rows = cursor.fetchall(); conn.close()
    hospitals = []
    for r in rows:
        if r[0] in HOSPITAL_INFO_DB:
            info = HOSPITAL_INFO_DB[r[0]].copy(); info['name'] = r[0]
            hospitals.append(info)
    return render_template('bookmarks.html', hospitals=hospitals)

@app.route('/toggle_bookmark', methods=['POST'])
def toggle_bookmark():
    if not current_user.is_authenticated: return jsonify({"error": "Login required"}), 401
    name = request.get_json().get('hospital_name')
    conn = sqlite3.connect(DB_FILE_PATH); cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM bookmarks WHERE user_id = ? AND hospital_name = ?", (current_user.id, name))
    if cursor.fetchone():
        cursor.execute("DELETE FROM bookmarks WHERE user_id = ? AND hospital_name = ?", (current_user.id, name))
        action = "removed"
    else:
        cursor.execute("INSERT INTO bookmarks (user_id, hospital_name) VALUES (?, ?)", (current_user.id, name))
        action = "added"
    conn.commit(); conn.close()
    return jsonify({"status": "success", "action": action})

if __name__ == '__main__':
    app.run(debug=True, port=5000)