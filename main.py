import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

# --- FastAPI & Pydantic ---
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# --- Database (SQLAlchemy) ---
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

# --- Security ---
from passlib.context import CryptContext
from jose import JWTError, jwt

# --- ML / TensorFlow ---
import tensorflow as tf
from tensorflow.keras.models import load_model

from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# 1. KONFIGURASI
# ==========================================

# Database Config
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password123@db_postgres:5432/film_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# JWT Config
SECRET_KEY = "rahasia_sangat_aman"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Model Paths
# Pastikan folder 'model/' dan 'data/' ada di root project
MODEL_PATH = "model/ncf_context_final_model.h5"
ARTIFACTS_PATH = "model/ncf_context_artifacts.joblib"
DATA_DIR = "data" # Folder tempat file .dat disimpan

# Global Variables
ml_model = None
ml_artifacts = None

# Security Setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# ==========================================
# 2. MODEL DATABASE
# ==========================================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    age = Column(Integer)
    occupation = Column(String) 
    # Note: Gender & ZipCode ada di dataset tapi tidak masuk model ini, 
    # bisa ditambahkan jika perlu.
    
class Film(Base):
    __tablename__ = "films"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    genres = Column(String)
    cover_img_url = Column(String, default="https://via.placeholder.com/150")
    
class Rating(Base):
    __tablename__ = "ratings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    film_id = Column(Integer, ForeignKey("films.id"))
    rating = Column(Float)

# Create Tables
Base.metadata.create_all(bind=engine)

# ==========================================
# 3. SEEDER LOGIC (Added Here)
# ==========================================

def seed_from_dat_files(db: Session):
    """
    Membaca file .dat (format MovieLens) dan mengisi database.
    Format file: 'data/users.dat', 'data/movies.dat', 'data/ratings.dat'
    """
    
    # Cek Idempotency: Jika user pertama sudah ada, skip seeding
    if db.query(User).first():
        print("ℹ️  Database already contains data. Skipping seeding.")
        return

    print("🌱 Starting Database Seeding...")
    start_time = time.time()
    
    # Pre-hash password biar cepat (jangan hash di dalam loop)
    default_password_hash = pwd_context.hash("password123")

    # --- A. SEED USERS ---
    users_path = os.path.join(DATA_DIR, "users.dat")
    if os.path.exists(users_path):
        print(f"   ↳ Reading {users_path}...")
        users_buffer = []
        try:
            with open(users_path, "r", encoding="latin-1") as f:
                for line in f:
                    # Format: UserID::Gender::Age::Occupation::Zip-code
                    parts = line.strip().split("::")
                    if len(parts) >= 4:
                        u_id = int(parts[0])
                        users_buffer.append(User(
                            id=u_id,
                            email=f"user{u_id}@test.com", # Generate fake email
                            password=default_password_hash,
                            age=int(parts[2]),
                            occupation=parts[3]
                        ))
            
            db.bulk_save_objects(users_buffer)
            db.commit()
            print(f"   ✅ Inserted {len(users_buffer)} users.")
        except Exception as e:
            print(f"   ❌ Error seeding users: {e}")
            db.rollback()
    else:
        print(f"   ⚠️ File {users_path} not found. Skipping Users.")

    # --- B. SEED FILMS ---
    movies_path = os.path.join(DATA_DIR, "movies.dat")
    if os.path.exists(movies_path):
        print(f"   ↳ Reading {movies_path}...")
        movies_buffer = []
        try:
            with open(movies_path, "r", encoding="latin-1") as f:
                for line in f:
                    # Format: MovieID::Title::Genres
                    parts = line.strip().split("::")
                    if len(parts) >= 3:
                        m_id = int(parts[0])
                        movies_buffer.append(Film(
                            id=m_id,
                            title=parts[1],
                            genres=parts[2],
                            cover_img_url="https://via.placeholder.com/150"
                        ))
            
            db.bulk_save_objects(movies_buffer)
            db.commit()
            print(f"   ✅ Inserted {len(movies_buffer)} films.")
        except Exception as e:
            print(f"   ❌ Error seeding films: {e}")
            db.rollback()
    else:
        print(f"   ⚠️ File {movies_path} not found. Skipping Films.")

    # --- C. SEED RATINGS (Chunked) ---
    ratings_path = os.path.join(DATA_DIR, "ratings.dat")
    if os.path.exists(ratings_path):
        print(f"   ↳ Reading {ratings_path}...")
        ratings_buffer = []
        chunk_size = 10000 # Insert per 10k data biar memori aman
        count = 0
        try:
            with open(ratings_path, "r", encoding="latin-1") as f:
                for line in f:
                    # Format: UserID::MovieID::Rating::Timestamp
                    parts = line.strip().split("::")
                    if len(parts) >= 3:
                        ratings_buffer.append(Rating(
                            user_id=int(parts[0]),
                            film_id=int(parts[1]),
                            rating=float(parts[2])
                        ))
                        
                        if len(ratings_buffer) >= chunk_size:
                            db.bulk_save_objects(ratings_buffer)
                            db.commit()
                            ratings_buffer = [] # Reset buffer
                            print(f"      ... inserted batch {count + 1}")
                            count += 1
            
            # Insert sisa data
            if ratings_buffer:
                db.bulk_save_objects(ratings_buffer)
                db.commit()
            
            print("   ✅ Ratings inserted successfully.")
        except Exception as e:
            print(f"   ❌ Error seeding ratings: {e}")
            db.rollback()
    else:
        print(f"   ⚠️ File {ratings_path} not found. Skipping Ratings.")

    print(f"🏁 Seeding finished in {round(time.time() - start_time, 2)} seconds.")

# ==========================================
# 4. SCHEMAS (Request/Response)
# ==========================================

class LoginRequest(BaseModel):
    email: str
    password: str

class FilmResponse(BaseModel):
    id: int  
    title: str
    cover_img_url: Optional[str]
    genres: List[str] 
    average_rating: float

# ==========================================
# 5. DEPENDENCIES & UTILS
# ==========================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token_obj: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = token_obj.credentials 
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def format_film_response(film_obj, db: Session):
    avg_rating = db.query(func.avg(Rating.rating)).filter(Rating.film_id == film_obj.id).scalar()
    avg_rating = round(avg_rating, 1) if avg_rating else 0.0
    
    genres_list = film_obj.genres.split('|') if film_obj.genres else []
    
    return {
        "id": film_obj.id,
        "title": film_obj.title,
        "cover_img_url": film_obj.cover_img_url,
        "genres": genres_list,
        "average_rating": avg_rating
    }

# ==========================================
# 6. API ENDPOINTS
# ==========================================

app = FastAPI(title="Movie Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan SEMUA origin (domain/ip)
    allow_credentials=True,
    allow_methods=["*"],  # Mengizinkan SEMUA method (GET, POST, PUT, DELETE, dll)
    allow_headers=["*"],  # Mengizinkan SEMUA header (Authorization, Content-Type, dll)
)

@app.on_event("startup")
def startup_event():
    """
    Event ini berjalan saat server menyala:
    1. Load Seeder (isi database jika kosong)
    2. Load ML Model
    """
    # 1. Run Seeder
    db = SessionLocal()
    try:
        seed_from_dat_files(db)
    finally:
        db.close()

    # 2. Load ML Models
    global ml_model, ml_artifacts
    print("⏳ Loading ML Model & Artifacts...")
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ARTIFACTS_PATH):
            ml_model = load_model(MODEL_PATH)
            ml_artifacts = joblib.load(ARTIFACTS_PATH)
            print("✅ Model Loaded Successfully!")
        else:
            print("⚠️ Warning: Model files not found in 'model/' folder.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# --- 1. LOGIN ---
@app.post("/api/v1/login")
def login(creds: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == creds.email).first()
    
    # Validasi user
    if not user or not pwd_context.verify(creds.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email atau password salah",
        )
        
    token = create_access_token(data={"sub": user.email})
    return {"access_token": token, "token_type": "bearer"}

# --- 2. GET RECOMMENDATION (TOP 5) ---
@app.get("/api/v1/recommendation", response_model=List[FilmResponse])
def get_recommendation(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    global ml_model, ml_artifacts
    TOP_K = 5
    
    # Fallback Function
    def get_fallback_films():
        films = db.query(Film).limit(TOP_K).all()
        return [format_film_response(f, db) for f in films]

    if not ml_model or not ml_artifacts:
        return get_fallback_films()

    try:
        # 1. Prepare Encoders
        user_enc = ml_artifacts['user_encoder']
        item_enc = ml_artifacts['item_encoder']
        age_enc = ml_artifacts['age_encoder']
        occ_enc = ml_artifacts['occ_encoder']

        # 2. Encode User Context
        try:
            # Gunakan ID user untuk encode
            u_encoded = user_enc.transform([current_user.id])[0]
            age_encoded = age_enc.transform([current_user.age])[0]
            occ_encoded = occ_enc.transform([current_user.occupation])[0]
        except ValueError:
            print("Cold Start User Detected")
            return get_fallback_films()

        # 3. Generate Input Batch
        all_movie_ids = item_enc.classes_
        n_items = len(all_movie_ids)
        movie_indices = np.arange(n_items)

        user_in = np.array([u_encoded] * n_items)
        age_in = np.array([age_encoded] * n_items)
        occ_in = np.array([occ_encoded] * n_items)
        item_in = movie_indices

        # 4. Predict
        predictions = ml_model.predict([user_in, item_in, age_in, occ_in], verbose=0)
        predictions = predictions.flatten()

        # 5. Get Top K Indices
        top_indices = np.argsort(predictions)[-TOP_K:][::-1]
        
        # 6. Convert Encoded ID -> Real DB ID
        top_enc_ids = movie_indices[top_indices]
        top_real_ids = item_enc.inverse_transform(top_enc_ids)
        top_real_ids_list = [int(x) for x in top_real_ids]

        # 7. Fetch Data from DB & Maintain Order
        films = db.query(Film).filter(Film.id.in_(top_real_ids_list)).all()
        film_map = {f.id: f for f in films}
        
        ordered_results = []
        for fid in top_real_ids_list:
            if fid in film_map:
                ordered_results.append(format_film_response(film_map[fid], db))
            
        return ordered_results

    except Exception as e:
        print(f"Error Recommendation: {e}")
        return get_fallback_films()

# --- 3. GET FILM DETAIL ---
@app.get("/api/v1/film/{film_id}", response_model=FilmResponse)
def get_film_detail(
    film_id: int, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    film = db.query(Film).filter(Film.id == film_id).first()
    
    if not film:
        raise HTTPException(status_code=404, detail="Film tidak ditemukan")
    
    return format_film_response(film, db)