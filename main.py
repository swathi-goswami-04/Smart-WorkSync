from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from datetime import datetime, timedelta
import pyotp
from typing import Optional
import smtplib
from typing import Optional, List, Dict
import pandas as pd
from fpdf import FPDF
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
from fastapi import BackgroundTasks
import os
from dotenv import load_dotenv
from email.message import EmailMessage
from fastapi import Request
from authlib.integrations.starlette_client import OAuth
import shutil
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import hashlib
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, ForeignKey, Float, JSON
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Date, Boolean  # Add missing imports
from sqlalchemy.orm import Session
from fastapi import Depends
from time_tracking import SessionLocal, Base  
from time_tracking import User, BaseModel # Import User model
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
import random
from time_tracking import UserResponse, TimeEntryResponse, TaskResponse, AttendanceResponse, LeaveRequestResponse, AttendanceChangeRequestResponse, AlertResponse, ReportRequest, PredictionRequest
from time_tracking import Base, engine, UserCreate, UserUpdate, UserRegister, ClockInRequest, ClockOutRequest, StartBreakRequest, EndBreakRequest, TimeRecordResponse, TaskUpdateRequest, LogTimeRequest, ProjectResponse, TaskAssignRequest # Import your models and database engine
import time
from passlib.context import CryptContext

# ✅ This ensures all tables are created
Base.metadata.create_all(bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user(username: str, db: Session):
    return db.query(User).filter(User.username == username).first()
    

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Secret key for JWT
token_secret_key = "O7FKHXU3OHft-r6CeWE2bcjcW1wXsbrA63XC_W4Nlkk"
algorithm = "HS256"
access_token_expire_minutes = 30
refresh_token_expire_days = 7


load_dotenv()

MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_FROM = os.getenv("MAIL_FROM")
MAIL_PORT = int(os.getenv("MAIL_PORT", 587))
MAIL_SERVER = os.getenv("MAIL_SERVER")
MAIL_STARTTLS = os.getenv("MAIL_STARTTLS") == "True"
MAIL_SSL_TLS = os.getenv("MAIL_SSL_TLS") == "True"
USE_CREDENTIALS = os.getenv("USE_CREDENTIALS") == "True"

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    authorize_params=None,
    access_token_url="https://oauth2.googleapis.com/token",
    access_token_params=None,
    redirect_uri="http://localhost:8000/auth/google/callback",
    client_kwargs={"scope": "openid email profile"},
)

              
    
def hash_password(password: str):
    return password  

def insert_csv_data():
    df = pd.read_csv("demo_users.csv")
    session = SessionLocal()
   
    try:
        for _, row in df.iterrows():
            if not session.query(User).filter(User.username == row["Username"]).first():
                otp_secret = pyotp.random_base32()
                user = User(
                    username=row["Username"],
                    email=row["Email"],
                    password=hash_password(row["Password (SHA-256 Hashed)"]),
                    role=row["Role"],
                    otp_secret=otp_secret,
                    userid=row["User_ID"],
                    ph_no=row["Phone_number"],
                    verify=row["Is_Verified"],
                    joining=row["Date_Joined"],
                    lastseen=row["Last_Login"],
                    status=row["Status"]
                )
                session.add(user)

        session.commit()
        print("✅ CSV Data Imported Successfully!")

    except Exception as e:
        session.rollback()
        print(f"❌ Error: {e}")

    finally:
        session.close()


class TimeEntryResponse(BaseModel):
    id: int
    user_id: int
    task_id: int  # ✅ Ensure this field exists
    clock_in: datetime
    clock_out: datetime | None
    break_time: int  # Example field for break duration


# OAuth2 Setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=access_token_expire_minutes))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, token_secret_key, algorithm=algorithm)
def send_otp_email(recipient_email, otp_code):
    msg = EmailMessage()
    msg.set_content(f"Your OTP code is: {otp_code}")
    msg["Subject"] = "Your OTP Code"
    msg["From"] = MAIL_FROM
    msg["To"] = recipient_email

    try:
        with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as server:
            if MAIL_STARTTLS:
                server.starttls()
            if USE_CREDENTIALS:
                server.login(MAIL_USERNAME, MAIL_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)  # Hashed password
    role = Column(String, nullable=False)
    userid = Column(Integer, unique=True, nullable=False)
    ph_no = Column(String, index=True)  # Phone number should be String
    verify = Column(Boolean, default=False)
    joining = Column(DateTime, index=True)
    lastseen = Column(DateTime, index=True)
    status = Column(String, nullable=False)
    otp_secret = Column(String, nullable=False)
     
    
class UserResponse(BaseModel):
    username: str
    email: str
    role: str
    userid: int
    ph_no: Optional[str]
    verify: bool
    joining: Optional[datetime]
    lastseen: Optional[datetime]
    status: str

    class Config:
        from_attributes = True
def send_otp_email(recipient_email: str, otp_code: str):
    """ Sends OTP to user's email using SMTP """
    msg = EmailMessage()
    msg.set_content(f"Your OTP code is: {otp_code}")
    msg["Subject"] = "Your OTP Code"
    msg["From"] = "swathigoswami98@gmail.com"  # Update with your SMTP email
    msg["To"] = recipient_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)  # Replace with actual SMTP server
        server.starttls()
        server.login("swathigoswami98@gmail.com", "eywx hwdk llgf kbqu")  # Replace with SMTP credentials
        server.send_message(msg)
        server.quit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")

otp_cache = {}

def generate_otp():
    return str(random.randint(100000, 999999))  # Generates a 6-digit OTP

@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """ Authenticate user and send OTP via email """
    user = db.query(User).filter(User.username == form_data.username).first()

    if not user or user.password != form_data.password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    otp = generate_otp()  # Generate OTP
    otp_cache[user.username] = {"otp": otp, "expires_at": time.time() + 300}  # Store OTP (expires in 5 minutes)

    send_otp_email(user.email, otp)  # Send OTP via SMTP

    return {"otp_required": True, "otp_sent": True, "message": "OTP sent to registered email"}

class UserResponse(BaseModel):
    username: str
    email: str
    role: str
    lastseen: str

@app.get("/users/{username}")
def get_user_by_username(username: str, db: Session = Depends(get_db)):
    """Fetch user details by username"""
    user = db.query(User).filter(User.username == username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return {
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "lastseen": user.lastseen
    }
@app.get("/auth/google")
async def google_login(request: Request):
    redirect_uri = request.url_for("google_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/google/callback")
async def google_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = await oauth.google.parse_id_token(request, token)
        return {"message": "Google login successful", "user": user_info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Google authentication failed: {str(e)}")
class OTPVerificationRequest(BaseModel):
    username: str
    otp: str

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=access_token_expire_minutes))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, token_secret_key, algorithm=algorithm)

@app.post("/verify-otp")
def verify_otp(request: OTPVerificationRequest, db: Session = Depends(get_db)):
    """ Verifies OTP and issues access token """
    user = db.query(User).filter(User.username == request.username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    if request.username not in otp_cache:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No OTP found. Please request a new one.")

    stored_otp_info = otp_cache[request.username]

    if time.time() > stored_otp_info["expires_at"]:
        del otp_cache[request.username]  # Remove expired OTP
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OTP expired. Request a new one.")

    if stored_otp_info["otp"] != request.otp:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid OTP")

    # ✅ OTP is valid, generate access token
    access_token = create_access_token({"sub": user.username, "role": user.role})

    del otp_cache[request.username]  # Remove OTP after successful verification

    return {
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "redirect_url": "/dashboard"  # ✅ Tell frontend where to go next
    }

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, token_secret_key, algorithms=[algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        user = db.query(User).filter(User.username == username).first()
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

@app.get("/protected-route")
def protected_route(user: User = Depends(get_current_user)):
    return {"message": "Access granted", "user": user.username}



@app.get("/admin")
def admin_route(user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user["username"]).first()
    if not db_user or db_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    return {"message": "Welcome, Admin!"}



@app.get("/admin-dashboard")
def get_admin_dashboard():
    def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
        user = get_user(form_data.username)
        users_count = len(user)
        projects_count = len(project_db)
        attendance_count = len(attendance_records)
        return {"users": users_count, "projects": projects_count, "attendance": attendance_count}

# Google OAuth2 Placeholder (To be implemented with Authlib)
@app.get("/oauth/google")
def google_oauth():
    return {"message": "Google OAuth2 SSO will be implemented here"}

# Logout and Token Blacklist Placeholder

@app.get("/users/{username}")
def get_user(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

class UserResponse(BaseModel):
    username: str
    email: str
    role: str

    class Config:
        from_attributes = True  # Ensures conversion from SQLAlchemy model
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@app.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user"""
    existing_user = db.query(User).filter(User.username == user.username).first()

    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")

    hashed_password = pwd_context.hash(user.password)  # Hash the password before storing

    new_user = User(
        username=user.username,
        email=user.email,
        password=hashed_password,  # Store hashed password
        role=user.role,
        userid=user.userid,
        ph_no=user.ph_no,
        verify=user.verify,
        joining=user.joining,
        lastseen=user.lastseen,
        status=user.status
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": "User created successfully",
        "username": new_user.username,
        "email": new_user.email,
        "role": new_user.role
    }

@app.put("/users/{username}")
def update_user(username: str, user_update: UserUpdate, db: Session = Depends(get_db)):
    """Update user details by username"""
    user = db.query(User).filter(User.username == username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if user_update.email:
        user.email = user_update.email
    if user_update.role:
        user.role = user_update.role
    if user_update.ph_no:
        user.ph_no = user_update.ph_no
    if user_update.verify is not None:
        user.verify = user_update.verify
    if user_update.joining:
        user.joining = user_update.joining
    if user_update.lastseen:
        user.lastseen = user_update.lastseen
    if user_update.status:
        user.status = user_update.status

    db.commit()
    db.refresh(user)

    return {"message": "User updated successfully"}

@app.delete("/users/{username}")
def delete_user(username: str, db: Session = Depends(get_db)):
    """Delete a user by username"""
    user = db.query(User).filter(User.username == username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    db.delete(user)
    db.commit()

    return {"message": f"User '{username}' deleted successfully"}
blacklisted_tokens = set()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/logout")
def logout(token: str = Depends(oauth2_scheme)):
    """Logout user by invalidating the token (handled on frontend)"""
    return {"message": "Logged out successfully"}

@app.get("/check-token")
def check_token(token: str = Depends(oauth2_scheme)):
    if token in blacklisted_tokens:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked")
    return {"message": "Token is valid"}

@app.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(user: UserRegister, db: Session = Depends(get_db)):
    """Register a new user"""
    existing_user = db.query(User).filter(User.username == user.username).first()

    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")

    hashed_password = pwd_context.hash(user.password)  # Hash the password before storing

    new_user = User(
        username=user.username,
        email=user.email,
        password=hashed_password,  # Store hashed password
        role=user.role,
        ph_no=user.ph_no,
        joining=user.joining,
        verify=False,  # Default to not verified
        lastseen=None,
        status="pending"  # Set default status
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": "User registered successfully",
        "username": new_user.username,
        "email": new_user.email,
        "role": new_user.role
    }

class TimeEntryDB(Base):
    __tablename__ = "time_entries"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, ForeignKey("users.username"), nullable=False)
    clock_in = Column(DateTime, nullable=True)
    clock_out = Column(DateTime, nullable=True)
    total_hours = Column(Integer, default=0)
    location = Column(String, nullable=True)
    approved = Column(Boolean, default=False)
  # Ensure table creation

# Pydantic Model for API Input
class TimeEntry(BaseModel):
    username: str
    clock_in: Optional[datetime] = None
    clock_out: Optional[datetime] = None
    breaks: List[dict] = []  # [{"start": datetime, "end": datetime}]
    total_hours: float = 0.0
    location: Optional[str] = None
    approved: bool = False  # Manual entries require approval

# Function to calculate total hours worked
def calculate_total_hours(entry: TimeEntry):
    if entry.clock_in and entry.clock_out:
        total_seconds = (entry.clock_out - entry.clock_in).total_seconds()
        break_seconds = sum([(b["end"] - b["start"]).total_seconds() for b in entry.breaks if b["end"]])
        return (total_seconds - break_seconds) / 3600
    return 0.0

# Clock-in API
@app.post("/clock-in", status_code=status.HTTP_201_CREATED)
def clock_in(clock_in_data: ClockInRequest, db: Session = Depends(get_db)):
    """Clock-in for a task"""
    user = db.query(User).filter(User.username == clock_in_data.username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    existing_entry = db.query(TimeEntryDB).filter(
        TimeEntryDB.username == clock_in_data.username,
        TimeEntryDB.task_id == clock_in_data.task_id,
        TimeEntryDB.clock_out == None  # User must not have an active session
    ).first()

    if existing_entry:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Already clocked in for this task")

    new_entry = TimeEntryDB(
        username=clock_in_data.username,
        task_id=clock_in_data.task_id,
        clock_in=datetime.utcnow(),
        location=clock_in_data.location,
        approved=False  # By default, clock-in needs approval
    )

    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)

    return {
        "message": "Clock-in successful",
        "username": new_entry.username,
        "task_id": new_entry.task_id,
        "clock_in": new_entry.clock_in,
        "location": new_entry.location
    }

@app.put("/clock-out")
def clock_out(clock_out_data: ClockOutRequest, db: Session = Depends(get_db)):
    """Clock-out from a task"""
    user = db.query(User).filter(User.username == clock_out_data.username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    time_entry = db.query(TimeEntryDB).filter(
        TimeEntryDB.username == clock_out_data.username,
        TimeEntryDB.task_id == clock_out_data.task_id,
        TimeEntryDB.clock_out == None  # Must be an active clock-in session
    ).first()

    if not time_entry:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No active clock-in found for this task")

    time_entry.clock_out = datetime.utcnow()

    # Calculate total hours worked
    time_entry.total_hours = (time_entry.clock_out - time_entry.clock_in).total_seconds() / 3600

    db.commit()
    db.refresh(time_entry)

    return {
        "message": "Clock-out successful",
        "username": time_entry.username,
        "task_id": time_entry.task_id,
        "clock_in": time_entry.clock_in,
        "clock_out": time_entry.clock_out,
        "total_hours": time_entry.total_hours
    }

@app.post("/break-start")
def start_break(start_break_data: StartBreakRequest, db: Session = Depends(get_db)):
    """Start a break for a task"""
    user = db.query(User).filter(User.username == start_break_data.username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    time_entry = db.query(TimeEntryDB).filter(
        TimeEntryDB.username == start_break_data.username,
        TimeEntryDB.task_id == start_break_data.task_id,
        TimeEntryDB.clock_out == None  # Must be an active clock-in session
    ).first()

    if not time_entry:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No active clock-in found for this task")

    if time_entry.break_start:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Break already started for this task")

    time_entry.break_start = datetime.utcnow()

    db.commit()
    db.refresh(time_entry)

    return {
        "message": "Break started successfully",
        "username": time_entry.username,
        "task_id": time_entry.task_id,
        "break_start": time_entry.break_start
    }

@app.put("/break-end")
def end_break(end_break_data: EndBreakRequest, db: Session = Depends(get_db)):
    """End a break for a task"""
    user = db.query(User).filter(User.username == end_break_data.username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    time_entry = db.query(TimeEntryDB).filter(
        TimeEntryDB.username == end_break_data.username,
        TimeEntryDB.task_id == end_break_data.task_id,
        TimeEntryDB.clock_out == None,  # Must be an active clock-in session
        TimeEntryDB.break_start != None  # Must have started a break
    ).first()

    if not time_entry:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No active break found for this task")

    time_entry.break_end = datetime.utcnow()

    # Calculate total break time in minutes
    break_duration = (time_entry.break_end - time_entry.break_start).total_seconds() / 60
    time_entry.total_break_time = (time_entry.total_break_time or 0) + break_duration

    # Reset break start time
    time_entry.break_start = None

    db.commit()
    db.refresh(time_entry)

    return {
        "message": "Break ended successfully",
        "username": time_entry.username,
        "task_id": time_entry.task_id,
        "break_end": time_entry.break_end,
        "total_break_time": time_entry.total_break_time
    }


# Get All Time Records API
@app.get("/time-records")
def get_time_records(db: Session = Depends(get_db)):
    """Fetch all time records"""
    records = db.query(TimeEntryDB).all()

    return [
        {
            "id": record.id,
            "username": record.username,
            "clock_in": record.clock_in.strftime("%Y-%m-%d %H:%M:%S"),
            "clock_out": record.clock_out.strftime("%Y-%m-%d %H:%M:%S") if record.clock_out else None,
            "total_hours": record.total_hours
            # ❌ Removed "task_id": record.task_id
        }
        for record in records
    ]

# ✅ SQLAlchemy Model for Projects
class ProjectDB(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    assigned_to = Column(String, nullable=False)  # Comma-separated usernames

    tasks = relationship("TaskDB", back_populates="project", cascade="all, delete")

# ✅ SQLAlchemy Model for Tasks
class TaskDB(Base):
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String, nullable=False)
    status = Column(String, default="To Do")
    time_logged = Column(Float, default=0.0)
    
    project = relationship("ProjectDB", back_populates="tasks")



# ✅ Pydantic Model for API Responses
class TaskResponse(BaseModel):
    id: int
    name: str
    status: str
    time_logged: float

    class Config:
        from_attributes = True

class ProjectResponse(BaseModel):
    id: int
    name: str
    description: str
    assigned_to: List[str]  # Convert comma-separated string to list
    tasks: List[TaskResponse] = []

    class Config:
        from_attributes = True

# ✅ Dependency for Database Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
from time_tracking import ProjectDB, get_db, ProjectCreate


@app.post("/assign-task", status_code=status.HTTP_201_CREATED)
def assign_task(task_data: TaskAssignRequest, db: Session = Depends(get_db)):
    """Assign a task to a user"""
    user = db.query(User).filter(User.username == task_data.username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    new_task = TaskDB(
        task_name=task_data.name,
        project_id=task_data.project_id,
        task_description=task_data.task_description,
        deadline=task_data.deadline,
        status=task_data.status
    )

    db.add(new_task)
    db.commit()
    db.refresh(new_task)

    return {
        "message": "Task assigned successfully",
        "task_id": new_task.id,
        "username": new_task.username,
        "project_id": new_task.project_id,
        "task_name": new_task.task_name,
        "task_description": new_task.task_description,
        "deadline": new_task.deadline,
        "status": new_task.status
    }


from time_tracking import TaskDB, get_db, TaskUpdateRequest

@app.put("/update-task/{task_id}")
def update_task(task_id: int, task_data: TaskUpdateRequest, db: Session = Depends(get_db)):
    """Update an existing task"""
    task = db.query(TaskDB).filter(TaskDB.id == task_id).first()

    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    if task_data.task_name:
        task.task_name = task_data.task_name
    if task_data.task_description:
        task.task_description = task_data.task_description
    if task_data.deadline:
        task.deadline = task_data.deadline
    if task_data.status:
        task.status = task_data.status

    db.commit()
    db.refresh(task)

    return {
        "message": "Task updated successfully",
        "task_id": task.id,
        "task_name": task.task_name,
        "task_description": task.task_description,
        "deadline": task.deadline,
        "status": task.status
    }

# ✅ Delete a Task
@app.delete("/delete-task/{task_id}")
def delete_task(task_id: int, db: Session = Depends(get_db)):
    """Delete a task by ID"""
    task = db.query(TaskDB).filter(TaskDB.id == task_id).first()

    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    db.delete(task)
    db.commit()

    return {"message": f"Task '{task_id}' deleted successfully"}

@app.post("/log-time", status_code=status.HTTP_201_CREATED)
def log_time(entry_data: LogTimeRequest, db: Session = Depends(get_db)):
    """Manually log time for a task"""
    user = db.query(User).filter(User.username == entry_data.username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if entry_data.clock_out <= entry_data.clock_in:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Clock-out time must be after clock-in time")

    total_hours = (entry_data.clock_out - entry_data.clock_in).total_seconds() / 3600

    new_entry = TimeEntryDB(
        username=entry_data.username,
        task_id=entry_data.task_id,
        clock_in=entry_data.clock_in,
        clock_out=entry_data.clock_out,
        total_hours=total_hours,
        location=entry_data.location,
        approved=True  # ✅ Manual logs are auto-approved
    )

    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)

    return {
        "message": "Time logged successfully",
        "username": new_entry.username,
        "task_id": new_entry.task_id,
        "clock_in": new_entry.clock_in,
        "clock_out": new_entry.clock_out,
        "total_hours": new_entry.total_hours,
        "location": new_entry.location
    }
@app.post("/projects", status_code=status.HTTP_201_CREATED)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    """Create a new project"""
    existing_project = db.query(ProjectDB).filter(ProjectDB.name == project.name).first()

    if existing_project:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Project already exists")

    new_project = ProjectDB(
        name=project.name,
        description=project.description,
        start_date=project.start_date,
        end_date=project.end_date,
        status=project.status
    )

    db.add(new_project)
    db.commit()
    db.refresh(new_project)

    return {
        "message": "Project created successfully",
        "project_id": new_project.id,
        "name": new_project.name,
        "description": new_project.description,
        "start_date": new_project.start_date,
        "end_date": new_project.end_date,
        "status": new_project.status
    }



@app.get("/project-details/{project_id}", response_model=ProjectResponse)
def get_project_by_id(project_id: int, db: Session = Depends(get_db)):
    """Fetch a project by ID"""
    project = db.query(ProjectDB).filter(ProjectDB.id == project_id).first()

    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    # ✅ Ensure assigned_to is a list
    assigned_users = [user.username for user in project.assigned_users] if hasattr(project, 'assigned_users') else []

    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "start_date": project.start_date,
        "end_date": project.end_date,
        "status": project.status,
        "assigned_to": assigned_users  # ✅ Return a list, not a single string
    }   

@app.delete("/delete-project/{project_id}", status_code=status.HTTP_200_OK)
def delete_project(project_id: int, db: Session = Depends(get_db)):
    """Delete a project by ID"""
    project = db.query(ProjectDB).filter(ProjectDB.id == project_id).first()

    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    db.delete(project)
    db.commit()

    return {
        "message": f"Project '{project_id}' deleted successfully",
        "deleted_project": {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "start_date": project.start_date,
            "end_date": project.end_date,
            "status": project.status
        }
    }
# ✅ SQLAlchemy Model for Attendance
class AttendanceDB(Base):
    __tablename__ = "attendance"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String, ForeignKey("users.username"), nullable=False)
    date = Column(DateTime, default=datetime.utcnow)
    status = Column(String, nullable=False)  # Present, Absent, Late
    check_in = Column(DateTime, nullable=True)
    check_out = Column(DateTime, nullable=True)
    approved = Column(Boolean, default=False)

# ✅ SQLAlchemy Model for Leave Requests
class LeaveRequestDB(Base):
    __tablename__ = "leave_requests"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String, ForeignKey("users.username"), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    reason = Column(String, nullable=False)
    status = Column(String, default="Pending")  # Pending, Approved, Rejected



# ✅ Pydantic Models for API Responses
class AttendanceResponse(BaseModel):
    username: str
    date: datetime
    status: str
    check_in: Optional[datetime]
    check_out: Optional[datetime]
    approved: bool
    
    class Config:
        from_attributes = True

class LeaveRequestResponse(BaseModel):
    username: str
    start_date: datetime
    end_date: datetime
    reason: str
    status: str
    
    class Config:
        from_attributes = True


# ✅ Mark Attendance
@app.post("/mark-attendance", response_model=AttendanceResponse, status_code=status.HTTP_201_CREATED)
def mark_attendance(username: str, status: str, db: Session = Depends(get_db)):
    """Mark attendance for a user"""
    user = db.query(User).filter(User.username == username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    today = datetime.utcnow().date()
    
    existing_attendance = db.query(AttendanceDB).filter(
        AttendanceDB.username == username,
        AttendanceDB.date == today
    ).first()

    if existing_attendance:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Attendance already marked for today")

    new_attendance = AttendanceDB(
        username=username,
        date=today,
        status=status,
        check_in=datetime.utcnow() if status.lower() == "present" else None,
        check_out=None,
        approved=False  # Attendance approval pending
    )

    db.add(new_attendance)
    db.commit()
    db.refresh(new_attendance)

    return new_attendance
@app.get("/attendance/{username}", response_model=List[AttendanceResponse])
def get_attendance(username: str, db: Session = Depends(get_db)):
    """Fetch attendance records for a user"""
    user = db.query(User).filter(User.username == username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    attendance_records = db.query(AttendanceDB).filter(AttendanceDB.username == username).all()

    if not attendance_records:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No attendance records found")

    return attendance_records
@app.post("/request-leave", response_model=LeaveRequestResponse, status_code=status.HTTP_201_CREATED)
def request_leave(
    username: str, start_date: datetime, end_date: datetime, reason: str, db: Session = Depends(get_db)
):
    """Allow users to request leave"""
    user = db.query(User).filter(User.username == username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if start_date >= end_date:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="End date must be after start date")

    new_leave_request = LeaveRequestDB(
        username=username,
        start_date=start_date,
        end_date=end_date,
        reason=reason,
        status="Pending"  # Default status
    )

    db.add(new_leave_request)
    db.commit()
    db.refresh(new_leave_request)

    return new_leave_request


@app.get("/leaves/{username}", response_model=List[LeaveRequestResponse])
def get_leaves(username: str, db: Session = Depends(get_db)):
    """Fetch leave requests for a user"""
    user = db.query(User).filter(User.username == username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    leave_requests = db.query(LeaveRequestDB).filter(LeaveRequestDB.username == username).all()

    if not leave_requests:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No leave requests found")

    return leave_requests
@app.put("/approve-leave/{leave_id}", response_model=LeaveRequestResponse)
def approve_leave(leave_id: int, status: str, db: Session = Depends(get_db)):
    """Approve or reject a leave request"""
    leave_request = db.query(LeaveRequestDB).filter(LeaveRequestDB.id == leave_id).first()

    if not leave_request:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Leave request not found")

    if status not in ["Approved", "Rejected"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid status. Use 'Approved' or 'Rejected'.")

    leave_request.status = status
    db.commit()
    db.refresh(leave_request)

    return leave_request


@app.get("/attendance-report/{username}", response_model=List[AttendanceResponse])
def get_attendance_report(username: str, db: Session = Depends(get_db)):
    """Fetch attendance report for a user"""
    user = db.query(User).filter(User.username == username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    attendance_records = db.query(AttendanceDB).filter(AttendanceDB.username == username).all()

    if not attendance_records:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No attendance records found")

    return attendance_records


# ✅ SQLAlchemy Model for Attendance Change Requests
class AttendanceChangeRequestDB(Base):
    __tablename__ = "attendance_change_requests"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String, ForeignKey("users.username"), nullable=False)
    date = Column(DateTime, nullable=False)
    status = Column(String, nullable=False)  # Present, Absent, Late
    approved = Column(Boolean, default=None)  # None = Pending, True = Approved, False = Rejected



# ✅ Pydantic Model for API Responses
class AttendanceChangeRequestResponse(BaseModel):
    username: str
    date: datetime
    status: str
    approved: Optional[bool]
    
    class Config:
        from_attributes = True

# ✅ Request Attendance Change
@app.post("/request-attendance-change", response_model=AttendanceChangeRequestResponse, status_code=status.HTTP_201_CREATED)
def request_attendance_change(username: str, date: datetime, new_status: str, db: Session = Depends(get_db)):
    """Allow users to request an attendance status change"""
    attendance_record = db.query(AttendanceDB).filter(
        AttendanceDB.username == username,
        AttendanceDB.date == date
    ).first()

    if not attendance_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attendance record not found")

    if attendance_record.status == new_status:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No change in attendance status requested")

    # Create a pending change request
    attendance_record.status = new_status
    attendance_record.approved = False  # Mark as pending approval

    db.commit()
    db.refresh(attendance_record)

    return attendance_record

@app.get("/attendance-requests", response_model=List[AttendanceChangeRequestResponse])
def get_attendance_requests(db: Session = Depends(get_db)):
    """Fetch all pending attendance change requests"""
    requests = db.query(AttendanceDB).filter(AttendanceDB.approved == False).all()

    if not requests:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No pending attendance change requests found")

    return requests
# ✅ Approve or Reject Attendance Change Requests
@app.post("/approve-attendance")
def approve_attendance(username: str, date: datetime, approved: bool, db: Session = Depends(get_db)):
    request = db.query(AttendanceChangeRequestDB).filter(
        AttendanceChangeRequestDB.username == username,
        AttendanceChangeRequestDB.date == date
    ).first()
    
    if not request:
        raise HTTPException(status_code=404, detail="Attendance request not found")
    
    request.approved = approved
    db.commit()
    
    if approved:
        attendance_entry = AttendanceDB(
            username=request.username,
            date=request.date,
            status=request.status,
            approved=True
        )
        db.add(attendance_entry)
        db.commit()
    
    return {"message": "Attendance request updated", "request": request}


# ✅ SQLAlchemy Model for Alerts
class AlertDB(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String, ForeignKey("users.username"), nullable=False)
    type = Column(String, nullable=False)  # Clock-in, Clock-out, Break, Overtime
    message = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

# ✅ Ensure Table is Created
Base.metadata.create_all(bind=SessionLocal().bind)

# ✅ Pydantic Model for API Responses
class AlertResponse(BaseModel):
    username: str
    type: str
    message: str
    timestamp: datetime

    class Config:
        from_attributes = True

# ✅ Set Alert
@app.post("/set-alert", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
def set_alert(username: str, alert_type: str, message: str, db: Session = Depends(get_db)):
    """Set an alert for a user"""
    user = db.query(User).filter(User.username == username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    new_alert = AlertDB(
        username=username,
        type=alert_type,
        message=message,
        timestamp=datetime.utcnow()
    )

    db.add(new_alert)
    db.commit()
    db.refresh(new_alert)

    return new_alert
@app.get("/alerts/{username}", response_model=List[AlertResponse])
def get_alerts(username: str, db: Session = Depends(get_db)):
    """Fetch all alerts for a user"""
    user = db.query(User).filter(User.username == username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    alerts = db.query(AlertDB).filter(AlertDB.username == username).all()

    if not alerts:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No alerts found")

    return alerts
# ✅ Send Email Alert
def send_email_alert(email: str, message: str):
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login("swathigoswami98@gmail.com", "eywx hwdk llgf kbqu")
        server.sendmail("swathigoswami98@gmail.com", email, message)
        server.quit()
    except Exception as e:
        print("Error sending email:", e)

@app.post("/trigger-alert", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
def trigger_alert(
    username: str,
    alert_type: str,
    message: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Send an immediate alert to a user and trigger an email notification"""
    user = db.query(User).filter(User.username == username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    new_alert = AlertDB(
        username=username,
        type=alert_type,
        message=message,
        timestamp=datetime.utcnow()
    )

    db.add(new_alert)
    db.commit()
    db.refresh(new_alert)

    # ✅ Trigger an email notification in the background
    background_tasks.add_task(send_email_alert, "nehagoswami226@gmail.com", message)

    return new_alert
# ✅ SQLAlchemy Model for Reports
class ReportDB(Base):
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String, ForeignKey("users.username"), nullable=True)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    report_type = Column(String, nullable=False)  # work_hours, project_allocation, overtime
    file_path = Column(String, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)

# ✅ Ensure Table is Created


# ✅ Pydantic Model for Report Requests
class ReportRequest(BaseModel):
    username: Optional[str] = None
    start_date: datetime
    end_date: datetime
    report_type: str  # work_hours, project_allocation, overtime


from fastapi import Depends, HTTPException, Query

@app.get("/generate-report")
def generate_report(
    username: str,
    report_type: str = Query(..., description="Type of report: work_hours, project_allocation, overtime"),
    start_date: datetime = Query(None, description="Start date (optional)"),
    end_date: datetime = Query(None, description="End date (optional)"),
    db: Session = Depends(get_db)
):
    data = []

    if report_type == "work_hours":
        if not start_date or not end_date:
            raise HTTPException(status_code=400, detail="Start date and end date are required for work_hours report")
        data = db.query(TimeEntryDB).filter(
            TimeEntryDB.clock_in >= start_date,
            TimeEntryDB.clock_out <= end_date
        ).all()

    elif report_type == "project_allocation":
        data = db.query(TaskDB).filter(TaskDB.project_id == ProjectDB.id).all()

    elif report_type == "overtime":
        if not start_date or not end_date:
            raise HTTPException(status_code=400, detail="Start date and end date are required for overtime report")
        data = db.query(TimeEntryDB).filter(
            TimeEntryDB.total_hours > 8,
            TimeEntryDB.clock_in >= start_date,
            TimeEntryDB.clock_out <= end_date
        ).all()

    else:
        raise HTTPException(status_code=400, detail="Invalid report type")

    if not data:
        raise HTTPException(status_code=404, detail="No data found for the report")

    df = pd.DataFrame([d.__dict__ for d in data])
    report_filename = f"reports/{report_type}_report.xlsx"
    os.makedirs("reports", exist_ok=True)
    df.to_excel(report_filename, index=False)

    new_report = ReportDB(
        username=username,
        start_date=start_date,
        end_date=end_date,
        report_type=report_type,
        file_path=report_filename
    )
    db.add(new_report)
    db.commit()

    return {"message": "Report generated successfully", "file": report_filename}

import pdfkit


@app.get("/export-report-pdf")
def export_report_pdf(
    username: str,
    report_type: str = Query(..., description="Type of report: work_hours, project_allocation, overtime"),
    start_date: datetime = Query(None, description="Start date (optional)"),
    end_date: datetime = Query(None, description="End date (optional)"),
    db: Session = Depends(get_db)
):
    data = []

    if report_type == "work_hours":
        if not start_date or not end_date:
            raise HTTPException(status_code=400, detail="Start date and end date are required for work_hours report")
        data = db.query(TimeEntryDB).filter(
            TimeEntryDB.clock_in >= start_date,
            TimeEntryDB.clock_out <= end_date
        ).all()

    elif report_type == "project_allocation":
        data = db.query(TaskDB).filter(TaskDB.project_id == ProjectDB.id).all()

    elif report_type == "overtime":
        if not start_date or not end_date:
            raise HTTPException(status_code=400, detail="Start date and end date are required for overtime report")
        data = db.query(TimeEntryDB).filter(
            TimeEntryDB.total_hours > 8,
            TimeEntryDB.clock_in >= start_date,
            TimeEntryDB.clock_out <= end_date
        ).all()

    else:
        raise HTTPException(status_code=400, detail="Invalid report type")

    if not data:
        raise HTTPException(status_code=404, detail="No data found for the report")

    # Convert Data to HTML
    df = pd.DataFrame([d.__dict__ for d in data])
    html_content = df.to_html()

    # Create PDF File
    report_filename = f"reports/{report_type}_report.pdf"
    os.makedirs("reports", exist_ok=True)
    pdfkit.from_string(html_content, report_filename)

    # Save Report Record
    new_report = ReportDB(
        username=username,
        start_date=start_date,
        end_date=end_date,
        report_type=report_type,
        file_path=report_filename
    )
    db.add(new_report)
    db.commit()

    return {"message": "Report exported as PDF successfully", "file": report_filename}



# ✅ SQLAlchemy Model for Workload Predictions
class PredictionDB(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String, ForeignKey("users.username"), nullable=True)
    prediction_type = Column(String, nullable=False)  # workload, burnout, project_delay
    predicted_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)



# ✅ Pydantic Model for Prediction Requests
class PredictionRequest(BaseModel):
    username: Optional[str] = None
    prediction_type: str  # workload, burnout, project_delay
    future_days: int

# ✅ Dependency for Database Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


import matplotlib.pyplot as plt
import io
import base64

@app.get("/predict-workload-graph")
def predict_workload_graph(
    username: str,
    days_ahead: int = Query(7, description="Number of days ahead to predict workload"),
    db: Session = Depends(get_db)
):
    """Predict workload and return a graph as an image"""
    user_time_entries = db.query(TimeEntryDB).filter(TimeEntryDB.username == username).all()

    if not user_time_entries:
        raise HTTPException(status_code=404, detail="No historical work hours found for the user")

    # Extract work hours and dates
    dates = [entry.clock_in.timestamp() for entry in user_time_entries]
    hours = [entry.total_hours for entry in user_time_entries]

    # Convert to NumPy arrays
    X = np.array(dates).reshape(-1, 1)
    y = np.array(hours)

    # Train a simple Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict workload for the next `days_ahead`
    future_dates = np.array([(datetime.utcnow() + timedelta(days=i)).timestamp() for i in range(1, days_ahead + 1)]).reshape(-1, 1)
    predicted_hours = model.predict(future_dates)

    # Convert future dates to human-readable format
    future_dates_formatted = [(datetime.utcnow() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, days_ahead + 1)]

    # Plot graph
    plt.figure(figsize=(8, 5))
    plt.plot(future_dates_formatted, predicted_hours, marker="o", linestyle="-", color="b", label="Predicted Workload")
    plt.xlabel("Date")
    plt.ylabel("Predicted Work Hours")
    plt.title(f"Predicted Workload for {username}")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()

    # Save the plot as an image in memory
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="png")
    plt.close()
    img_bytes.seek(0)

    # Convert image to base64
    img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

    return {"username": username, "image": f"data:image/png;base64,{img_base64}"}

@app.get("/detect-burnout-graph")
def detect_burnout_graph(db: Session = Depends(get_db)):
    """Generate a bar chart for employees at risk of burnout (>50 hours/week)"""
    
    one_week_ago = datetime.utcnow() - timedelta(days=7)

    burnout_risks = (
        db.query(TimeEntryDB.username, func.sum(TimeEntryDB.total_hours).label("total_hours"))
        .filter(TimeEntryDB.clock_in >= one_week_ago)
        .group_by(TimeEntryDB.username)
        .having(func.sum(TimeEntryDB.total_hours) > 50)  # Employees working >50 hours/week
        .all()
    )

    if not burnout_risks:
        return {"message": "No employees detected at risk of burnout"}

    usernames = [record.username for record in burnout_risks]
    total_hours = [record.total_hours for record in burnout_risks]

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(usernames, total_hours, color="red")
    plt.xlabel("Employees")
    plt.ylabel("Total Work Hours (Last 7 Days)")
    plt.title("Employees at Risk of Burnout")
    plt.xticks(rotation=45)
    plt.grid(axis="y")

    # Save the plot as an image in memory
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="png")
    plt.close()
    img_bytes.seek(0)

    # Convert image to base64
    img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

    return {"image": f"data:image/png;base64,{img_base64}"}

@app.get("/predict-project-delay-graph")
def predict_project_delay_graph(project_id: int, db: Session = Depends(get_db)):
    """Generate a pie chart for project task completion & estimated delay"""
    
    project = db.query(ProjectDB).filter(ProjectDB.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    tasks = db.query(TaskDB).filter(TaskDB.project_id == project_id).all()
    if not tasks:
        return {"message": "No tasks found for this project"}

    completed_tasks = sum(1 for task in tasks if task.status == "Completed")
    total_tasks = len(tasks)

    completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0

    # Calculate estimated delay using project end date instead of fixed deadline_days
    days_remaining = (project.end_date - datetime.utcnow()).days
    estimated_delay_days = days_remaining * (1 - completion_rate)

    # Plot pie chart
    plt.figure(figsize=(6, 6))
    labels = ["Completed Tasks", "Pending Tasks"]
    sizes = [completed_tasks, total_tasks - completed_tasks]
    colors = ["green", "gray"]
    
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
    plt.title(f"Task Completion for Project {project_id}")

    # Save the plot as an image in memory
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="png")
    plt.close()
    img_bytes.seek(0)

    # Convert image to base64
    img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

    return {
        "completion_rate": round(completion_rate * 100, 2),
        "estimated_delay_days": max(0, round(estimated_delay_days)),
        "image": f"data:image/png;base64,{img_base64}"
    }

class IntegrationLogDB(Base):
    __tablename__ = "integration_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    system = Column(String, nullable=False)  # hr, payroll, project_management
    action = Column(String, nullable=False)  # sync, update
    request_data = Column(JSON, nullable=False)
    response_data = Column(JSON, nullable=True)
    status = Column(String, nullable=False)  # success, failed
    timestamp = Column(DateTime, default=datetime.utcnow)


# ✅ Pydantic Model for Integration Requests
class IntegrationRequest(BaseModel):
    system: str  # hr, payroll, project_management
    action: str  # sync, update
    data: Dict

# ✅ Dependency for Database Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

external_systems = {
    "hr": "https://hr-system.example.com/api",
    "payroll": "https://payroll.example.com/api"
}

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from time_tracking import IntegrationLogDB, get_db, IntegrationRequest
from datetime import datetime
import random

@app.post("/integrate-system")
def integrate_system(request: IntegrationRequest, db: Session = Depends(get_db)):
    """Handle system integration and log the attempt"""
    
    if request.system not in ["hr", "payroll", "project_management"]:
        raise HTTPException(status_code=400, detail="Invalid system type")

    if request.action not in ["sync", "update"]:
        raise HTTPException(status_code=400, detail="Invalid action type")

    # Simulating an external system API call (mocked response)
    response = {"message": f"{request.action} request sent to {request.system} system"}
    
    # Simulated success/failure for logging
    integration_status = "success" if random.choice([True, False]) else "failed"

    # Save integration log to the database
    log_entry = IntegrationLogDB(
        system=request.system,
        action=request.action,
        request_data=request.data,
        response_data=response,
        status=integration_status,
        timestamp=datetime.utcnow()
    )

    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)

    return {
        "message": f"Integration with {request.system} system {integration_status}",
        "log_id": log_entry.id,
        "response": response
    }


# Define external system URLs (Replace with actual endpoints)
external_systems = {
    "payroll": "https://payroll.example.com/api"
}

@app.get("/sync-payroll")
def sync_payroll(db: Session = Depends(get_db)) -> Dict:
    """Sync payroll data with an external payroll system"""

    # Fetch all payroll-related employee time entries
    payroll_data = {
        "employees": [
            {
                "username": entry.username,
                "clock_in": entry.clock_in.strftime("%Y-%m-%d %H:%M:%S"),
                "clock_out": entry.clock_out.strftime("%Y-%m-%d %H:%M:%S") if entry.clock_out else None,
                "total_hours": entry.total_hours
            }
            for entry in db.query(TimeEntryDB).all()
        ]
    }

    # Check if payroll system URL is set
    if "payroll" not in external_systems or not external_systems["payroll"]:
        raise HTTPException(status_code=500, detail="Payroll system integration URL is missing")

    try:
        # Send data to external payroll system
        response = requests.post(f"{external_systems['payroll']}/sync", json=payroll_data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to sync with payroll system: {str(e)}")
external_systems = {
    "hr": "https://hr.example.com/api"
}

@app.get("/sync-hr")
def sync_hr(db: Session = Depends(get_db)) -> Dict:
    """Sync HR data with an external HR system"""

    # Fetch all HR-related employee data
    hr_data = {
        "attendance": [
            {
                "username": entry.username,
                "date": entry.date.strftime("%Y-%m-%d"),
                "status": entry.status,
                "check_in": entry.check_in.strftime("%Y-%m-%d %H:%M:%S") if entry.check_in else None,
                "check_out": entry.check_out.strftime("%Y-%m-%d %H:%M:%S") if entry.check_out else None
            }
            for entry in db.query(AttendanceDB).all()
        ],
        "leaves": [
            {
                "username": entry.username,
                "start_date": entry.start_date.strftime("%Y-%m-%d"),
                "end_date": entry.end_date.strftime("%Y-%m-%d"),
                "reason": entry.reason,
                "status": entry.status
            }
            for entry in db.query(LeaveRequestDB).all()
        ]
    }

    # Check if HR system URL is set
    if "hr" not in external_systems or not external_systems["hr"]:
        raise HTTPException(status_code=500, detail="HR system integration URL is missing")

    try:
        # Send data to external HR system
        response = requests.post(f"{external_systems['hr']}/sync", json=hr_data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to sync with HR system: {str(e)}")


from sqlalchemy.sql import func

@app.get("/ai-analytics")
def get_ai_analytics(db: Session = Depends(get_db)):
    """Fetch AI-based performance trends and insights with a visualization"""

    today = datetime.today()
    last_14_days = today - timedelta(days=14)

    # Fetch real work hours for the past 14 days
    performance_trends = (
        db.query(
            TimeEntryDB.clock_in.label("date"),
            func.sum(TimeEntryDB.total_hours).label("performance_score")
        )
        .filter(TimeEntryDB.clock_in >= last_14_days)
        .group_by(TimeEntryDB.clock_in)
        .all()
    )

    # Convert results to a structured format
    trends = [{"date": record.date.strftime("%Y-%m-%d"), "performance_score": record.performance_score} for record in performance_trends]

    # Generate AI-based insights
    avg_performance = sum(record.performance_score for record in performance_trends) / len(performance_trends) if performance_trends else 0
    insights = f"AI predicts an increase in productivity next week. Average performance score: {round(avg_performance, 2)} hours/day."

    # Generate Graph
    dates = [record.date.strftime("%Y-%m-%d") for record in performance_trends]
    scores = [record.performance_score for record in performance_trends]

    plt.figure(figsize=(8, 5))
    plt.plot(dates, scores, marker="o", linestyle="-", color="b", label="Performance Trend")
    plt.xlabel("Date")
    plt.ylabel("Total Work Hours")
    plt.title("Employee Performance Trends (Last 14 Days)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()

    # Save graph as base64
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="png")
    plt.close()
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

    return {
        "performance_trends": trends,
        "insights": insights,
        "graph": f"data:image/png;base64,{img_base64}"
    }

Base.metadata.create_all(bind=engine)  # Ensure this is called only once
