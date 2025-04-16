from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, ForeignKey, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List
from typing import Dict

DATABASE_URL = "sqlite:///./users.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# ✅ User Model
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
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str
    userid: Optional[int] = None
    ph_no: Optional[str] = None
    verify: Optional[bool] = False
    joining: Optional[datetime] = None
    lastseen: Optional[datetime] = None
    status: Optional[str] = "active"  
class ClockInRequest(BaseModel):
    username: str
    task_id: int
    location: str
class ClockOutRequest(BaseModel):
    username: str
    task_id: int
class StartBreakRequest(BaseModel):
    username: str
    task_id: int
class EndBreakRequest(BaseModel):
    username: str
    task_id: int
    
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str
    ph_no: Optional[str] = None
    joining: Optional[datetime] = None

# ✅ Project Model (FIXED MISSING REFERENCE)
class ProjectDB(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    assigned_to = Column(String, nullable=False)  # Comma-separated usernames
    
    tasks = relationship("TaskDB", back_populates="project", cascade="all, delete")

# ✅ Task Model (FIXED RELATIONSHIP & TIME LOGGED TYPE)
class TaskDB(Base):
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String, nullable=False)
    status = Column(String, default="To Do")
    time_logged = Column(Float, default=0.0)  # FIXED TYPE
    
    project = relationship("ProjectDB", back_populates="tasks")
    time_entries = relationship("TimeEntryDB", back_populates="task", cascade="all, delete")

# ✅ Time Entry Model (FIXED RELATIONSHIP)
class TimeEntryDB(Base):
    __tablename__ = "time_entries"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String, ForeignKey("users.username"), nullable=False)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)  # ADD FOREIGN KEY TO TASK
    clock_in = Column(DateTime, nullable=True)
    clock_out = Column(DateTime, nullable=True)
    total_hours = Column(Float)
    location = Column(String, nullable=True)
    approved = Column(Boolean, default=False)

    task = relationship("TaskDB", back_populates="time_entries")  # FIXED RELATIONSHIP

# ✅ Attendance Model
class AttendanceDB(Base):
    __tablename__ = "attendance"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String, ForeignKey("users.username"), nullable=False)
    date = Column(DateTime, nullable=False)
    status = Column(String, nullable=False)  # Present, Absent, Late
    check_in = Column(DateTime, nullable=True)
    check_out = Column(DateTime, nullable=True)
    approved = Column(Boolean, default=False)

# ✅ Leave Request Model
class LeaveRequestDB(Base):
    __tablename__ = "leave_requests"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String, ForeignKey("users.username"), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    reason = Column(String, nullable=False)
    status = Column(String, default="Pending")  # Pending, Approved, Rejected

# ✅ Attendance Change Request Model
class AttendanceChangeRequestDB(Base):
    __tablename__ = "attendance_change_requests"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String, ForeignKey("users.username"), nullable=False)
    date = Column(DateTime, nullable=False)
    status = Column(String, nullable=False)  # Present, Absent, Late
    approved = Column(Boolean, default=None)  # None = Pending, True = Approved, False = Rejected

# ✅ Ensure Tables Are Created (ONLY ONCE)
Base.metadata.create_all(bind=engine)

# ✅ Pydantic Models for API Responses
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

class TimeEntryResponse(BaseModel):
    id: int
    username: str
    task_id: int
    clock_in: Optional[datetime]
    clock_out: Optional[datetime]
    total_hours: Optional[float]
    location: Optional[str]
    approved: bool

    class Config:
        from_attributes = True
class TimeRecordResponse(BaseModel):
    id: int
    username: str
    task_id: int
    clock_in: datetime
    clock_out: Optional[datetime]
    total_hours: Optional[float]
    location: Optional[str]
    approved: bool
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    start_date: datetime
    end_date: Optional[datetime] = None
    status: Optional[str] = "ongoing"  # Default status
class TaskAssignRequest(BaseModel):
    username: str
    project_id: int
    name: str
    task_description: Optional[str] = None
    deadline: Optional[datetime] = None
    status: Optional[str] = "pending"  # Default status
class TaskUpdateRequest(BaseModel):
    task_name: Optional[str] = None
    task_description: Optional[str] = None
    deadline: Optional[datetime] = None
    status: Optional[str] = None
class LogTimeRequest(BaseModel):
    username: str
    task_id: int
    clock_in: datetime
    clock_out: datetime
    location: Optional[str] = None
class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    start_date: datetime
    end_date: Optional[datetime]
    status: str
    assigned_to: List[str] 

class TaskResponse(BaseModel):
    id: int
    project_id: int
    name: str
    status: str
    time_logged: float
    time_entries: List["TimeEntryResponse"] = []

    class Config:
        from_attributes = True

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

class AttendanceChangeRequestResponse(BaseModel):
    username: str
    date: datetime
    status: str
    approved: Optional[bool]

    class Config:
        from_attributes = True
        
class AlertDB(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String, ForeignKey("users.username"), nullable=False)
    type = Column(String, nullable=False)  # Clock-in, Clock-out, Break, Overtime
    message = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

# ✅ Pydantic Model for API Responses
class AlertResponse(BaseModel):
    username: str
    type: str
    message: str
    timestamp: datetime

    class Config:
        from_attributes = True
        
class ReportDB(Base):
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String, ForeignKey("users.username"), nullable=True)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    report_type = Column(String, nullable=False)  # work_hours, project_allocation, overtime
    file_path = Column(String, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)

class UserUpdate(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None
    ph_no: Optional[str] = None
    verify: Optional[bool] = None
    joining: Optional[datetime] = None
    lastseen: Optional[datetime] = None
    status: Optional[str] = None

# ✅ Pydantic Model for Report Requests
class ReportRequest(BaseModel):
    username: Optional[str] = None
    start_date: datetime
    end_date: datetime
    report_type: str  # work_hours, project_allocation, overtime
    
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
    
    
# ✅ SQLAlchemy Model for Integration Logs
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

# ✅ Ensure Tables Are Created
Base.metadata.create_all(bind=engine)

# ✅ Dependency for Database Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

