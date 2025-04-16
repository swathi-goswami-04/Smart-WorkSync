# 🚀 Smart WorkSync - Intelligent Time & Task Management System (FastAPI)

> A robust and scalable time tracking and workforce management backend API built using FastAPI. Includes secure login (OTP), project and task allocation, AI-powered workload predictions, HR/payroll integrations, automated reporting, and alert systems.

---

 Features

## Authentication
- Secure login using **OTP** via email
- **JWT-based token** authorization
- **Google OAuth 2.0** integration

## User Management
- User **registration**, **update**, **deletion**, and **role-based access**
- Role-based permissions (`admin`, `employee`, etc.)
- User verification & activity logs

## Task & Project Management
- Create and assign **projects** and **tasks**
- Track task status (`To Do`, `In Progress`, `Completed`)
- Log time manually or with live **Clock In/Clock Out**

## Attendance & Leave Management
- Record attendance status (Present, Absent, Late)
- Request & approve **leave applications**
- Attendance change requests with approval system

## AI + Analytics
- 📊 Predict employee **workload trends** with ML
- 🔥 Detect **burnout risk** using weekly work hours
- ⏳ Estimate **project delays** based on task completion

## Reporting
- Generate **Excel** and **PDF** reports:
  - Work hours
  - Project allocations
  - Overtime analysis

## Alerts & Notifications
- In-app alerts for key events
- Email notifications (SMTP configured)
- Admin alert dashboard

## External Integrations
- 🔄 Sync with mock **HR** and **Payroll** APIs
- Log integration attempts with status reports

---

##  Tech Stack

| Category         | Tech                              |
|------------------|-----------------------------------|
| Backend API      | FastAPI                           |
| ORM              | SQLAlchemy                        |
| Auth             | JWT, OAuth2, PyOTP                |
| Database         | SQLite (easily swappable)         |
| ML & Analytics   | Scikit-learn, Pandas, Matplotlib  |
| Email            | SMTP + `email.message`            |
| Reporting        | FPDF, PDFKit, Excel               |
| Deployment Ready | CORS, Env Config, Modular Setup   |

---

## 🛠 Setup & Run Locally

```bash
# 1. Clone this repo
git clone https://github.com/swathi-goswami-04/smart-worksync.git
cd smart-worksync

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables in `.env`
MAIL_USERNAME=your_email@example.com
MAIL_PASSWORD=your_password
MAIL_FROM=your_email@example.com
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_STARTTLS=True
MAIL_SSL_TLS=False
USE_CREDENTIALS=True
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# 5. Run the server
uvicorn main:app --reload
