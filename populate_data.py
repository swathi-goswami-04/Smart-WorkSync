import pandas as pd
from sqlalchemy.orm import Session
from time_tracking import (
    SessionLocal, User, ProjectDB, TaskDB, TimeEntryDB, AttendanceDB, LeaveRequestDB,
    AttendanceChangeRequestDB, AlertDB, ReportDB, PredictionDB, IntegrationLogDB
)
from datetime import datetime

def parse_date(date_value):
    """Automatically detects and converts date formats"""
    if pd.isna(date_value) or date_value == "" or date_value is None:
        return None  # ✅ Handle empty values safely

    # ✅ Convert to string if it's an integer or float (to avoid type errors)
    date_str = str(date_value)

    # ✅ List of possible formats
    formats = ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y"]  

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)  # ✅ Parse correctly
        except ValueError:
            continue  # Try the next format

    print(f"⚠️ Warning: Unknown date format '{date_str}', skipping...")
    return None  # ✅ Return None if no format matches
def clear_existing_data(session):
    """Clears existing data from all tables before inserting new records."""
    try:
        session.query(User).delete()
        session.query(ProjectDB).delete()
        session.query(TaskDB).delete()
        session.query(TimeEntryDB).delete()
        session.query(AttendanceDB).delete()
        session.query(LeaveRequestDB).delete()
        session.query(AttendanceChangeRequestDB).delete()
        session.query(AlertDB).delete()
        session.query(ReportDB).delete()
        session.query(PredictionDB).delete()
        session.query(IntegrationLogDB).delete()

        session.commit()  # ✅ Commit changes before inserting new data
        print("✅ Cleared existing data successfully!")

    except Exception as e:
        session.rollback()
        print(f"❌ Error while clearing data: {e}")


def insert_data_from_csv():
    df = pd.read_csv("demo_users.csv")
    session = SessionLocal()

    try:
        for _, row in df.iterrows():
            
                # ✅ Insert User Data
            if not session.query(User).filter(User.username == row["Username"]).first():
                    user = User(
                        username=row["Username"],
                        email=row["Email"],
                        password=row["Password (SHA-256 Hashed)"],
                        role=row["Role"],
                        otp_secret=row.get("Otp_Code", ""),
                        
                        userid=int(row["User_ID"]),
                        ph_no=row.get("Phone_number"),
                        verify=bool(row.get("Is_Verified", False)),
                        joining=parse_date(row["Date_Joined"]),  # ✅ Fix date parsing
                        lastseen=parse_date(row["Last_Login"]),  # ✅ Fix date parsing
                        status=row.get("Status", "Active")
                    )
                    session.add(user)

                # ✅ Insert Project Data
            if pd.notna(row.get("Project_ID")):
                    project = session.query(ProjectDB).filter(ProjectDB.id == row["Project_ID"]).first()
                    if not project:
                        project = ProjectDB(
                            id=row["Project_ID"],
                            name=row["Project_Name"],
                            description=row["Project_Description"],
                            assigned_to=row["Username"]
                        )
                        session.add(project)

                # ✅ Insert Task Data
            if pd.notna(row.get("Task_ID")):
                    task = session.query(TaskDB).filter(TaskDB.id == row["Task_ID"]).first()
                    if not task:
                        task = TaskDB(
                            id=row["Task_ID"],
                            project_id=row["Project_ID"],
                            name=row["Task_Name"],
                            status=row["Task_Status"],
                            time_logged=float(row.get("Time_Logged", 0))
                        )
                        session.add(task)

            # ✅ Insert Time Tracking Data
            if pd.notna(row.get("Clock_In")) and pd.notna(row.get("Clock_Out")):
                    clock_in = parse_date(row["Clock_In"])
                    clock_out = parse_date(row["Clock_Out"])

                    if not session.query(TimeEntryDB).filter(TimeEntryDB.username == row["Username"], TimeEntryDB.clock_in == clock_in).first():
                        time_entry = TimeEntryDB(
                            username=row["Username"],
                            task_id=row["Task_ID"],
                            clock_in=clock_in,
                            clock_out=clock_out,
                            total_hours=row["Time_Logged"],
                            location=row["Location"],
                            approved=row["Approved"]
                        )
                        session.add(time_entry)
                
                # ✅ Insert Attendance Data
            if pd.notna(row.get("Attendance_Date")):
                    attendance = session.query(AttendanceDB).filter(
                        AttendanceDB.username == row["Username"],
                        AttendanceDB.date == parse_date(row["Attendance_Date"])
                    ).first()
                    if not attendance:
                        attendance_entry = AttendanceDB(
                            username=row["Username"],
                            date=parse_date(row["Attendance_Date"]),
                            status=row["Attendance_Status"],
                            check_in=parse_date(row.get("Check_In")),
                            check_out=parse_date(row.get("Check_Out")),
                            approved=row.get("Approved", False)
                        )
                        session.add(attendance_entry)
                
                # ✅ Insert Leave Requests
            if pd.notna(row.get("Leave_Start_Date")):
                    leave_request = session.query(LeaveRequestDB).filter(
                        LeaveRequestDB.username == row["Username"],
                        LeaveRequestDB.start_date == parse_date(row["Leave_Start_Date"])
                    ).first()
                    if not leave_request:
                        leave_entry = LeaveRequestDB(
                            username=row["Username"],
                            start_date=parse_date(row["Leave_Start_Date"]),
                            end_date=parse_date(row["Leave_End_Date"]),
                            reason=row["Leave_Reason"],
                            status=row["Leave_Status"]
                        )
                        session.add(leave_entry)
                
            if pd.notna(row.get("Attendance_Change_Date")):
                    request = session.query(AttendanceChangeRequestDB).filter(
                        AttendanceChangeRequestDB.username == row["Username"],
                        AttendanceChangeRequestDB.date == parse_date(row["Attendance_Change_Date"])
                    ).first()
                    if not request:
                        attendance_request = AttendanceChangeRequestDB(
                            username=row["Username"],
                            date=parse_date(row["Attendance_Change_Date"]),
                            status=row["Attendance_Change_Status"]
                        )
                        session.add(attendance_request)
                
            if pd.notna(row.get("Alert_Type")):
                    alert = session.query(AlertDB).filter(
                        AlertDB.username == row["Username"],
                        AlertDB.timestamp == parse_date(row["Alert_Timestamp"])
                    ).first()
                    if not alert:
                        new_alert = AlertDB(
                            username=row["Username"],
                            type=row["Alert_Type"],
                            message=row["Alert_Message"],
                            timestamp=parse_date(row["Alert_Timestamp"])
                        )
                        session.add(new_alert)
                
            if pd.notna(row.get("Report_Type")):
                    report = session.query(ReportDB).filter(
                        ReportDB.start_date == parse_date(row["Start_Date"]),
                        ReportDB.end_date == parse_date(row["End_Date"]),
                        ReportDB.report_type == row["Report_Type"]
                    ).first()
                    if not report:
                        new_report = ReportDB(
                            username=row["Username"],
                            start_date=parse_date(row["Start_Date"]),
                            end_date=parse_date(row["End_Date"]),
                            report_type=row["Report_Type"],
                            file_path=f"reports/{row['Report_Type']}_report.xlsx"
                        )
                        session.add(new_report)
                
            if pd.notna(row.get("Prediction_Type")):
                    prediction = session.query(PredictionDB).filter(
                        PredictionDB.username == row["Username"],
                        PredictionDB.prediction_type == row["Prediction_Type"],
                        PredictionDB.timestamp == parse_date(row["Timestamp"])
                    ).first()
                    if not prediction:
                        new_prediction = PredictionDB(
                            username=row["Username"],
                            prediction_type=row["Prediction_Type"],
                            predicted_value=row["Predicted_Value"],
                            timestamp=parse_date(row["Timestamp"])
                        )
                        session.add(new_prediction)
                
            if pd.notna(row.get("System")):
                    log_entry = session.query(IntegrationLogDB).filter(
                        IntegrationLogDB.system == row["System"],
                        IntegrationLogDB.action == row["Action"],
                        IntegrationLogDB.timestamp == parse_date(row["Timestamp"])
                    ).first()
                    if not log_entry:
                        new_log = IntegrationLogDB(
                            system=row["System"],
                            action=row["Action"],
                            request_data=row["Request_Data"],
                            response_data=row["Response_Data"] if pd.notna(row.get("Response_Data")) else None,
                            status=row["Status"],
                            timestamp=parse_date(row["Timestamp"])
                        )
                        session.add(new_log)
            session.commit()
            print("✅ Data imported successfully!")

        

    except Exception as e:
        session.rollback()
        print(f"❌ Error: {e}")
        


    finally:
        session.close()

if __name__ == "__main__":
    insert_data_from_csv()






