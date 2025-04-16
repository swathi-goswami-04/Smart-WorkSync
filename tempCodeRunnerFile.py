class AttendanceResponse(BaseModel):
    username: str
    date: datetime
    status: str
    check_in: Optional[datetime]
    check_out: Optional[datetime]
    approved: bool

    class Config:
        from_attributes = True
