from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator, ValidationError
from datetime import datetime

# ---------------------------------------------------------
# MODEL DEFINITION
# ---------------------------------------------------------

class UserProfile(BaseModel):
    """
    In Pydantic, models inherit from BaseModel. 
    Think of this as a 'schema' that defines what valid data looks like.
    """

    # 1. POSITIVE INTEGER ENFORCEMENT
    # 'gt=0' ensures the ID is greater than zero.
    id: int = Field(gt=0, description="The unique primary key for the user")

    # 2. STRING CONSTRAINTS
    # Pydantic will strip whitespace and check length before accepting the string.
    username: str = Field(
        min_length=3, 
        max_length=20, 
        pattern=r"^[a-zA-Z0-9_-]+$" # Regex: Only letters, numbers, underscores, hyphens
    )

    # 3. SPECIALIZED TYPES
    # EmailStr requires the 'email-validator' package. 
    # It checks for valid TLDs and '@' symbol placement.
    email: EmailStr

    # 4. DATA COERCION
    # If you pass a valid ISO string (e.g., "2026-01-01"), Pydantic 
    # automatically converts it into a Python datetime object.
    joined_at: Optional[datetime] = None

    # 5. NESTED MODELS & DEFAULTS
    # An empty list is the default if no tags are provided.
    tags: List[str] = []

    # ---------------------------------------------------------
    # CUSTOM VALIDATION (The 'Logic' Layer)
    # ---------------------------------------------------------
    
    @field_validator('email')
    @classmethod
    def block_temporary_emails(cls, v: str) -> str:
        """Custom logic to reject specific domains."""
        if "dispostable.com" in v:
            raise ValueError("Temporary email addresses are not allowed.")
        return v.lower() # We can also normalize data (lowercase it) here

# ---------------------------------------------------------
# EXECUTION & ERROR HANDLING
# ---------------------------------------------------------

def process_user_data(data_input: dict):
    try:
        # The '**' unpacks the dictionary into keyword arguments.
        # This is where validation happens. If it fails, an exception is raised.
        user = UserProfile(**data_input)
        
        print("--- Validation Success ---")
        print(f"Object: {user}")
        
        # Accessing data is easy and IDE-friendly:
        print(f"Username: {user.username}")
        
        # Exporting back to a dictionary or JSON:
        # .model_dump() replaces the old .dict() method in Pydantic v2
        print(f"Clean Dictionary: {user.model_dump()}")

    except ValidationError as e:
        print("--- Validation Failed ---")
        # e.errors() returns a list of dictionaries explaining 
        # exactly which fields failed and why.
        for error in e.errors():
            print(f"Field: {error['loc']} | Message: {error['msg']}")

# ---------------------------------------------------------
# TEST CASES
# ---------------------------------------------------------

# Case 1: Valid data with coercion (ID is a string, joined_at is a string)
# Pydantic will try to "fix" these into an int and a datetime.
print("\nRunning Test 1 (Fixable Data):")
process_user_data({
    "id": "123", 
    "username": "python_fan",
    "email": "USER@Example.com",
    "joined_at": "2026-04-04T12:00:00"
})

# Case 2: Invalid data
print("\nRunning Test 2 (Broken Data):")
process_user_data({
    "id": -5,                    # Fails 'gt=0'
    "username": "!!",            # Fails min_length and regex
    "email": "not-an-email",     # Fails EmailStr validation
    "tags": "not-a-list"         # Fails type check (expects List[str])
})