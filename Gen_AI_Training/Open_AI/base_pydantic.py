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
    # 1. id: int (The Type Hint)
    # This tells Python (and Pydantic) that the data must be an integer.
    # Coercion: If you pass the string "123", Pydantic is smart enough to convert it to the integer 123.
    # Strictness: If you pass "abc", it will raise a ValidationError because it cannot be turned into an integer.

    # 2. Field(...) (The Configuration)
    # The Field function is used to define attributes that go beyond simple type hints. Without it, you could only check if the input is an integer, but you couldn't check the value of that integer.
    
    # 3. gt=0 (The Validation Logic)
    # This stands for "Greater Than 0".
    # It ensures the ID is a positive number.
    # If a user tries to initialize the model with id=0 or id=-5, Pydantic will block it.
    # Other common constraints: lt (less than), ge (greater than or equal to), and le (less than or equal to).

    # 4. description="..." (The Metadata)
    # This does not affect the code's logic, but it is incredibly useful for documentation:
    # Self-Documenting Code: Other developers reading your code immediately know what this field represents.
    # Automated APIs: If you use this model in FastAPI, this description automatically appears in the Swagger/OpenAPI documentation, telling the front-end developers or API users exactly what that field is for.
    
    # The Flow of Validation
    # When you call UserProfile(id=10), Pydantic performs these steps in order:
    # Input: Receives data (e.g., id="10").
    # Cast: Sees int hint, converts "10" → 10.
    # Validate: Checks gt=0. Since 10 > 0, it passes.
    # Finalize: Assigns the value to the object attribute.

    # To put it simply: gt is a keyword argument (a specific parameter name) defined by the Pydantic library.

    # -----------------------------------------------------------------------------
    # PYDANTIC FIELD KEYWORDS REFERENCE
    # These are reserved arguments used inside Field(). They cannot be renamed.
    # -----------------------------------------------------------------------------

    # NUMERIC CONSTRAINTS (int, float)
    # gt: Greater Than
    # ge: Greater than or Equal to
    # lt: Less Than
    # le: Less than or Equal to
    # multiple_of: Number must be divisible by this value

    # STRING CONSTRAINTS (str)
    # min_length: Minimum number of characters
    # max_length: Maximum number of characters
    # pattern:    A Regex string the value must match (e.g., r"^[a-z]+$")

    # LIST/COLLECTION CONSTRAINTS (list, set, tuple)
    # min_length: Minimum number of items in the collection
    # max_length: Maximum number of items in the collection

    # METADATA (For Documentation/APIs)
    # default:     The value used if none is provided (first positional argument)
    # alias:       The name expected in the input data (e.g., alias="ID" for field 'id')
    # title:       A human-readable title for the field
    # description: A detailed explanation for documentation (Swagger/OpenAPI)
    # examples:    List of example values for documentation

    # -----------------------------------------------------------------------------
    # QUICK EXAMPLE USAGE:
    # -----------------------------------------------------------------------------
    # from pydantic import BaseModel, Field
    #
    # class Product(BaseModel):
    #     price: float = Field(gt=0, le=1000, description="Price between 0 and 1000")
    #     code: str = Field(min_length=3, pattern=r"^[A-Z]+$", alias="product_code")
    # -----------------------------------------------------------------------------

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