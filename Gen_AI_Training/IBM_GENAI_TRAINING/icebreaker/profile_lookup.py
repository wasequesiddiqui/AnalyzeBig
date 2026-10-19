
# profile_lookup.py

import requests


def lookup_profile():
    """Fetch sample profile data from a free test API.

    Note: JSONPlaceholder returns fictional sample user data,
    not actual LinkedIn profile information.
    """
    url = "https://jsonplaceholder.typicode.com/users/1"

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    profile = response.json()

    # Normalize the response into a simple structure
    # for use by the Icebreaker Bot.
    result = {
        "name": profile.get("name"),
        "username": profile.get("username"),
        "email": profile.get("email"),
        "company": profile.get("company", {}).get("name"),
        "website": profile.get("website"),
        "location": {
            "city": profile.get("address", {}).get("city"),
            "zipcode": profile.get("address", {}).get("zipcode"),
        },
    }

    return result


if __name__ == "__main__":
    try:
        profile = lookup_profile()

        print("Name:", profile.get("name"))
        print("Email:", profile.get("email"))
        print("Company:", profile.get("company"))
        print("Location:", profile.get("location", {}).get("city"))

    except requests.RequestException as exc:
        print(f"Profile lookup failed: {exc}")
