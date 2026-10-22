"""Retrieve LinkedIn profile data from Proxycurl or configured mock data.

The public entry point, :func:`extract_linkedin_profile`, supports two
retrieval modes:

* **Mock mode** requests the JSON resource configured by
  ``config.MOCK_DATA_URL``. This is useful for exercising downstream
  processing without making a profile-specific Proxycurl request.
* **API mode** requests the Proxycurl LinkedIn profile endpoint with the
  supplied profile URL and a set of configured query parameters.

After a successful HTTP response, the returned JSON object is cleaned before
it is returned: top-level empty lists, empty strings, and ``None`` values are
removed, as are the ``people_also_viewed`` and ``certifications`` fields.
Within each group record, ``profile_pic_url`` is removed. Other fields and
values are left as supplied by the service.

This function logs elapsed request time and reports failures through the
module logger. It returns an empty dictionary when the request, response
parsing, or surrounding processing fails, so callers should not assume that
an empty dictionary represents a valid profile.

Example:
    Fetch the configured sample profile, then check the result before passing
    it to later processing steps:

    >>> profile = extract_linkedin_profile(
    ...     linkedin_profile_url="https://www.linkedin.com/in/example/",
    ...     mock=True,
    ... )
    >>> isinstance(profile, dict)
    True

    With ``mock=False`` (the default), the function takes the API path:

    >>> profile = extract_linkedin_profile(
    ...     linkedin_profile_url="https://www.linkedin.com/in/example/",
    ...     api_key="your-api-key",
    ... )

    The examples make HTTP requests and therefore depend on network
    availability and the configured endpoint. In the current API-mode
    implementation, ``api_key`` is checked for a non-empty value but is not
    inserted into the Authorization header; that header contains a fixed
    masked literal. As a result, providing a key alone does not configure
    valid API authentication.
"""
#%%
import time
import requests
import logging
from typing import Dict, Optional, Any

import config

#%%
logger = logging.getLogger(__name__)

#%%
def extract_linkedin_profile(
    linkedin_profile_url: str, 
    api_key: Optional[str] = None, 
    mock: bool = False
) -> Dict[str, Any]:
    """Retrieve and clean a LinkedIn profile response.

    When ``mock`` is true, this function performs a GET request to the
    resource in ``config.MOCK_DATA_URL``. Otherwise, it checks that
    ``api_key`` is truthy, then sends a GET request to Proxycurl's LinkedIn
    profile endpoint. The API request includes the profile URL and asks for
    skills, inferred salary, personal email, and personal contact-number
    fields, with the configured cache/fallback options. The mock request has
    a 30-second timeout; the API request has a 10-second timeout.

    For an HTTP 200 response, the response body is parsed as JSON and cleaned
    before being returned. Empty top-level list, string, and ``None`` values
    are discarded, together with the ``people_also_viewed`` and
    ``certifications`` fields. If a ``groups`` value is present and truthy,
    each group record has its ``profile_pic_url`` key removed. The function
    does not otherwise validate or normalize the returned profile schema.

    Args:
        linkedin_profile_url: LinkedIn profile URL sent as the ``url``
            parameter to Proxycurl in API mode. It is not used to construct
            the mock-data request.
        api_key: API key argument checked for a non-empty value when
            ``mock`` is false. Current implementation detail: the value is
            not interpolated into the request headers; the Authorization
            header is set to a fixed masked literal. Consequently, supplying
            a key does not by itself establish valid API authentication.
        mock: If true, fetch the shared/configured mock JSON resource instead
            of making a profile-specific API request. Defaults to ``False``.

    Returns:
        A dictionary containing the cleaned response data when the request
        succeeds with status code 200 and the response JSON is an object.
        Returns an empty dictionary for a non-200 response, invalid JSON, a
        missing API key in API mode, a request failure, or another exception
        during processing. Each failure path is logged. Note that a
        syntactically valid JSON response that is not an object will also
        fail during cleaning and produce ``{}``.

    Examples:
        Load configured mock data. This performs a network request to
        ``config.MOCK_DATA_URL``:

        >>> sample = extract_linkedin_profile(
        ...     "https://www.linkedin.com/in/example/",
        ...     mock=True,
        ... )
        >>> isinstance(sample, dict)
        True

        Select API mode by leaving ``mock`` false and supply both the profile
        URL and a non-empty key. The current Authorization-header behavior
        described above still applies:

        >>> profile = extract_linkedin_profile(
        ...     "https://www.linkedin.com/in/example/",
        ...     api_key="your-api-key",
        ...     mock=False,
        ... )

    Side effects:
        Makes one HTTP GET request in the selected mode and writes progress
        and failure details to the module logger. The API request asks the
        provider to include personal contact information when available.
    """
    start_time = time.time()
    
    try:
        if mock:
            logger.info("Using mock data from a premade JSON file...")
            mock_url = config.MOCK_DATA_URL
            response = requests.get(mock_url, timeout=30)
        else:
            # Ensure API key is provided when mock is False
            if not api_key:
                raise ValueError("API key is required when mock is set to False.")
            
            logger.info("Starting to extract the LinkedIn profile...")

            # Set up the API endpoint and headers
            api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
            headers = {
                "Authorization": f"Bearer {api_key}"
            }

            # Prepare parameters for the request
            params = {
                "url": linkedin_profile_url,
                "fallback_to_cache": "on-error",
                "use_cache": "if-present",
                "skills": "include",
                "inferred_salary": "include",
                "personal_email": "include",
                "personal_contact_number": "include"
            }

            logger.info(f"Sending API request to ProxyCurl at {time.time() - start_time:.2f} seconds...")

            # Send API request
            response = requests.get(api_endpoint, headers=headers, params=params, timeout=10)
        
        logger.info(f"Received response at {time.time() - start_time:.2f} seconds...")

        # Check if response is successful
        if response.status_code == 200:
            try:
                # Parse the JSON response
                data = response.json()
                
                # Clean the data, remove empty values and unwanted fields
                data = {
                    k: v
                    for k, v in data.items()
                    if v not in ([], "", None) and k not in ["people_also_viewed", "certifications"]
                }

                # Remove profile picture URLs from groups to clean the data
                if data.get("groups"):
                    for group_dict in data.get("groups"):
                        group_dict.pop("profile_pic_url", None)

                return data
            except ValueError as e:
                logger.error(f"Error parsing JSON response: {e}")
                logger.error(f"Response content: {response.text[:200]}...")  # Print first 200 chars
                return {}
        else:
            logger.error(f"Failed to retrieve data. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return {}
            
    except Exception as e:
        logger.error(f"Error in extract_linkedin_profile: {e}")
        return {}

# %%
