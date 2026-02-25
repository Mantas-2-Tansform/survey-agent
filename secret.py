import os
from google.cloud import secretmanager
import google.auth

def access_secret_version(secret_id: str, version_id: str = "latest") -> str:
    """
    Access the payload for the given secret version and return it as a string.
    """
    # Automatically determine the project ID from the environment
    try:
        _, project_id = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        print("Could not automatically determine credentials. Please run 'gcloud auth application-default login'")
        # For local development, you might fall back to an env var
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise

    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version.
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version.
    try:
        response = client.access_secret_version(name=name)
    except Exception as e:
        print(f"Error accessing secret: {secret_id}. Ensure it exists and you have permissions.")
        raise e

    # Return the decoded payload.
    payload = response.payload.data.decode("UTF-8")
    return payload