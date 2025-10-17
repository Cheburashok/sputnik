import requests
import yaml
import sys
from pathlib import Path


def get_copernicus_token(creds_path: str = ".credentials.yaml") -> str:
    path = Path(creds_path)
    if not path.exists():
        sys.exit(f"Credentials file not found: {creds_path}")

    with open(path, "r") as f:
        creds = yaml.safe_load(f)

    username = creds.get("username")
    password = creds.get("password")
    if not username or not password:
        sys.exit("Missing username or password in credentials file.")

    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
    }

    resp = requests.post(url, data=data)
    if resp.status_code != 200:
        sys.exit(f"Token request failed: {resp.status_code} {resp.text}")

    token = resp.json().get("access_token")
    if not token:
        sys.exit("Failed to obtain access token.")

    print("Access token obtained successfully.")
    return token

if __name__ == "__main__":
    token = get_copernicus_token()
    print(token)
    with open('.copernicus_token.txt', 'w') as f:
        f.write(token)
