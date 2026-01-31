from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.exceptions import InvalidSignature
import requests
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent          # .../src/auth
SECRETS_DIR = BASE_DIR / "secrets"


class UserAuth:
    def __init__(self):
        # Currently no state is required on initialization
        pass

    def load_private_key_from_file(self, file_path):
        with open(file_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None,  # or provide a password if your key is encrypted
                backend=default_backend()
            )
        return private_key

    def sign_pss_text(self, private_key, text: str) -> str:
        message = text.encode('utf-8')
        try:
            signature = private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH
                ),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode('utf-8')
        except InvalidSignature as e:
            raise ValueError("RSA sign PSS failed") from e
    
    def test_authentication(self) -> bool:
        current_time = datetime.datetime.now()
        timestamp = current_time.timestamp()
        current_time_milliseconds = int(timestamp * 1000)
        timestampt_str = str(current_time_milliseconds)

        private_key = self.load_private_key_from_file(SECRETS_DIR / 'kalshi_private_key.pem')
        api_key = Path(SECRETS_DIR / 'kalshi-key-2.key').read_text().strip()

        method = "GET"
        base_url = 'https://api.elections.kalshi.com'
        path='/trade-api/v2/portfolio/balance'

        # Strip query parameters from path before signing
        path_without_query = path.split('?')[0]
        msg_string = timestampt_str + method + path_without_query
        sig = self.sign_pss_text(private_key, msg_string)

        headers = {
            'KALSHI-ACCESS-KEY': api_key,
            'KALSHI-ACCESS-SIGNATURE': sig,
            'KALSHI-ACCESS-TIMESTAMP': timestampt_str
        }

        response = requests.get(base_url + path, headers=headers)

        if response.status_code == 200:
            return True
        else:
            return False

