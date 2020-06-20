import requests
import json
import base64
import hmac
import hashlib
import datetime, time


base_url = "https://api.sandbox.gemini.com"
endpoint = "/v1/order/new"
#endpoint = "/v1/balances"
url = base_url + endpoint

gemini_api_key = "account-lDs0Oamjmhfke0Cd63Sw"
gemini_api_secret = "CqLUV7NqU9yqiSqLxX8fZNZrktE".encode()

t = datetime.datetime.now()
payload_nonce = str(int(time.mktime(t.timetuple())))

payload = {"request": "/v1/order/new",
           "nonce": payload_nonce,
           "symbol": "btcusd",
           "amount": "0.01",
           "price": "9427",
           "side": "sell",
           "type": "exchange limit"
           }
#payload = {"request": "/v1/balances",
#           "nonce": payload_nonce,
#           }

encoded_payload = json.dumps(payload).encode()
b64 = base64.b64encode(encoded_payload)
signature = hmac.new(gemini_api_secret, b64, hashlib.sha384).hexdigest()

request_headers = {'Content-Type': "text/plain",
                   'Content-Length': "0",
                   'X-GEMINI-APIKEY': gemini_api_key,
                   'X-GEMINI-PAYLOAD': b64,
                   'X-GEMINI-SIGNATURE': signature,
                   'Cache-Control': "no-cache"}

response = requests.post(url,
                         data=None,
                         headers=request_headers)

new_clearing_order = response.json()
print(new_clearing_order)
