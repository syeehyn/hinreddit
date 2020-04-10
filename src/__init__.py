import json
from pathlib import Path
import os
# import praw as pr
ROOT_DIR = Path(__file__).parent.parent
CLIENTID = json.load(open(os.path.join(ROOT_DIR, 'config/credentials.json')))['client_id']
CLIENTSECRET = json.load(open(os.path.join(ROOT_DIR, 'config/credentials.json')))['client_secret']
USERAGENT = json.load(open(os.path.join(ROOT_DIR, 'config/credentials.json')))['user_agent']
USERNAME = json.load(open(os.path.join(ROOT_DIR, 'config/credentials.json')))['username']
PASSWORD = json.load(open(os.path.join(ROOT_DIR, 'config/credentials.json')))['password']