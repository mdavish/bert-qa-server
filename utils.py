import requests
from functools import lru_cache

@lru_cache()
def boilerpipe_from_url(url):
    bp_endpoint = 'https://boilerpipe-web.appspot.com/extract'
    params = {
        'url': url,
        'output': 'json'
    }
    response = requests.get(bp_endpoint, params)
    response = response.json()
    return response
