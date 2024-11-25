#!/usr/bin/env python3
"""a script that prints the location of a specific user"""
import requests
import sys
from datetime import datetime
if __name__ == '__main__':
    """If the user doesn’t exist, print Not found
    If the status code is 403, print Reset in X min where X
    is the number of minutes from now and the value of X-Ratelimit-Reset"""

    user_url = sys.argv[1]
    resp = requests.get(user_url)
    if resp.status_code == 404:
        print("Not found")
    elif resp.status_code == 403:
        reset_time = int(resp.headers.get('X-Ratelimit-Reset', 0))
        reset_time_min = int(
            max(0, (reset_time - datetime.now().timestamp()) // 60))
        print(f"Reset in {reset_time_min} min")
    elif resp.status_code == 200:
        data = resp.json().get('location')
        print(data)
