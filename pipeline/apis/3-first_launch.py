#!/usr/bin/env python3
"""a script that displays the first launch with Name of the launch
The date (in local time), The rocket name,The name (with the locality)
of the launchpad"""
import requests
from datetime import datetime
if __name__ == '__main__':
    """use the date_unix for sorting it - and if 2 launches
    have the same date, use the first one in the API result"""
    url = "https://api.spacexdata.com/v4/launches/"
    resp = requests.get(url).json()
    resp.sort(key=lambda launch: launch["date_unix"])
    first_launch = resp[0]
    launch_name = first_launch.get('name')
    launch_date_unix = first_launch.get('date_unix')
    rocket_id = first_launch.get('rocket')
    launchpad_id = first_launch.get('launchpad')
    launch_date = datetime.fromtimestamp(launch_date_unix).isoformat()
    rocket_resp = requests.get(
        f"https://api.spacexdata.com/v4/rockets/{rocket_id}").json()
    rocket_name = rocket_resp.get("name")

    launchpad_resp = requests.get(
        f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}").json()
    launchpad_name = launchpad_resp.get("name")
    launchpad_locality = launchpad_resp.get('locality')
    s1 = f"{launch_name} ({launch_date}) {rocket_name}"
    s2 = f" - {launchpad_name} ({launchpad_locality})"
    print(s1 + s2)
