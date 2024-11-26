#!/usr/bin/env python3
"""a script that displays the number of launches per rocket."""
import requests
if __name__ == '__main__':
    """the result ordered by the number launches (descending)"""

    url = "https://api.spacexdata.com/v4/launches/"
    resp = requests.get(url).json()
    rocket_launch_c = {}
    for launch in resp:
        rocket_id = launch['rocket']
        if rocket_id in rocket_launch_c:
            rocket_launch_c[rocket_id] += 1
        else:
            rocket_launch_c[rocket_id] = 1

    rocket_url = "https://api.spacexdata.com/v4/rockets/"
    rock_res = requests.get(rocket_url).json()
    rocket_id_name = {}
    for rocket in rock_res:
        rocket_id_name[rocket['id']] = rocket["name"]

    launch_data = []
    for id, count in rocket_launch_c.items():
        if id in rocket_id_name:
            rocket_name = rocket_id_name[id]
            launch_data.append((rocket_name, count))
    launch_data.sort(key=lambda x: (-x[1], x[0]))

    for name, count in launch_data:
        print(f"{name}: {count}")
