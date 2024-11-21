#!/usr/bin/env python3
"""a method that returns the list of names of
the home planets of all sentient species."""
import requests


def sentientPlanets():
    """sentient type is either in the classification
    or designation attributes."""
    url = "https://swapi-api.hbtn.io/api/species/"
    home_planets = []
    while url:
        resp = requests.get(url)
        data = resp.json()
        for spec in data.get('results'):
            try:
                if spec['designation'] == 'sentient' or \
                        spec['classification'] == 'sentient':
                    homeworld_url = spec.get('homeworld')
                    if homeworld_url:
                        homeworld_rsp = requests.get(homeworld_url).json()
                        home_planets.append(homeworld_rsp['name'])
            except BaseException:
                continue
        url = data.get('next')
    return home_planets
