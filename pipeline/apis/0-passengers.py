#!/usr/bin/env python3
"""a method that returns the list of ships that can hold
a given number of passengers"""
import requests


def availableShips(passengerCount):
    """If no ship available, return an empty list."""
    url = "https://swapi-api.hbtn.io/api/starships/"
    ships = []
    while url:
        resp = requests.get(url)
        data = resp.json()
        for ship in data.get('results'):
            try:
                passengers = ship['passengers']
                if passengers.isdigit() and int(passengers) >= passengerCount:
                    ships.append(ship['name'])
            except ValueError:
                continue
        url = data.get('next')
    return ships
