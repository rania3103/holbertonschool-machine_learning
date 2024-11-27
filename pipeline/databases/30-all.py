#!/usr/bin/env python3
""" a Python function that lists all documents in a collection"""


def list_all(mongo_collection):
    """Returns an empty list if no document in the collection"""
    if mongo_collection is None:
        return []
    else:
        return list(mongo_collection.find())
