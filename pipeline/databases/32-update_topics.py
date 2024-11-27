#!/usr/bin/env python3
"""a Python function that changes all topics
of a school document based on the name"""


def update_topics(mongo_collection, name, topics):
    """mongo_collection will be the pymongo collection objec
    topics (list of strings)will be the list
    of topics approached in the school"""
    mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}}
    )
