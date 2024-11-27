#!/usr/bin/env python3
"""a Python script that provides some stats
about Nginx logs stored in MongoD"""
from pymongo import MongoClient

if __name__ == "__main__":
    """Display (same as the example):
    first line: x logs where x is the number of documents in this collection
    second line: Methods:5 lines with the number of documents
    with the method = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    in this order (see example below - warning: it’s a tabulation
    before each line) one line with the number of documents with:
    method=GET, path=/status"""
    con = MongoClient('mongodb://127.0.0.1:27017')
    db = con.logs
    collection = db.nginx
    n_logs = collection.count_documents({})
    print(f'{n_logs} logs')
    method = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods:")
    for m in method:
        print(f'\tmethod {m}: {collection.count_documents({"method":m})}')
    check = collection.count_documents({"method": "GET", "path": "/status"})
    print(f"{check} status check")
