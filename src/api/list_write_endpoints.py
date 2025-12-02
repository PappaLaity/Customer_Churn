import importlib
from fastapi import FastAPI
from fastapi.routing import APIRoute

# 🔹 Adapte ce chemin vers ton app FastAPI
# Exemple : from src.api.main import app
from src.api.main import app  

def list_routes(app: FastAPI):
    print("=== Tous les endpoints ===\n")
    write_endpoints = []
    read_endpoints = []

    for route in app.routes:
        if isinstance(route, APIRoute):
            methods = ",".join(route.methods)
            line = f"{methods:15} {route.path}"
            print(line)

            # Détecte les endpoints qui modifient la DB
            if any(m in route.methods for m in ["POST", "PUT", "DELETE"]):
                write_endpoints.append(line)
            else:
                read_endpoints.append(line)

    print("\n=== Endpoints qui écrivent dans la DB ===\n")
    for e in write_endpoints:
        print(e)

    print("\n=== Endpoints lecture seule (GET) ===\n")
    for e in read_endpoints:
        print(e)

    return write_endpoints, read_endpoints

if __name__ == "__main__":
    writes, reads = list_routes(app)
    print(f"\nNombre d'endpoints écriture: {len(writes)}")
    print(f"Nombre d'endpoints lecture seule: {len(reads)}")
