import argparse
import os
import uvicorn

def main():
    port = int(os.getenv("DATABRICKS_APP_PORT", "8000"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()
    # importante: carregar exatamente server.app:combined_app
    uvicorn.run("server.app:combined_app", host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()