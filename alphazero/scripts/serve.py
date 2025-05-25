import os

import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("LOCAL_ALPHAZERO", 8000))
    uvicorn.run(
        "adapters.api_fastapi:create_app",
        factory=True,
        host="0.0.0.0",
        port=port,
        reload=True,
    )
