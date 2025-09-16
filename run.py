import uvicorn  
from utils.config import config  
  
if __name__ == "__main__":  
    uvicorn.run(  
        "api:app",  
        host=config.HOST,  
        port=config.PORT,  
        log_level=config.LOG_LEVEL.lower(),  
        reload=config.RELOAD
    )