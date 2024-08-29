# Usage
## Docker
### Build the Docker imagte
```
docker build -t yolo_streamlit_app .
```

### Run the Docker container
```
docker run -p 8501:8501 yolo_streamlit_app
```

## Streamlit
### Test mode
```
streamlit run app.py
```