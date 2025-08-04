import requests

def fetch_weather(query, context):
    # Extract city from query more robustly in production
    city = "Mumbai"  # default or parse from query
    api_key = context["weather_api_key"]
    url = (f"http://api.openweathermap.org/data/2.5/weather?q={city}"
           f"&appid={api_key}&units=metric")
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.json()  # Pass to LLM for formatting
    else:
        return {"error": "Weather data not found."}
