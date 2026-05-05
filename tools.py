import math
import os
import requests
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase


# ── Math tools ──────────────────────────────────────────────────────────────

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: float, b: float) -> str:
    """Divide a by b. Returns an error string if b is zero."""
    if b == 0:
        return "Error: division by zero"
    return str(a / b)


@tool
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return base ** exponent


@tool
def sqrt(x: float) -> str:
    """Compute the square root of x. Returns an error string for negative input."""
    if x < 0:
        return "Error: square root of negative number"
    return str(math.sqrt(x))


@tool
def modulo(a: float, b: float) -> str:
    """Return the remainder of a divided by b."""
    if b == 0:
        return "Error: modulo by zero"
    return str(a % b)


@tool
def absolute(x: float) -> float:
    """Return the absolute value of x."""
    return abs(x)


@tool
def factorial(n: int) -> str:
    """Return the factorial of a non-negative integer n."""
    if n < 0:
        return "Error: factorial of negative number"
    if n > 170:
        return "Error: n too large (max 170)"
    return str(math.factorial(n))


@tool
def log(x: float, base: float = math.e) -> str:
    """Return the logarithm of x. Defaults to natural log; pass base=10 for log10."""
    if x <= 0:
        return "Error: logarithm of non-positive number"
    if base <= 0 or base == 1:
        return "Error: invalid base"
    return str(math.log(x, base))


@tool
def sin(degrees: float) -> float:
    """Return the sine of an angle given in degrees."""
    return math.sin(math.radians(degrees))


@tool
def cos(degrees: float) -> float:
    """Return the cosine of an angle given in degrees."""
    return math.cos(math.radians(degrees))


@tool
def tan(degrees: float) -> str:
    """Return the tangent of an angle given in degrees. Returns error for undefined values (90, 270, ...)."""
    rad = math.radians(degrees)
    if abs(math.cos(rad)) < 1e-10:
        return "Error: tangent undefined at this angle"
    return str(math.tan(rad))


@tool
def round_number(x: float, decimal_places: int = 0) -> float:
    """Round x to the given number of decimal places (default 0)."""
    return round(x, decimal_places)


MATH_TOOLS = [
    add, subtract, multiply, divide, power, sqrt,
    modulo, absolute, factorial, log, sin, cos, tan, round_number,
]


# ── Weather / search tools ───────────────────────────────────────────────────

@tool
def weather(city: str) -> str:
    """Fetch current weather for a given city. Returns temperature, conditions, and humidity."""
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    if not api_key:
        return "Weather tool unavailable: OPENWEATHER_API_KEY not set."
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        temp = data["main"]["temp"]
        feels = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]
        return f"{city}: {temp}°C (feels like {feels}°C), humidity {humidity}%, {description}"
    except requests.HTTPError as e:
        return f"Weather API error: {e.response.status_code} — city not found or API key invalid."
    except Exception as e:
        return f"Weather error: {e}"


_shelter_db = SQLDatabase.from_uri("sqlite:///data/animal_shelter.db")


@tool
def query_db(query: str) -> str:
    """Run a read-only SQL query against the animal shelter database.
    The database has one table called 'animals' with columns: id, name, type, color, age.
    type is one of 'dog', 'cat', or 'bird'.
    If you run into an error, inspect the schema first with: SELECT * FROM sqlite_master WHERE type='table';
    """
    try:
        return _shelter_db.run(query)
    except Exception as e:
        return f"Error querying database: {e}"


@tool
def web_search(query: str) -> str:
    """Search the web and return a summary of the top results."""
    if os.getenv("TAVILY_API_KEY"):
        from langchain_community.tools.tavily_search import TavilySearchResults
        results = TavilySearchResults(max_results=3).invoke(query)
        return "\n".join(f"- {r['url']}: {r['content'][:300]}" for r in results)
    else:
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun().invoke(query)
