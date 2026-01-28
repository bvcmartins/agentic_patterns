import httpx
from mcp.server.fastmcp import FastMCP
from bs4 import BeautifulSoup

mcp = FastMCP('tech_news')

USER_AGENT = "news-app/1.0"
NEWS_SITES = {
    "arstechnica": "https://arstechnica.com",
}


async def fetch_news(url: str) -> str:
    """
    Pulls and summarizes the latest news from the specified news site.
    """
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = " ".join([p.get_text() for p in paragraphs[:5]])
            return text
        except httpx.TimeoutException:
            return "Timeout Error"
        except Exception as e:
            return f"Error fetching news: {str(e)}"


@mcp.tool()
async def get_tech_news(source: str) -> str:
    """
    Fetches the latest news from a specific tech news source.

    Args:
        source: Name of the news source (e.g., 'arstechnica')

    Returns:
        A brief summary of the latest news.
    """
    if source not in NEWS_SITES:
        available = ", ".join(NEWS_SITES.keys())
        raise ValueError(f"Source '{source}' is not supported. Available: {available}")

    news_text = await fetch_news(NEWS_SITES[source])
    return news_text


@mcp.tool()
def list_sources() -> list[str]:
    """
    Lists all available news sources.

    Returns:
        List of supported news source names.
    """
    return list(NEWS_SITES.keys())


if __name__ == "__main__":
    mcp.run(transport="stdio")
