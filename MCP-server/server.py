import logging
import requests
from mcp.server.fastmcp import FastMCP
# from serpapi import GoogleSearch
from config import SERVER_NAME,  SERPAPI_KEY , WIKI_USER_AGENT
import wikipediaapi
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the MCP server
mcp = FastMCP(name=SERVER_NAME)


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/91.0.4472.124 Safari/537.36"
}

# wiki = wikipediaapi.Wikipedia('WikipediaMCP (nexangan@gmail.com)', 'en')
wiki = wikipediaapi.Wikipedia(WIKI_USER_AGENT, 'en')
WIKI_API_ENDPOINT = "https://en.wikipedia.org/w/api.php"


@mcp.tool()
def fetch_wikipedia_page(query: str, summary_length: int = 1500) -> dict:
    """
    Fetch Wikipedia page matching query. Returns title, summary, URL.
    Args:
        query: Wikipedia page title or search term
        summary_length: Number of characters in the summary
    Returns:
        Dict with result string containing formatted info
    """
    page = wiki.page(query.title())
    if page.exists():
        summary = page.summary if summary_length is None else page.summary[:summary_length]
        return {"result": f"**{page.title}**\n{summary}\nURL: {page.fullurl}"}
    else:
        return {"result": f"No Wikipedia page found for '{query}'."}


@mcp.tool()
def search_wikipedia_relevance(query: str, max_results: int = 3) -> dict:
    """
    Search Wikipedia articles ranked by relevance, fetch summaries & URLs.
    Performs this in a single, efficient API call.
    Args:
        query: Search string
        max_results: How many top articles to return
    Returns:
        Dict with results list
    """
    # params = {
    #     'action': 'query',
    #     'list': 'search',
    #     'srsearch': query,
    #     'format': 'json',
    #     'srlimit': max_results
    # }
    params = {
        'action': 'query',
        'format': 'json',
        'generator': 'search',     # Use search as a generator
        'gsrsearch': query,        # The search query
        'gsrlimit': max_results,   # Max search results to "generate"
        'prop': 'extracts|info',   # Get properties for the generated pages
        'exintro': True,           # Get the intro "extract" only
        'explaintext': True,       # Get it as plain text
        'inprop': 'url',           # Also get the full URL
        'exchars': 1500,           # Limit the extract to 1500 chars
        'indexpageids': True       # Helps order the results
    }
    try:
        r = requests.get(WIKI_API_ENDPOINT, params=params, headers=headers)
        if not r.headers.get('Content-Type','').startswith('application/json'):
                return {"result": "Wikipedia API returned non-JSON response."}
                
        data = r.json()

        results = []
        pages = data['query']['pages']
        
    #     for item in data.get('query', {}).get('search', []):
    #         title = item['title']
    #         # Fetch page extract (summary)
    #         detail_params = {
    #             'action': 'query',
    #             'prop': 'extracts|info',
    #             'exintro': True,
    #             'explaintext': True,
    #             'inprop': 'url',
    #             'titles': title,
    #             'format': 'json'
    #         }
    #         d_resp = requests.get(WIKI_API_ENDPOINT, params=detail_params, headers=headers)
    #         if d_resp.headers.get('Content-Type','').startswith('application/json'):
    #             d = d_resp.json()
    #             page_info = next(iter(d['query']['pages'].values()))
    #             extract = page_info.get('extract', '')[:1500]
    #             url = page_info.get('fullurl', '')
    #         else:
    #             extract = "[Could not fetch page summary]"
    #             url = ""
    #         results.append({
    #             'title': title,
    #             'summary': extract,
    #             'url': url
    #         })
    #         pass
    #     if not results:
    #         return {"result": "No relevant articles found."}
    #     return {"result": results}
    # except Exception as e:
    #     return {"result": f"Error occurred: {str(e)}"}
        for page_id in data['query'].get('pageids', []):
            page = pages.get(page_id, {})
            if not page:
                continue
                
            results.append({
                'title': page.get('title', 'N/A'),
                'summary': page.get('extract', ''),
                'url': page.get('fullurl', '')
            })

        if not results:
            return {"result": "No relevant articles found."}
            
        return {"result": results}
        
    except requests.RequestException as e:
        return {"result": f"HTTP Error occurred: {str(e)}"}
    except Exception as e:
        return {"result": f"An unexpected error occurred: {str(e)}"}


@mcp.tool()
def fetch_wikipedia_sections(page_title: str) -> dict:
    """
    List all section titles for a Wikipedia page.
    Args:
        page_title: Exact Wikipedia page name
    Returns:
        Dict with list of section titles
    """
    params = {
        'action': 'parse',
        'page': page_title,
        'format': 'json',
        'prop': 'sections'
    }
    r = requests.get(WIKI_API_ENDPOINT, params=params, headers = headers)
    data = r.json()
    sections = [sec['line'] for sec in data.get('parse', {}).get('sections', [])]
    if not sections:
        return {"result": f"No sections found for '{page_title}'."}
    return {"result": sections}



if __name__ == "__main__":
    mcp.run(transport="stdio")