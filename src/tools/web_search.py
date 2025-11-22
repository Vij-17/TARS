"""Web Search Tool"""

import requests
import os


class WebSearchTool:
    """
    Web search using SerpAPI or similar service.
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('SERPAPI_KEY')
        self.base_url = "https://serpapi.com/search"
        
    def search(self, query, num_results=5):
        """
        Search the web.
        
        Args:
            query: Search query string
            num_results: Number of results to return
        
        Returns:
            list of result dicts with title, snippet, url
        """
        
        if not self.api_key:
            # Return mock results if no API key
            return self._mock_search(query, num_results)
        
        try:
            params = {
                'q': query,
                'api_key': self.api_key,
                'num': num_results
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            for item in data.get('organic_results', [])[:num_results]:
                results.append({
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'url': item.get('link', '')
                })
            
            return results
            
        except Exception as e:
            print(f"Web search error: {e}")
            return self._mock_search(query, num_results)
    
    def _mock_search(self, query, num_results):
        """Mock search results for testing"""
        return [
            {
                'title': f'Result {i+1} for: {query}',
                'snippet': f'This is a mock search result snippet for query: {query}',
                'url': f'https://example.com/result{i+1}'
            }
            for i in range(num_results)
        ]
    
    def __call__(self, query):
        """Make callable like a function"""
        results = self.search(query if isinstance(query, str) else str(query))
        
        # Format as text
        if results:
            formatted = "\n\n".join([
                f"**{r['title']}**\n{r['snippet']}\nURL: {r['url']}"
                for r in results
            ])
            return formatted
        
        return "No results found"