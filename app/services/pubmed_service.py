"""PubMed API integration service for literature search and retrieval"""
import logging
import time
from typing import Dict, List, Optional
from urllib.parse import quote
import xml.etree.ElementTree as ET
import requests
from app.config.config import Config

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for PubMed API to stay under 4 requests/second limit"""
    
    def __init__(self, requests_per_second: int = 3):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

class PubMedService:
    """Service for integrating PubMed literature search and retrieval"""
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.api_key = getattr(Config, 'PUBMED_API_KEY', None)
        self.email = getattr(Config, 'PUBMED_EMAIL', 'patient-analysis@domain.com')
        self.tool_name = getattr(Config, 'PUBMED_TOOL_NAME', 'PatientAnalysis')
        self.rate_limiter = RateLimiter(requests_per_second=9)
        
        # API endpoints
        self.endpoints = {
            'search': 'esearch.fcgi',
            'summary': 'esummary.fcgi', 
            'fetch': 'efetch.fcgi',
            'link': 'elink.fcgi'
        }
    
    def _build_base_params(self) -> Dict[str, str]:
        """Build base parameters for all API requests"""
        params = {
            'email': self.email,
            'tool': self.tool_name
        }
        if self.api_key:
            params['api_key'] = self.api_key
        return params
    
    def _make_request(self, endpoint: str, params: Dict[str, str]) -> Optional[str]:
        """Make rate-limited request to PubMed API"""
        try:
            self.rate_limiter.wait_if_needed()
            
            url = f"{self.base_url}{self.endpoints[endpoint]}"
            all_params = {**self._build_base_params(), **params}
            
            response = requests.get(url, params=all_params, timeout=30)
            response.raise_for_status()
            
            return response.text
            
        except requests.RequestException as e:
            logger.error(f"PubMed API request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in PubMed request: {str(e)}")
            return None
    
    def search_literature(self, query_terms: List[str], max_results: int = 10) -> List[Dict]:
        """
        Search PubMed literature for given query terms
        
        Args:
            query_terms: List of terms to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of article dictionaries with PMIDs and basic info
        """
        try:
            # Build query from terms
            query = self._build_search_query(query_terms)
            
            # Search for PMIDs
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': str(max_results),
                'retmode': 'xml',
                'sort': 'relevance'
            }
            
            search_response = self._make_request('search', search_params)
            if not search_response:
                return []
            
            # Parse PMIDs from search response
            pmids = self._parse_search_response(search_response)
            
            if not pmids:
                return []
            
            # Get article summaries
            return self.get_article_summaries(pmids)
            
        except Exception as e:
            logger.error(f"Error in literature search: {str(e)}")
            return []
    
    def get_article_summaries(self, pmids: List[str]) -> List[Dict]:
        """
        Get article summaries for given PMIDs
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of article summary dictionaries
        """
        try:
            if not pmids:
                return []
            
            # Join PMIDs for batch request
            pmid_string = ','.join(pmids)
            
            summary_params = {
                'db': 'pubmed',
                'id': pmid_string,
                'retmode': 'xml'
            }
            
            summary_response = self._make_request('summary', summary_params)
            if not summary_response:
                return []
            
            return self._parse_summary_response(summary_response)
            
        except Exception as e:
            logger.error(f"Error getting article summaries: {str(e)}")
            return []
    
    def get_article_details(self, pmids: List[str]) -> List[Dict]:
        """
        Get detailed article information including abstracts
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of detailed article dictionaries
        """
        try:
            if not pmids:
                return []
            
            pmid_string = ','.join(pmids)
            
            fetch_params = {
                'db': 'pubmed',
                'id': pmid_string,
                'retmode': 'xml',
                'rettype': 'abstract'
            }
            
            fetch_response = self._make_request('fetch', fetch_params)
            if not fetch_response:
                return []
            
            return self._parse_fetch_response(fetch_response)
            
        except Exception as e:
            logger.error(f"Error getting article details: {str(e)}")
            return []
    
    def find_evidence_for_condition(self, condition: str, treatment: str = None) -> List[Dict]:
        """
        Find literature evidence for a specific condition and optional treatment
        
        Args:
            condition: Medical condition to search for
            treatment: Optional treatment to include in search
            
        Returns:
            List of relevant articles
        """
        try:
            query_terms = [condition]
            if treatment:
                query_terms.append(treatment)
            
            # Add clinical study filters
            query_terms.extend(['clinical trial', 'case report', 'systematic review'])
            
            return self.search_literature(query_terms, max_results=15)
            
        except Exception as e:
            logger.error(f"Error finding evidence for condition: {str(e)}")
            return []
    
    def get_clinical_trials(self, condition: str) -> List[Dict]:
        """
        Search specifically for clinical trials related to a condition
        
        Args:
            condition: Medical condition to search for
            
        Returns:
            List of clinical trial articles
        """
        try:
            query_terms = [condition, 'clinical trial[Publication Type]']
            return self.search_literature(query_terms, max_results=10)
            
        except Exception as e:
            logger.error(f"Error getting clinical trials: {str(e)}")
            return []
    
    def _build_search_query(self, terms: List[str]) -> str:
        """Build PubMed search query from terms"""
        # Escape special characters and join with AND
        escaped_terms = [quote(term.strip()) for term in terms if term.strip()]
        return ' AND '.join(escaped_terms)
    
    def _parse_search_response(self, xml_response: str) -> List[str]:
        """Parse PMIDs from search response XML"""
        try:
            root = ET.fromstring(xml_response)
            pmids = []
            
            for id_elem in root.findall('.//Id'):
                if id_elem.text:
                    pmids.append(id_elem.text)
            
            return pmids
            
        except ET.ParseError as e:
            logger.error(f"Error parsing search response XML: {str(e)}")
            return []
    
    def _parse_summary_response(self, xml_response: str) -> List[Dict]:
        """Parse article summaries from summary response XML"""
        try:
            root = ET.fromstring(xml_response)
            articles = []
            
            for doc_elem in root.findall('.//DocSum'):
                article = self._extract_article_summary(doc_elem)
                if article:
                    articles.append(article)
            
            return articles
            
        except ET.ParseError as e:
            logger.error(f"Error parsing summary response XML: {str(e)}")
            return []
    
    def _parse_fetch_response(self, xml_response: str) -> List[Dict]:
        """Parse detailed article information from fetch response XML"""
        try:
            root = ET.fromstring(xml_response)
            articles = []
            
            for article_elem in root.findall('.//PubmedArticle'):
                article = self._extract_article_details(article_elem)
                if article:
                    articles.append(article)
            
            return articles
            
        except ET.ParseError as e:
            logger.error(f"Error parsing fetch response XML: {str(e)}")
            return []
    
    def _extract_article_summary(self, doc_elem) -> Optional[Dict]:
        """Extract article summary from XML element"""
        try:
            pmid = doc_elem.find('Id')
            if pmid is None or not pmid.text:
                return None
            
            article = {
                'pmid': pmid.text,
                'title': '',
                'authors': [],
                'journal': '',
                'publication_date': '',
                'study_type': 'unknown'
            }
            
            # Extract title
            title_elem = doc_elem.find('.//Item[@Name="Title"]')
            if title_elem is not None and title_elem.text:
                article['title'] = title_elem.text
            
            # Extract authors
            authors_elem = doc_elem.find('.//Item[@Name="AuthorList"]')
            if authors_elem is not None and authors_elem.text:
                article['authors'] = [authors_elem.text]
            
            # Extract journal
            journal_elem = doc_elem.find('.//Item[@Name="Source"]')
            if journal_elem is not None and journal_elem.text:
                article['journal'] = journal_elem.text
            
            # Extract publication date
            date_elem = doc_elem.find('.//Item[@Name="PubDate"]')
            if date_elem is not None and date_elem.text:
                article['publication_date'] = date_elem.text
            
            return article
            
        except Exception as e:
            logger.error(f"Error extracting article summary: {str(e)}")
            return None
    
    def _extract_article_details(self, article_elem) -> Optional[Dict]:
        """Extract detailed article information from XML element"""
        try:
            # Get PMID
            pmid_elem = article_elem.find('.//PMID')
            if pmid_elem is None or not pmid_elem.text:
                return None
            
            article = {
                'pmid': pmid_elem.text,
                'title': '',
                'abstract': '',
                'authors': [],
                'journal': '',
                'publication_date': '',
                'study_type': 'unknown',
                'keywords': []
            }
            
            # Extract title
            title_elem = article_elem.find('.//ArticleTitle')
            if title_elem is not None and title_elem.text:
                article['title'] = title_elem.text
            
            # Extract abstract
            abstract_elem = article_elem.find('.//AbstractText')
            if abstract_elem is not None and abstract_elem.text:
                article['abstract'] = abstract_elem.text
            
            # Extract authors
            author_elems = article_elem.findall('.//Author')
            authors = []
            for author_elem in author_elems:
                last_name = author_elem.find('LastName')
                first_name = author_elem.find('ForeName')
                if last_name is not None and first_name is not None:
                    authors.append(f"{first_name.text} {last_name.text}")
            article['authors'] = authors
            
            # Extract journal
            journal_elem = article_elem.find('.//Journal/Title')
            if journal_elem is not None and journal_elem.text:
                article['journal'] = journal_elem.text
            
            # Extract publication date
            year_elem = article_elem.find('.//PubDate/Year')
            month_elem = article_elem.find('.//PubDate/Month')
            if year_elem is not None:
                date_str = year_elem.text
                if month_elem is not None:
                    date_str += f"-{month_elem.text}"
                article['publication_date'] = date_str
            
            return article
            
        except Exception as e:
            logger.error(f"Error extracting article details: {str(e)}")
            return None

    def build_clinical_query(self, condition: str, treatment: str = None, study_type: str = None) -> str:
        """
        Build optimized PubMed query for clinical evidence
        
        Args:
            condition: Medical condition
            treatment: Optional treatment/intervention
            study_type: Optional study type filter
            
        Returns:
            Formatted PubMed query string
        """
        query_parts = [condition]
        
        if treatment:
            query_parts.append(treatment)
        
        if study_type:
            if study_type.lower() == 'clinical_trial':
                query_parts.append('clinical trial[Publication Type]')
            elif study_type.lower() == 'systematic_review':
                query_parts.append('systematic review[Publication Type]')
            elif study_type.lower() == 'case_report':
                query_parts.append('case report[Publication Type]')
        
        return ' AND '.join(query_parts)