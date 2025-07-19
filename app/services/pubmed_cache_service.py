import json
import hashlib
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from app.services.supabase_service import SupabaseService

logger = logging.getLogger(__name__)

class PubMedCacheService:
    """Intelligent caching for PubMed queries to reduce API calls"""
    
    def __init__(self):
        self.supabase = SupabaseService()
        self.cache_ttl = 86400 * 7  # 7 days for literature
        self.table_name = 'pubmed_cache'
    
    def get_cached_search(self, query_hash: str) -> Optional[List[Dict]]:
        """
        Get cached search results if available and not expired
        
        Args:
            query_hash: Hash of the search query
            
        Returns:
            Cached results or None if not available/expired
        """
        try:
            result = self.supabase.client.table(self.table_name).select('*').eq('query_hash', query_hash).execute()
            
            if not result.data:
                return None
            
            cache_entry = result.data[0]
            
            # Check if cache entry is expired
            cached_at = datetime.fromisoformat(cache_entry['cached_at'].replace('Z', '+00:00'))
            expiry_time = cached_at + timedelta(seconds=self.cache_ttl)
            
            if datetime.now(cached_at.tzinfo) > expiry_time:
                # Cache expired, delete entry
                self._delete_cache_entry(query_hash)
                return None
            
            # Update access count and last accessed
            self._update_cache_access(query_hash)
            
            return json.loads(cache_entry['results'])
            
        except Exception as e:
            logger.error(f"Error retrieving cached search: {str(e)}")
            return None
    
    def cache_search_results(self, query_hash: str, query: str, results: List[Dict]) -> bool:
        """
        Cache search results for future use
        
        Args:
            query_hash: Hash of the search query
            query: Original query string
            results: Search results to cache
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            cache_data = {
                'query_hash': query_hash,
                'query': query,
                'results': json.dumps(results),
                'result_count': len(results),
                'cached_at': datetime.utcnow().isoformat(),
                'last_accessed': datetime.utcnow().isoformat(),
                'access_count': 1
            }
            
            # Upsert cache entry
            result = self.supabase.client.table(self.table_name).upsert(cache_data).execute()
            
            return len(result.data) > 0
            
        except Exception as e:
            logger.error(f"Error caching search results: {str(e)}")
            return False
    
    def invalidate_outdated_cache(self) -> int:
        """
        Remove expired cache entries
        
        Returns:
            Number of entries removed
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(seconds=self.cache_ttl)
            cutoff_iso = cutoff_time.isoformat()
            
            # Delete expired entries
            result = self.supabase.client.table(self.table_name).delete().lt('cached_at', cutoff_iso).execute()
            
            deleted_count = len(result.data) if result.data else 0
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired cache entries")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}")
            return 0
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            # Get total cache entries
            total_result = self.supabase.client.table(self.table_name).select('count', count='exact').execute()
            total_entries = total_result.count or 0
            
            # Get recent entries (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            recent_result = self.supabase.client.table(self.table_name).select('*').gte('last_accessed', cutoff_time.isoformat()).execute()
            recent_entries = len(recent_result.data) if recent_result.data else 0
            
            # Calculate total access count
            all_entries = self.supabase.client.table(self.table_name).select('access_count').execute()
            total_accesses = sum(entry['access_count'] for entry in all_entries.data) if all_entries.data else 0
            
            # Get expired entries count
            expired_cutoff = datetime.utcnow() - timedelta(seconds=self.cache_ttl)
            expired_result = self.supabase.client.table(self.table_name).select('count', count='exact').lt('cached_at', expired_cutoff.isoformat()).execute()
            expired_entries = expired_result.count or 0
            
            return {
                'total_entries': total_entries,
                'recent_entries_24h': recent_entries,
                'total_accesses': total_accesses,
                'expired_entries': expired_entries,
                'cache_hit_ratio': round(total_accesses / max(total_entries, 1), 3),
                'cache_ttl_hours': self.cache_ttl / 3600,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {str(e)}")
            return {
                'total_entries': 0,
                'recent_entries_24h': 0,
                'total_accesses': 0,
                'expired_entries': 0,
                'cache_hit_ratio': 0,
                'cache_ttl_hours': self.cache_ttl / 3600,
                'last_updated': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def _delete_cache_entry(self, query_hash: str) -> bool:
        """Delete a specific cache entry"""
        try:
            result = self.supabase.client.table(self.table_name).delete().eq('query_hash', query_hash).execute()
            return len(result.data) > 0 if result.data else False
        except Exception as e:
            logger.error(f"Error deleting cache entry: {str(e)}")
            return False
    
    def _update_cache_access(self, query_hash: str) -> bool:
        """Update access count and last accessed time for cache entry"""
        try:
            # Get current access count
            result = self.supabase.client.table(self.table_name).select('access_count').eq('query_hash', query_hash).execute()
            
            if not result.data:
                return False
            
            current_count = result.data[0]['access_count']
            
            # Update access count and last accessed time
            update_data = {
                'access_count': current_count + 1,
                'last_accessed': datetime.utcnow().isoformat()
            }
            
            update_result = self.supabase.client.table(self.table_name).update(update_data).eq('query_hash', query_hash).execute()
            
            return len(update_result.data) > 0 if update_result.data else False
            
        except Exception as e:
            logger.error(f"Error updating cache access: {str(e)}")
            return False

    @staticmethod
    def generate_query_hash(query: str, max_results: int = 10) -> str:
        """
        Generate hash for a query to use as cache key
        
        Args:
            query: Search query string
            max_results: Maximum results requested
            
        Returns:
            Hash string for cache key
        """
        cache_key = f"{query}_{max_results}"
        return hashlib.md5(cache_key.encode()).hexdigest()