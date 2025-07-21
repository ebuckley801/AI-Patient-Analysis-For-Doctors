#!/usr/bin/env python3
"""
Test the Claude ICD Matcher service
"""

import sys
import os
import time

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from app.services.claude_icd_matcher import create_claude_icd_matcher

def test_claude_icd_basic():
    """Test basic Claude ICD functionality"""
    print("🤖 Testing Claude ICD Matcher")
    print("=" * 40)
    
    matcher = create_claude_icd_matcher()
    
    # Test cases that were problematic with vector search
    test_cases = [
        {'entity': 'hypertension', 'type': 'condition', 'expected_prefix': 'I'},
        {'entity': 'diabetes', 'type': 'condition', 'expected_prefix': 'E'},
        {'entity': 'chest pain', 'type': 'symptom', 'expected_prefix': 'R'},
        {'entity': 'shortness of breath', 'type': 'symptom', 'expected_prefix': 'R'},
        {'entity': 'heart attack', 'type': 'condition', 'expected_prefix': 'I'}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        entity = test_case['entity']
        entity_type = test_case['type']
        expected_prefix = test_case['expected_prefix']
        
        print(f"\n🧪 Test {i}: '{entity}' ({entity_type})")
        
        start_time = time.time()
        suggestions = matcher.suggest_icd_codes(entity, entity_type, top_k=3)
        call_time = (time.time() - start_time) * 1000
        
        print(f"⏱️  Response time: {call_time:.0f}ms")
        print(f"📋 Suggestions: {len(suggestions)}")
        
        if suggestions:
            print("🎯 Top suggestions:")
            for j, suggestion in enumerate(suggestions[:3], 1):
                code = suggestion.get('code', 'N/A')
                description = suggestion.get('description', 'N/A')
                confidence = suggestion.get('confidence', 0)
                validated = suggestion.get('validated', False)
                reasoning = suggestion.get('reasoning', '')[:60]
                
                status = "✅" if validated else "⚠️"
                print(f"   {j}. {status} {code}: {description}")
                print(f"      Confidence: {confidence:.3f} | {reasoning}...")
                
                # Check if it starts with expected category
                if code.startswith(expected_prefix):
                    print(f"      ✅ Correct ICD category ({expected_prefix})")
                else:
                    print(f"      ⚠️ Expected category {expected_prefix}, got {code[0]}")
        else:
            print("❌ No suggestions returned")

def test_claude_icd_caching():
    """Test caching performance"""
    print(f"\n🚀 Testing Claude ICD Caching")
    print("=" * 30)
    
    matcher = create_claude_icd_matcher()
    
    entity = "diabetes mellitus"
    
    # First call (should hit API)
    print("First call (API)...")
    start_time = time.time()
    results1 = matcher.suggest_icd_codes(entity, 'condition')
    first_call_time = (time.time() - start_time) * 1000
    
    # Second call (should hit cache)
    print("Second call (cache)...")
    start_time = time.time()
    results2 = matcher.suggest_icd_codes(entity, 'condition')
    cached_call_time = (time.time() - start_time) * 1000
    
    print(f"📊 Performance:")
    print(f"   First call: {first_call_time:.0f}ms")
    print(f"   Cached call: {cached_call_time:.0f}ms")
    if cached_call_time > 0:
        print(f"   Speedup: {first_call_time/cached_call_time:.1f}x faster")
    else:
        print(f"   Speedup: Instant (cached)")
    
    # Verify same results
    if len(results1) == len(results2):
        print(f"✅ Cache returned same {len(results1)} results")
    else:
        print(f"❌ Cache mismatch: {len(results1)} vs {len(results2)}")

def test_claude_icd_context():
    """Test context-aware suggestions"""
    print(f"\n🎯 Testing Context-Aware Suggestions")
    print("=" * 40)
    
    matcher = create_claude_icd_matcher()
    
    # Test with different contexts
    base_entity = "chest pain"
    
    contexts = [
        {'severity': 'severe', 'temporal': 'acute'},
        {'severity': 'mild', 'temporal': 'chronic'},
        {'age': 65, 'severity': 'moderate'}
    ]
    
    for i, context in enumerate(contexts, 1):
        print(f"\n🔍 Context {i}: {context}")
        
        suggestions = matcher.suggest_icd_codes(base_entity, 'symptom', context=context)
        
        if suggestions:
            best = suggestions[0]
            print(f"   Best suggestion: {best.get('code')} ({best.get('confidence', 0):.3f})")
            print(f"   Reasoning: {best.get('reasoning', 'N/A')[:80]}...")
        else:
            print("   No suggestions")

def test_claude_icd_stats():
    """Test statistics and performance tracking"""
    print(f"\n📊 Testing Statistics Tracking")
    print("=" * 32)
    
    matcher = create_claude_icd_matcher()
    
    # Make a few calls
    test_entities = ['diabetes', 'hypertension', 'diabetes']  # Note: diabetes repeated for cache test
    
    for entity in test_entities:
        matcher.suggest_icd_codes(entity, 'condition', top_k=2)
    
    # Get stats
    stats = matcher.get_stats()
    
    print("📈 Performance Statistics:")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   API calls: {stats['api_calls']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"   Avg response time: {stats['avg_response_time_ms']:.0f}ms")
    print(f"   Cached entities: {stats['cached_entities']}")

if __name__ == '__main__':
    test_claude_icd_basic()
    test_claude_icd_caching()
    test_claude_icd_context()
    test_claude_icd_stats()