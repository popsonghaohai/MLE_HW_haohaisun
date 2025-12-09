"""
Simple Test Script for Function Calling
Tests the core function calling logic without GUI
"""

import json
import re
from sympy import sympify


# ============================================================================
# TOOL FUNCTIONS
# ============================================================================

def search_arxiv(query: str) -> str:
    """Search arXiv for papers (simulated for testing)"""
    print(f"🔍 [TOOL CALLED] search_arxiv(query='{query}')")
    
    try:
        import arxiv
        search = arxiv.Search(
            query=query,
            max_results=1,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = list(search.results())
        
        if results:
            paper = results[0]
            return f"""Found: {paper.title}
Authors: {', '.join([a.name for a in paper.authors[:2]])}
Published: {paper.published.strftime('%Y-%m-%d')}
Summary: {paper.summary[:200]}..."""
    except:
        pass
    
    return f"[Simulated arXiv result for '{query}']: Research papers about {query}"


def calculate(expression: str) -> str:
    """Calculate mathematical expression"""
    print(f"🧮 [TOOL CALLED] calculate(expression='{expression}')")
    
    try:
        result = sympify(expression)
        if result.is_number:
            evaluated = float(result.evalf())
            if evaluated == int(evaluated):
                return f"The result is: {int(evaluated)}"
            else:
                return f"The result is: {evaluated}"
        else:
            return f"The result is: {result}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def route_llm_output(llm_output: str) -> tuple[str, bool]:
    """Route LLM output to tools or return text"""
    
    # Try to extract JSON
    json_pattern = r'```json\s*(.*?)\s*```'
    json_match = re.search(json_pattern, llm_output, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
    else:
        json_pattern2 = r'\{[\s\S]*?"function"[\s\S]*?\}'
        json_match2 = re.search(json_pattern2, llm_output)
        json_str = json_match2.group(0) if json_match2 else llm_output
    
    try:
        output = json.loads(json_str.strip())
        func_name = output.get("function")
        args = output.get("arguments", {})
        
        print(f"\n✅ JSON Function Call Detected:")
        print(f"   Function: {func_name}")
        print(f"   Arguments: {args}\n")
        
        if func_name == "search_arxiv":
            query = args.get("query", "")
            return search_arxiv(query), True
            
        elif func_name == "calculate":
            expr = args.get("expression", "")
            return calculate(expr), True
            
        else:
            return f"Error: Unknown function '{func_name}'", True
            
    except (json.JSONDecodeError, TypeError, AttributeError):
        print("💬 Normal Text Response (no function call)")
        return llm_output, False


# ============================================================================
# SIMULATED LLM RESPONSES FOR TESTING
# ============================================================================

test_cases = [
    {
        "name": "Calculate - Simple",
        "user_input": "What's 15 times 23?",
        "llm_output": '{"function": "calculate", "arguments": {"expression": "15*23"}}'
    },
    {
        "name": "Calculate - Complex",
        "user_input": "Calculate sqrt(144) + 5",
        "llm_output": '{"function": "calculate", "arguments": {"expression": "sqrt(144) + 5"}}'
    },
    {
        "name": "Search arXiv",
        "user_input": "Find papers about quantum computing",
        "llm_output": '{"function": "search_arxiv", "arguments": {"query": "quantum computing"}}'
    },
    {
        "name": "Search arXiv - Machine Learning",
        "user_input": "Search for research on transformers",
        "llm_output": '{"function": "search_arxiv", "arguments": {"query": "transformers neural networks"}}'
    },
    {
        "name": "Normal Conversation",
        "user_input": "Hello, how are you?",
        "llm_output": "Hello! I'm doing great, thank you for asking. How can I help you today?"
    },
    {
        "name": "JSON in Markdown",
        "user_input": "What's 100 divided by 4?",
        "llm_output": '''```json
{"function": "calculate", "arguments": {"expression": "100/4"}}
```'''
    },
    {
        "name": "Calculate - Division with Decimals",
        "user_input": "What's 22 divided by 7?",
        "llm_output": '{"function": "calculate", "arguments": {"expression": "22/7"}}'
    },
]


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_tests():
    """Run all test cases"""
    print("=" * 70)
    print("🧪 FUNCTION CALLING TEST SUITE")
    print("=" * 70)
    print()
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"TEST {i}: {test['name']}")
        print("=" * 70)
        
        print(f"\n👤 User Input:")
        print(f"   {test['user_input']}")
        
        print(f"\n🤖 LLM Output (raw):")
        print(f"   {test['llm_output'][:100]}{'...' if len(test['llm_output']) > 100 else ''}")
        
        print(f"\n⚙️  Processing...")
        
        # Route the output
        response, was_function_call = route_llm_output(test['llm_output'])
        
        print(f"\n📤 Final Response:")
        print(f"   {response}")
        
        if was_function_call:
            print(f"\n🎯 Result: FUNCTION CALLED ✅")
        else:
            print(f"\n🎯 Result: NORMAL TEXT RESPONSE")
        
        print()
    
    print("=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode():
    """Interactive testing mode"""
    print("\n" + "=" * 70)
    print("🎮 INTERACTIVE FUNCTION CALLING TEST")
    print("=" * 70)
    print("\nSimulate LLM responses to test the routing logic.")
    print("Enter LLM output (JSON for function calls, text for normal responses)")
    print("Type 'quit' to exit\n")
    
    while True:
        print("-" * 70)
        user_input = input("\n👤 User Query (for context): ").strip()
        
        if user_input.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        
        llm_output = input("🤖 LLM Output (JSON or text): ").strip()
        
        if not llm_output:
            print("⚠️  Empty output, skipping...")
            continue
        
        print("\n⚙️  Processing...")
        response, was_function_call = route_llm_output(llm_output)
        
        print(f"\n📤 Final Response:")
        print(f"   {response}")
        
        if was_function_call:
            print(f"\n🎯 Result: FUNCTION CALLED ✅")
        else:
            print(f"\n🎯 Result: NORMAL TEXT RESPONSE")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n🚀 Function Calling Test Script\n")
    print("Choose mode:")
    print("  1. Run automated tests")
    print("  2. Interactive mode")
    print("  3. Run both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        run_tests()
    elif choice == "2":
        interactive_mode()
    elif choice == "3":
        run_tests()
        interactive_mode()
    else:
        print("Invalid choice, running automated tests...")
        run_tests()
