"""Mock LLM for testing without API calls"""


class MockLLM:
    """Mock Language Model for testing"""
    
    def __init__(self, model_name="mock-gpt-4"):
        self.model_name = model_name
        self.call_count = 0
    
    def generate(self, prompt: str) -> str:
        """Generate mock response"""
        self.call_count += 1
        return f"Mock response to: {prompt[:50]}..."
    
    def chat(self, messages: list) -> dict:
        """Mock chat completion"""
        self.call_count += 1
        return {
            "id": f"mock-{self.call_count}",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"Mock response to {len(messages)} messages"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        }
