import random
import uuid
from typing import List, Dict
from faker import Faker
from core.logger import get_logger


logger = get_logger(__name__)
fake = Faker()


class DataGenerator:
    """Generate synthetic test data for AI workflows."""
    
    @staticmethod
    def generate_trace_id() -> str:
        """Generate a valid trace ID."""
        return uuid.uuid4().hex
    
    @staticmethod
    def generate_span_id() -> str:
        """Generate a valid span ID."""
        return uuid.uuid4().hex[:16]
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate a session ID."""
        return f"session_{uuid.uuid4().hex[:12]}"
    
    @staticmethod
    def generate_prompts(count: int = 10) -> List[str]:
        """
        Generate test prompts.
        
        Args:
            count: Number of prompts to generate
        
        Returns:
            List of prompt strings
        """
        templates = [
            "Explain {topic} in simple terms",
            "What is the difference between {topic1} and {topic2}?",
            "Write a {length} summary about {topic}",
            "How do I {action} using {tool}?",
            "Create a list of {count} {items}",
            "Translate '{text}' to {language}",
            "Analyze the sentiment of: {text}",
            "Generate code to {action} in {language}"
        ]
        
        topics = ["AI", "machine learning", "Python", "databases", "APIs", "cloud computing"]
        actions = ["deploy", "configure", "optimize", "debug", "test"]
        tools = ["Docker", "Kubernetes", "Terraform", "Jenkins", "Git"]
        languages = ["Spanish", "French", "German", "Japanese", "Chinese"]
        
        prompts = []
        for _ in range(count):
            template = random.choice(templates)
            prompt = template.format(
                topic=random.choice(topics),
                topic1=random.choice(topics),
                topic2=random.choice(topics),
                length=random.choice(["brief", "detailed", "comprehensive"]),
                action=random.choice(actions),
                tool=random.choice(tools),
                count=random.randint(3, 10),
                items=random.choice(["tips", "examples", "best practices", "tools"]),
                text=fake.sentence(),
                language=random.choice(languages)
            )
            prompts.append(prompt)
        
        logger.info(f"Generated {count} test prompts")
        return prompts
    
    @staticmethod
    def generate_conversation(turns: int = 3) -> List[Dict[str, str]]:
        """
        Generate a multi-turn conversation.
        
        Args:
            turns: Number of conversation turns
        
        Returns:
            List of message dictionaries
        """
        conversation = []
        
        for i in range(turns):
            # User message
            conversation.append({
                "role": "user",
                "content": fake.sentence()
            })
            
            # Assistant response
            conversation.append({
                "role": "assistant",
                "content": fake.paragraph()
            })
        
        logger.info(f"Generated conversation with {turns} turns")
        return conversation
    
    @staticmethod
    def generate_agent_names(count: int = 5) -> List[str]:
        """Generate agent names."""
        prefixes = ["Research", "Analysis", "Planning", "Execution", "Review"]
        suffixes = ["Agent", "Specialist", "Coordinator", "Manager", "Assistant"]
        
        names = []
        for i in range(count):
            prefix = prefixes[i % len(prefixes)]
            suffix = suffixes[i % len(suffixes)]
            names.append(f"{prefix} {suffix}")
        
        return names
    
    @staticmethod
    def generate_test_user() -> Dict[str, str]:
        """Generate test user data."""
        return {
            "email": fake.email(),
            "name": fake.name(),
            "user_id": f"user_{uuid.uuid4().hex[:8]}"
        }
