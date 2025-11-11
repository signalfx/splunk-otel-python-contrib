"""Mock tools for agent testing"""


class MockTools:
    """Mock tools for testing agents"""
    
    def search_flights(self, origin: str, destination: str, date: str) -> dict:
        """Mock flight search"""
        return {
            "flights": [
                {"airline": "MockAir", "price": 299, "departure": "10:00"},
                {"airline": "TestFly", "price": 349, "departure": "14:00"}
            ]
        }
    
    def search_hotels(self, location: str, checkin: str, checkout: str) -> dict:
        """Mock hotel search"""
        return {
            "hotels": [
                {"name": "Mock Hotel", "price": 150, "rating": 4.5},
                {"name": "Test Inn", "price": 120, "rating": 4.0}
            ]
        }
    
    def search_activities(self, location: str) -> dict:
        """Mock activity search"""
        return {
            "activities": [
                {"name": "City Tour", "price": 50, "duration": "3 hours"},
                {"name": "Museum Visit", "price": 25, "duration": "2 hours"}
            ]
        }
