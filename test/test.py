from app.ai_engine import CrisisEngine
engine = CrisisEngine()
data = engine.get_dashboard_data()
print(data[0])

print("Test passed!")