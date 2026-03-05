import google.generativeai as genai
import os

key = os.environ.get("GOOGLE_API_KEY", "")
genai.configure(api_key=key)

models = []
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        models.append(m.name)

# Print sorted
for name in sorted(models):
    print(name)
