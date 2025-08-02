from google.cloud import translate_v2 as translate

client = translate.Client.from_service_account_json('seventh-odyssey-467708-m2-d9a0f4ebe813.json')

text = "Hello, world!"

target_language = "vi" 

translation = client.translate(text,target_language=target_language)
print(translation['translatedText'])
