import json

from typing import Text
import requests


def convert_language(userText: str, locale: str = "hi") -> Text:
    """ 
        Translate's the text from locale language to English.
        Default Locale in hi which is Hindi.
    """

    if locale == "en":
        return userText
    
    try:
        endpoint = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
        headers = {
                "Authorization":"KIqYRYKJWTmMMta9y1KLSxTfIPP-lARCsKEMYlXYo7H66wQD1VaBveulRWuaZUbU",
                "Content-Type":"application/json"
        }

        obj = {
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": locale,
                            "targetLanguage": "en"
                        },
                        "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4"
                    }   
                }
            ],
            "inputData": {
                "input": [
                    {
                        "source": f"{userText}"
                    }
                ],
                "audio": [
                    {
                        "audioContent": "null"
                    }
                ]
            }
        }

        response = requests.post(endpoint, json=obj, headers=headers)
        response = response.json()

        print(response["pipelineResponse"][0]["output"][0]["target"])
        
        return response["pipelineResponse"][0]["output"][0]["target"]
    except Exception as e:
        print("Exception in translation of language", e)
        return userText


    # """ 
    #     Translate's the text into locale language.
    #     Default Locale in en which is English
    # """
    # print("Convert Language for text: ", text, locale)
    # if(locale == "en" or locale == None):
    #     return text
    # text = text.replace("&", "and")
    # redis_client = RedisClient().connection

    # locale_data = redis_client.get(RedisPreferences.LANGUAGE_CONVERSTION.value)
    
    # if(locale_data):
    #     locale_data = json.loads(locale_data)

    # if type(locale_data) is dict:
    #     try:
    #         if text in locale_data[locale].keys():
    #             return locale_data[locale][text]
    #     except:
    #         print("error in finding the locale variable, hence using AI")

    # try:        
    #     parent = f"projects/plutosone-prod-project/locations/global"
    #     translate_client = translate.TranslationServiceClient()
    #     response = translate_client.translate_text(
    #         request={
    #             "parent": parent,
    #             "contents": [text],
    #             "mime_type": "text/plain",  # mime types: text/plain, text/html
    #             "source_language_code": "en",
    #             "target_language_code": locale,
    #         }
    #     )
    #     print("translate_text", response, type(response))
        
    #     # V2 Method:
    #     # translated_text = translate_client.translate(text, locale)
    #     # Response: {'translatedText': 'मनप्रीत सिंह', 'detectedSourceLanguage': 'gu', 'input': 'Manpreet Singh'}
    #     # print(translated_text)


    #     translated_text = text
    #     for translation in response.translations:
    #         translated_text = translation.translated_text
    #         print(f"Translated text: {translation.translated_text}")

    #         if(type(locale_data) is dict):
    #             print("inside if")
    #             # locale_data[locale][text] = translated_text["translatedText"]
    #             locale_data[locale][text] = translation.translated_text
    #         else:
    #             print("inside else")
    #             locale_data = {
    #                 locale: {
    #                     # text: translated_text["translatedText"]
    #                     text: translation.translated_text
    #                 }
    #             }
        
    #     redis_client.set(RedisPreferences.LANGUAGE_CONVERSTION.value, json.dumps(locale_data))


    #     # return translated_text["translatedText"]
    #     return translated_text
    # except Exception as e:
    #     print("Exception in translation of language", e)
    #     return text