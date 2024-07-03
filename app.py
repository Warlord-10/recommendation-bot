import asgiref
from flask import Flask, request, jsonify
from bhashini.detect_language import detect_language
from bhashini.convert_language import convert_language
from llms.recommendation_llm import RecommendationModel 
from asgiref.wsgi import WsgiToAsgi

llm = RecommendationModel()

# Initialize the Flask app
app = Flask(__name__)
asgi_app = WsgiToAsgi(app)


# POST endpoint 
@app.route('/chat', methods=['POST'])
def chat():
    # Get the user's message from the request
    user_message = request.json['prompt']
    
    # Detecting the language
    detectedLang = detect_language(user_message)

    # Converting the language to English
    user_converted_message = convert_language(userText=user_message, locale=detectedLang)

    # Process the user's message and generate a response
    response = llm.querySimpleChromaDB(user_converted_message, top_res=30, use_tags=False)

    # Return the response as a JSON object
    return jsonify({'response': response})

if __name__ == '__main__':
    # recommendationLLM = RecommendationModel()
    import uvicorn
    uvicorn.run("app:asgi_app", host="0.0.0.0", port=5000, log_level="info", reload=False, use_colors=True)
    # app.run(port=5000, debug=False)
    