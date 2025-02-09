from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import traceback
from chatbot.lang import chatbot

# Create your views here.

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def chat_api(request):
    if request.method == 'POST':
        try:
            print("Initializing chatbot...")
            if not chatbot.initialize():
                raise Exception("Failed to initialize chatbot")
            
            data = json.loads(request.body)
            question = data.get('message')
            print(f"Received question: {question}")
            
            if not question:
                return JsonResponse({'error': 'No message provided'}, status=400)
            
            # 챗봇 응답 가져오기
            answer = chatbot.get_response(question)
            
            return JsonResponse({'response': answer})
            
        except Exception as e:
            print("Error occurred in chat_api:")
            print(traceback.format_exc())
            return JsonResponse({
                'error': str(e),
                'traceback': traceback.format_exc()
            }, status=500)
            
    return JsonResponse({'error': 'Invalid request'}, status=400)
