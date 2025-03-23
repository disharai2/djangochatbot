from django.shortcuts import render
from django.http import JsonResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
# Load the DialoGPT model and tokenizer (runs once when server starts)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def ask_model(message):
    # Encode user message
    input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')
    # Generate response
    chat_history_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.75
    )
    # Decode and return the response
    reply = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return reply

def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        response_text = ask_model(message)
        return JsonResponse({'message': message, 'response': response_text})
    return render(request, 'chatbot.html')
