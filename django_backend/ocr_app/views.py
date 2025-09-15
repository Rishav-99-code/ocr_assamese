from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import sys
import os
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interface import ocr_function


@csrf_exempt
def ocr_api_view(request):
    if request.method == 'POST':
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image file provided'}, status=400)
        image_file = request.FILES['image']

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, image_file.name )
        with open(temp_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        try:
            recognized_text = ocr_function(temp_path)
            os.remove(temp_path)
        except Exception as e:
            os.remove(temp_path)
            return JsonResponse({'error': f"Processing failed: {e}"}, status=500)
        return JsonResponse({'text': recognized_text})
    return JsonResponse({'error': 'Method not allowed'}, status=405)