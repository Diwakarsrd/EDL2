import json
import os
import sys

# Add the parent directory to the path so we can import our Flask app
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import the Flask app
try:
    from app_vercel import app
except ImportError:
    # If that doesn't work, try direct import
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from app_vercel import app

def handler(event, context):
    # Create a test client
    with app.test_client() as client:
        # Get the request details
        path = event.get('path', '/')
        method = event.get('httpMethod', 'GET')
        headers = event.get('headers', {})
        query_string = event.get('queryStringParameters', {})
        body = event.get('body', '')
        
        # Make the request to our Flask app
        response = client.open(
            path=path,
            method=method,
            headers=headers,
            data=body,
            query_string=query_string
        )
        
        # Return the response in the format Netlify expects
        return {
            'statusCode': response.status_code,
            'headers': {
                'Content-Type': 'application/json' if response.is_json else response.content_type
            },
            'body': response.get_data(as_text=True)
        }