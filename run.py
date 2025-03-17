"""
Run script for the Enhanced AI Tool with web interface and audio output

This script runs the Flask web application for the Enhanced AI Tool.
"""

from app import app

if __name__ == "__main__":
    print("Starting Enhanced AI Tool Web Application...")
    print("Open your browser and navigate to http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
