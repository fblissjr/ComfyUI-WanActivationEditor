#!/usr/bin/env python3
"""
Simple HTTP server to run the WanVideo Data Flow Explorer web app.
No external dependencies required - uses Python's built-in HTTP server.
"""

import http.server
import socketserver
import os
import sys
import webbrowser
import signal
import argparse
from pathlib import Path

# Configuration
DEFAULT_PORT = 8080
HOST = "127.0.0.1"

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Set the directory to serve files from
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)
    
    def end_headers(self):
        # Add headers to prevent caching during development
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Expires', '0')
        super().end_headers()
    
    def do_GET(self):
        # Serve the main HTML file for the root path
        if self.path == '/':
            self.path = '/wan_dataflow_webapp.html'
        return super().do_GET()
    
    def log_message(self, format, *args):
        # Suppress request logging unless verbose
        if hasattr(self.server, 'verbose') and self.server.verbose:
            super().log_message(format, *args)

def find_free_port(start_port, max_attempts=10):
    """Find a free port starting from start_port"""
    for i in range(max_attempts):
        port = start_port + i
        try:
            with socketserver.TCPServer((HOST, port), None) as test_server:
                return port
        except OSError:
            continue
    return None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nReceived interrupt signal. Shutting down...")
    sys.exit(0)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the WanVideo Data Flow Explorer web app')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, 
                        help=f'Port to run the server on (default: {DEFAULT_PORT})')
    parser.add_argument('--no-browser', action='store_true', 
                        help='Do not open browser automatically')
    parser.add_argument('--verbose', action='store_true',
                        help='Show HTTP request logs')
    args = parser.parse_args()
    
    # Register signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Change to the script's directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check if the HTML file exists
    if not Path("wan_dataflow_webapp.html").exists():
        print("Error: wan_dataflow_webapp.html not found!")
        print("Make sure you're running this script from the tools directory.")
        sys.exit(1)
    
    # Find a free port if the requested one is in use
    port = find_free_port(args.port)
    if port is None:
        print(f"Error: Could not find a free port starting from {args.port}")
        print("Try specifying a different port with --port")
        sys.exit(1)
    
    if port != args.port:
        print(f"Port {args.port} is in use, using port {port} instead")
    
    # Create server with SO_REUSEADDR to avoid "Address already in use" errors
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer((HOST, port), CustomHTTPRequestHandler) as httpd:
            httpd.verbose = args.verbose
            url = f"http://{HOST}:{port}"
            
            print(f"\nWanVideo Data Flow Explorer")
            print("=" * 40)
            print(f"Server running at: {url}")
            print(f"Serving files from: {script_dir}")
            print("\nPress Ctrl+C to stop the server")
            print("=" * 40)
            
            # Try to open the browser
            if not args.no_browser:
                try:
                    webbrowser.open(url)
                    print(f"\nOpened browser at {url}")
                except:
                    print(f"\nPlease open your browser and navigate to: {url}")
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
    except Exception as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)
    finally:
        print("Server stopped.")

if __name__ == "__main__":
    main()