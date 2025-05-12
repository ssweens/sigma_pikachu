import socket
import sys
import threading
import urllib.parse

# Configuration
TARGET_URL = "http://localhost:5000" # Default target URL, change as needed
PORT = 8080 # Default proxy port
BUFFER_SIZE = 4096
CONNECTION_TIMEOUT = 10 # Default connection and read timeout in seconds

def forward_data(source_socket, destination_socket, direction):
    """Forwards data from source to destination and prints it."""
    try:
        source_socket.settimeout(CONNECTION_TIMEOUT) # Set timeout for recv
        while True:
            try:
                data = source_socket.recv(BUFFER_SIZE)
            except socket.timeout:
                print(f"Socket timeout in {direction} while waiting for data.")
                break # Exit loop on timeout
            
            if not data:
                print(f"Connection closed by peer (recv returned 0 bytes) in {direction}.")
                break # Connection closed

            # Print raw data for visibility
            print(f"\n{'='*30}")
            print(f"Forwarding {len(data)} bytes ({direction}):")
            try:
                # Attempt to decode as UTF-8 for readability, but print raw if it fails
                print(data.decode('utf-8', errors='ignore'))
            except Exception:
                print(data) # Print raw bytes if decoding fails
            print(f"{'='*30}\n")

            destination_socket.sendall(data)

    except socket.error as se: # Handle socket-specific errors
        # errno 9: Bad file descriptor
        # errno 104: Connection reset by peer
        # errno 32: Broken pipe
        print(f"Socket error in {direction}: {se}")
    except Exception as e: # Catch other unexpected errors
        print(f"Unexpected error forwarding data ({direction}): {e}")
    finally:
        # Signal that this direction is done sending data.
        # Actual socket closing is handled by handle_client after threads join.
        try:
            if destination_socket.fileno() != -1: # Check if socket is still valid
                destination_socket.shutdown(socket.SHUT_WR)
        except socket.error as se_shutdown:
            # This can happen if the socket is already closed or not connected.
            # (e.g., [Errno 57] Socket is not connected, [Errno 9] Bad file descriptor)
            print(f"Socket error during shutdown(SHUT_WR) for {direction}: {se_shutdown}. Socket might be already closed.")
        except Exception as e_shutdown_generic:
            print(f"Generic error during shutdown(SHUT_WR) for {direction}: {e_shutdown_generic}")
        print(f"Forwarding loop for {direction} ended.")


def handle_client(client_socket):
    """Handle a single client connection by forwarding data to/from the target."""
    target_socket = None # Initialize target_socket to ensure it's defined for the finally block
    try:
        # Parse target host and port from TARGET_URL
        parsed_url = urllib.parse.urlparse(TARGET_URL)
        target_host = parsed_url.hostname
        target_port = parsed_url.port if parsed_url.port else (443 if parsed_url.scheme == 'https' else 80)

        if not target_host:
             print(f"Invalid TARGET_URL: {TARGET_URL}")
             # client_socket will be closed in the finally block
             return

        # Connect to the target server
        target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        target_socket.settimeout(CONNECTION_TIMEOUT) # Set timeout for connect
        try:
            target_socket.connect((target_host, target_port))
        except socket.timeout:
            print(f"Connection to target {target_host}:{target_port} timed out after {CONNECTION_TIMEOUT} seconds.")
            # client_socket will be closed in the finally block
            return
        except socket.error as se_connect:
            print(f"Failed to connect to target {target_host}:{target_port}: {se_connect}")
            # client_socket will be closed in the finally block
            return
        
        target_socket.settimeout(None) # Reset timeout to blocking for subsequent operations if needed, or keep for send/recv
        print(f"Connected to target: {target_host}:{target_port}")

        # Start threads to forward data bidirectionally
        client_to_target_thread = threading.Thread(
            target=forward_data,
            args=(client_socket, target_socket, "Client -> Target")
        )
        target_to_client_thread = threading.Thread(
            target=forward_data,
            args=(target_socket, client_socket, "Target -> Client")
        )

        client_to_target_thread.start()
        target_to_client_thread.start()

        # Wait for both threads to complete before handle_client exits
        client_to_target_thread.join()
        target_to_client_thread.join()
        print("Both forwarding threads have completed.")

    except socket.error as se: # More specific error handling for socket operations
        print(f"Socket error in handle_client (e.g., connection to target failed): {se}")
        # Sockets will be cleaned up in the finally block
    except Exception as e:
        print(f"Error handling client connection: {e}")
        # Sockets will be cleaned up in the finally block
    finally:
        # Ensure sockets are closed regardless of what happened.
        if target_socket:
            try:
                if target_socket.fileno() != -1: # Check if socket descriptor is valid
                    target_socket.close()
            except Exception as e_close_target:
                print(f"Error closing target_socket: {e_close_target}")
        
        if client_socket:
            try:
                if client_socket.fileno() != -1: # Check if socket descriptor is valid
                    client_socket.close()
            except Exception as e_close_client:
                print(f"Error closing client_socket: {e_close_client}")
        print("handle_client finished.")


def run(port=PORT):
    """Starts the raw TCP forwarding proxy server."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allow reusing the address

    server_address = ('', port)
    server_socket.bind(server_address)
    server_socket.listen(5) # Listen for up to 5 pending connections

    print(f"Starting raw TCP proxy server on port {port}...")
    print(f"Forwarding connections to {TARGET_URL}")

    try:
        while True:
            client_socket, client_address = server_socket.accept()
            print(f"Accepted connection from {client_address}")
            # Handle client connection in a new thread
            client_handler = threading.Thread(target=handle_client, args=(client_socket,))
            client_handler.start()

    except KeyboardInterrupt:
        print("Stopping proxy server.")
    finally:
        server_socket.close()

if __name__ == "__main__":
    # Allow changing target URL and port via command line arguments
    if len(sys.argv) > 1:
        TARGET_URL = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            PORT = int(sys.argv[2])
        except ValueError:
            print("Invalid port number provided. Using default port.")
    if len(sys.argv) > 3:
        try:
            CONNECTION_TIMEOUT = int(sys.argv[3])
            if CONNECTION_TIMEOUT <= 0:
                print("Timeout must be a positive integer. Using default timeout.")
                CONNECTION_TIMEOUT = 10 # Reset to default if invalid
        except ValueError:
            print("Invalid timeout value provided. Using default timeout.")


    run(port=PORT)
