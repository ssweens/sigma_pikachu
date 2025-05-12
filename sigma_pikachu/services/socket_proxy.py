import asyncio
import logging
import json # Import json for parsing the body

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

async def handle_client(reader, writer, downstream_host, downstream_port, model_callback=None):
    """Handles a single client connection."""
    client_address = writer.get_extra_info('peername')
    logging.info(f"Accepted connection from {client_address}")

    downstream_reader = None
    downstream_writer = None
    full_request_data = b"" # Initialize full request data

    try:
        logging.debug("handle_client: Reading initial data from client...")
        # Read initial data to peek at headers
        initial_data = await reader.read(4096) # Read up to 4KB to get headers
        if not initial_data:
            logging.info("Client closed connection immediately.")
            return # Client closed connection
        logging.debug(f"handle_client: Initial data read ({len(initial_data)} bytes). Peeking for headers...")
        logging.debug(f"Initial data read: {initial_data[:200]}...") # Log first 200 bytes for brevity

        request_body = b""
        model_name = None
        content_length = 0
        body_start_index = -1 # Initialize body start index

        data_str = initial_data.decode('utf-8', errors='ignore')
        headers_end = data_str.find('\r\n\r\n')

        if data_str.startswith('POST ') and headers_end != -1:
            logging.debug("handle_client: Detected POST request with headers. Parsing headers for Content-Length...")
            headers_section = data_str[:headers_end]
            headers = headers_section.split('\r\n')[1:] # Skip the request line
            for header in headers:
                if header.lower().startswith('content-length:'):
                    try:
                        content_length = int(header[len('content-length:'):].strip())
                        logging.debug(f"handle_client: Detected Content-Length: {content_length}")
                        break
                    except ValueError:
                        logging.warning(f"handle_client: Could not parse Content-Length header: {header}")
                        content_length = 0 # Treat as no body or error

            # Read the rest of the body if Content-Length is specified and initial data didn't contain it all
            body_start_index = headers_end + 4 # +4 for \r\n\r\n
            body_already_read = initial_data[body_start_index:]
            request_body = body_already_read

            if content_length > 0 and len(request_body) < content_length:
                bytes_to_read = content_length - len(request_body)
                logging.debug(f"handle_client: Need to read {bytes_to_read} more bytes for the body.")
                try:
                    # Read exactly the remaining bytes for the body
                    remaining_body = await reader.readexactly(bytes_to_read)
                    request_body += remaining_body
                    logging.debug(f"handle_client: Read remaining body ({len(remaining_body)} bytes). Total body size: {len(request_body)}")
                except asyncio.IncompleteReadError as e:
                    logging.error(f"handle_client: Incomplete read of request body: Expected {e.expected}, got {e.partial.decode('utf-8', errors='ignore')[:100]}...")
                    # Decide how to handle incomplete body - for now, proceed with what we have
                    # Note: This might cause JSON parsing to fail later
                    request_body += e.partial # Append partial data
                except Exception as e:
                    logging.error(f"handle_client: Error reading remaining body: {e}")


            # Attempt to parse the body as JSON and find the 'model' field
            if request_body:
                try:
                    # Decode body for JSON parsing, assuming UTF-8
                    body_str = request_body.decode('utf-8')
                    body_json = json.loads(body_str)
                    model_name = body_json.get('model')
                    if model_name:
                        logging.info(f"handle_client: Detected model in JSON body: {model_name}")
                        if model_callback:
                            logging.debug(f"handle_client: Calling model_callback with {model_name}")
                            # Call the callback with the detected model name
                            # Schedule the callback as a task in the current loop
                            asyncio.create_task(model_callback(model_name))
                    else:
                        logging.debug("handle_client: 'model' field not found in JSON body.")
                except json.JSONDecodeError:
                    logging.debug(f"handle_client: Request body is not valid JSON. Body starts with: {request_body.decode('utf-8', errors='ignore')[:100]}...")
                except Exception as e:
                    logging.error(f"handle_client: Error processing JSON body: {e}")
            else:
                logging.debug("handle_client: Request body is empty.")

        elif headers_end == -1:
             logging.debug("handle_client: Could not find end of headers in initial data. Cannot parse body.")
        else:
            logging.debug(f"handle_client: Not a POST request (starts with '{data_str[:10]}...'). Skipping body parsing.")


        # Reconstruct the full request data to send downstream
        # If headers_end was found, combine headers + body. Otherwise, send initial_data as is.
        full_request_data = initial_data[:body_start_index] + request_body if headers_end != -1 else initial_data
        logging.debug(f"handle_client: Full request data size to send downstream: {len(full_request_data)}")

        # Connect to the downstream server
        logging.info(f"handle_client: Connecting to downstream server {downstream_host}:{downstream_port}")
        downstream_reader, downstream_writer = await asyncio.open_connection(
            downstream_host, downstream_port)
        logging.info(f"handle_client: Connected to downstream server")

        # Send the full request data
        logging.debug(f"handle_client: Sending full request data ({len(full_request_data)} bytes) to downstream.")
        downstream_writer.write(full_request_data)
        await downstream_writer.drain()
        logging.debug("handle_client: Full request data sent to downstream.")

        # Set up bidirectional tunneling
        async def forward_data(src_reader, dest_writer):
            try:
                while True:
                    data = await src_reader.read(4096) # Read in chunks
                    if not data:
                        break # Connection closed
                    dest_writer.write(data)
                    await dest_writer.drain()
            except ConnectionResetError:
                logging.warning("Connection reset during data forwarding")
            except Exception as e:
                logging.error(f"Error during data forwarding: {e}")
            finally:
                # Attempt to close connections gracefully
                if not src_reader.at_eof():
                     src_reader.feed_eof()
                if not dest_writer.is_closing():
                    try:
                        dest_writer.close()
                        await dest_writer.wait_closed()
                    except Exception:
                        pass # Ignore errors on closing

        # Run forwarding tasks concurrently
        client_to_downstream = asyncio.create_task(forward_data(reader, downstream_writer))
        downstream_to_client = asyncio.create_task(forward_data(downstream_reader, writer))

        # Wait for either connection to close
        await asyncio.gather(client_to_downstream, downstream_to_client)

    except ConnectionRefusedError:
        logging.error(f"Connection to downstream server {downstream_host}:{downstream_port} refused.")
        # Send a basic HTTP error response to the client
        error_response = b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n"
        writer.write(error_response)
        await writer.drain()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        # Attempt to send a generic error response
        if not writer.is_closing():
             try:
                error_response = b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n"
                writer.write(error_response)
                await writer.drain()
             except Exception:
                 pass # Ignore errors on sending error response
    finally:
        logging.info(f"Closing connection from {client_address}")
        if writer and not writer.is_closing():
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass # Ignore errors on closing
        if downstream_writer and not downstream_writer.is_closing():
            try:
                downstream_writer.close()
                await downstream_writer.wait_closed()
            except Exception:
                pass # Ignore errors on closing


async def start_proxy(listen_host, listen_port, downstream_host, downstream_port, model_callback=None):
    """Starts the proxy server."""
    logging.info(f"Starting proxy server on {listen_host}:{listen_port}")
    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, downstream_host, downstream_port, model_callback),
        listen_host, listen_port)

    addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
    logging.info(f'Serving on {addrs}')

    return server # Return the server object to allow stopping it later
