import cairosvg
from PIL import Image
import io

def load_svg_with_cairosvg(svg_file_path, output_format='png', output_path=None):
    """
    Loads an SVG file and converts it to a raster format using cairosvg.

    Args:
        svg_file_path (str): Path to the SVG file.
        output_format (str, optional): Output format ('png', 'pdf', 'ps', 'eps', 'svg'). Defaults to 'png'.
        output_path (str, optional): Path to save the output file. If None, the output is not saved.

    Returns:
        PIL.Image.Image or bytes: A PIL Image object if the output format is 'png', otherwise the raw bytes of the converted file.
    """
    try:
        with open(svg_file_path, 'rb') as f:
            svg_data = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {svg_file_path}")
    
    if output_format == 'png':
        output_data = cairosvg.svg2png(svg_data)
        image = Image.open(io.BytesIO(output_data))
        if output_path:
            image.save(output_path)
        return image
    else:
        converter = getattr(cairosvg, f"svg2{output_format}")
        output_data = converter(svg_data, write_to=output_path)
        return output_data

# # Example usage:
# try:
#     image = load_svg_with_cairosvg('example.svg', output_format='png', output_path='output.png')
#     if image:
#         print("SVG loaded and converted to PNG successfully.")
# except FileNotFoundError as e:
#     print(e)
# except Exception as e:
#     print(f"An error occurred: {e}")

