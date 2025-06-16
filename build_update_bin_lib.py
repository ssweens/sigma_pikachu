import os
import zipfile
import tarfile
import shutil
import subprocess
import glob

# Define base directories
base_dir = os.path.dirname(os.path.abspath(__file__))
drop_dir = os.path.join(base_dir, '_drop')
sigma_pikachu_dir = os.path.join(base_dir, 'sigma_pikachu')
lib_dir = os.path.join(sigma_pikachu_dir, 'lib')
bin_dir = os.path.join(sigma_pikachu_dir, 'bin')
temp_extract_dir = os.path.join(base_dir, 'temp_extracted_files')

# Ensure destination directories exist
os.makedirs(lib_dir, exist_ok=True)
os.makedirs(bin_dir, exist_ok=True)
os.makedirs(temp_extract_dir, exist_ok=True)

print(f"Ensured {lib_dir} and {bin_dir} exist.")

# --- Handle llama-*.zip file ---
llama_zip_path = None
for f in os.listdir(drop_dir):
    if f.startswith('llama-') and f.endswith('.zip') and "-swap" not in f:
        llama_zip_path = os.path.join(drop_dir, f)
        break

if llama_zip_path:
    print(f"Processing {llama_zip_path}...")
    copied_to_lib_count = 0
    copied_to_bin_count = 0
    with zipfile.ZipFile(llama_zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            # 1) From the llama-*.zip, in bin, grab all the .dylib, .h, and .metal files and copy them to sigma_pikachu/lib
            if member.startswith('build/bin/') and (member.endswith('.dylib') or member.endswith('.h') or member.endswith('.metal')):
                filename = os.path.basename(member)
                source_path = zip_ref.extract(member, temp_extract_dir)
                destination_path = os.path.join(lib_dir, filename)
                if os.path.exists(destination_path):
                    os.remove(destination_path)
                    #print(f"Overwriting existing file: {destination_path}")
                shutil.move(source_path, destination_path)
                #print(f"Copied {filename} to {lib_dir}")
                copied_to_lib_count += 1

            # 2) From the same zip, in bin, grab llama-mtmd-cli llama-server and copy them to sigma_pikachu/bin
            elif member.startswith('build/bin/') and (os.path.basename(member) == 'llama-mtmd-cli' or os.path.basename(member) == 'llama-server'):
                filename = os.path.basename(member)
                source_path = zip_ref.extract(member, temp_extract_dir)
                destination_path = os.path.join(bin_dir, filename)
                if os.path.exists(destination_path):
                    os.remove(destination_path)
                    #print(f"Overwriting existing file: {destination_path}")
                shutil.move(source_path, destination_path)
                # Make executable
                os.chmod(destination_path, 0o755)
                #print(f"Copied and made executable {filename} to {bin_dir}")
                copied_to_bin_count += 1
    print(f"Finished processing {llama_zip_path}. Copied {copied_to_lib_count} files to {lib_dir} and {copied_to_bin_count} files to {bin_dir}.")
else:
    print("No llama-*.zip file found in _drop directory.")

# Copy corral from ~/src/corral to sigma_pikachu/bin
corral_source_path = os.path.expanduser('~/src/corral/target/release/corral')
corral_destination_path = os.path.join(bin_dir, 'corral')
if os.path.exists(corral_source_path):
    shutil.copy(corral_source_path, corral_destination_path)
    os.chmod(corral_destination_path, 0o755)
    print(f"Copied and made executable {corral_source_path} to {corral_destination_path}")

# Copy toolshed from ~/src/toolshed to sigma_pikachu/bintoolshed_source_path = os.path.expanduser('~/src/toolshed/toolshed')
toolshed_destination_path = os.path.join(bin_dir, 'toolshed')
toolshed_source_path = os.path.expanduser('~/src/toolshed/target/release/toolshed')
if os.path.exists(toolshed_source_path):
    shutil.copy(toolshed_source_path, toolshed_destination_path)
    os.chmod(toolshed_destination_path, 0o755)
    print(f"Copied and made executable {toolshed_source_path} to {toolshed_destination_path}")

# --- Handle llama-swap*.tar.gz file ---
# llama_swap_tar_gz_path = None
# for f in os.listdir(drop_dir):
#     if f.startswith('llama-swap') and f.endswith('.tar.gz'):
#         llama_swap_tar_gz_path = os.path.join(drop_dir, f)
#         break

# if llama_swap_tar_gz_path:
#     print(f"Processing {llama_swap_tar_gz_path}...")
#     with tarfile.open(llama_swap_tar_gz_path, 'r:gz') as tar_ref:
#         for member in tar_ref.getmembers():
#             # 3) From the llama-swap*.tar.gz, copy llama-swap to sigma_pikachu/bin
#             if os.path.basename(member.name) == 'llama-swap':
#                 source_path = os.path.join(temp_extract_dir, member.name)
#                 tar_ref.extract(member, temp_extract_dir)
#                 destination_path = os.path.join(bin_dir, 'llama-swap')
#                 if os.path.exists(destination_path):
#                     os.remove(destination_path)
#                     print(f"Overwriting existing file: {destination_path}")
#                 shutil.move(source_path, destination_path)
#                 # Make executable
#                 os.chmod(destination_path, 0o755)
#                 print(f"Copied and made executable llama-swap to {bin_dir}")
# else:
#     print("No llama-swap*.tar.gz file found in _drop directory.")

# Clean up temporary extraction directory
if os.path.exists(temp_extract_dir):
    shutil.rmtree(temp_extract_dir)
    print(f"Cleaned up temporary directory: {temp_extract_dir}")

# --- Copy /usr/local/bin/ollama to sigma_pikachu/bin ---
ollama_source_path = '/usr/local/bin/ollama'
ollama_destination_path = os.path.join(bin_dir, 'ollama')

if os.path.exists(ollama_source_path):
    print(f"Copying {ollama_source_path} to {ollama_destination_path}...")
    if os.path.exists(ollama_destination_path):
        os.remove(ollama_destination_path)
        print(f"Overwriting existing file: {ollama_destination_path}")
    shutil.copy(ollama_source_path, ollama_destination_path)
    os.chmod(ollama_destination_path, 0o755)
    print(f"Copied and made executable {ollama_source_path} to {ollama_destination_path}")
else:
    print(f"Warning: {ollama_source_path} not found. Skipping ollama copy.")

print("File movement script finished.")

# --- Remove quarantine attribute from copied files ---
print("\n--- Removing quarantine attributes ---")
# Remove quarantine attribute from copied files
print("\n--- Removing quarantine attributes ---")

# For bin directory
bin_files = glob.glob(os.path.join(bin_dir, '*'))
if bin_files:
    print(f"Attempting to remove quarantine attributes from files in {bin_dir}...")
    # Use shell=True to allow wildcard expansion, but be cautious with untrusted input
    # Here, paths are constructed internally, so it's safe.
    result = subprocess.run(['sudo', 'xattr', '-dr', 'com.apple.quarantine'] + bin_files, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Removed quarantine attribute from files in {bin_dir}")
    else:
        print(f"Error removing quarantine attribute from {bin_dir}/* (Exit Code: {result.returncode}):")
        if result.stdout: print(f"STDOUT:\n{result.stdout.strip()}")
        if result.stderr: print(f"STDERR:\n{result.stderr.strip()}")
        print("Note: If prompted for a password, you may need to run the 'sudo xattr' commands manually in your terminal.")
else:
    print(f"No files found in {bin_dir} to remove quarantine attributes from.")

# For lib directory
lib_files = glob.glob(os.path.join(lib_dir, '*'))
if lib_files:
    print(f"Attempting to remove quarantine attributes from files in {lib_dir}...")
    result = subprocess.run(['sudo', 'xattr', '-dr', 'com.apple.quarantine'] + lib_files, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Removed quarantine attribute from files in {lib_dir}")
    else:
        print(f"Error removing quarantine attribute from {lib_dir}/* (Exit Code: {result.returncode}):")
        if result.stdout: print(f"STDOUT:\n{result.stdout.strip()}")
        if result.stderr: print(f"STDERR:\n{result.stderr.strip()}")
        print("Note: If prompted for a password, you may need to run the 'sudo xattr' commands manually in your terminal.")
else:
    print(f"No files found in {lib_dir} to remove quarantine attributes from.")

# --- Print versions of moved executables ---
print("\n--- Executable Versions ---")

executables_to_check = {
    "llama-server": os.path.join(bin_dir, 'llama-server'),
    "corral": os.path.join(bin_dir, 'corral'),
    "toolshed": os.path.join(bin_dir, 'toolshed'),
    "ollama": os.path.join(bin_dir, 'ollama')
}

for name, path in executables_to_check.items():
    if os.path.exists(path):
        print(f"Checking {name} version...")
        try:
            # Special handling for llama-server due to dynamic library loading
            if name == "llama-server":
                # Construct the exact command string as provided by the user
                # Note: Using shell=True is generally discouraged for security with untrusted input,
                # but here the command is constructed internally.
                llama_server_cmd = f"DYLD_LIBRARY_PATH={lib_dir}:$DYLD_LIBRARY_PATH {path} --version"
                
                result = subprocess.run(llama_server_cmd, shell=True, capture_output=True, text=True)
                
                print(f"{name} version:")
                if result.stderr.strip(): # llama-server outputs version to stderr
                    print("\t" + result.stderr.strip() + "\n")
                elif result.stdout.strip():
                    print("\t" + result.stdout.strip() + "\n")
                else:
                    print("\t" + "(No version output found on stdout or stderr)")
                
                if result.returncode != 0:
                    print(f"Error getting {name} version (Exit Code: {result.returncode}).")
                    print(f"Could not determine version for {name}. Try running '{path} --help' for usage.")
            else:
                # For other executables, try --version then -v
                try:
                    result = subprocess.run([path, '--version'], capture_output=True, text=True, check=True)
                    print(f"{name} version:\n\t{result.stdout.strip()}\n")
                except subprocess.CalledProcessError:
                    result = subprocess.run([path, '-v'], capture_output=True, text=True, check=True)
                    print(f"{name} version:\n\t{result.stdout.strip()}\n")
        except subprocess.CalledProcessError as e:
            print(f"Error getting {name} version: {e}")
            print(f"STDOUT:\n{e.stdout.strip()}")
            print(f"STDERR:\n{e.stderr.strip()}")
            print(f"Could not determine version for {name}. Try running '{path} --help' for usage.")
        except FileNotFoundError:
            print(f"Error: {name} not found at {path}")
    else:
        print(f"Skipping version check for {name}: executable not found at {path}")

print("\n--- Script execution complete ---")