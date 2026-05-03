import os

def replace_in_file(file_path, search_text, replace_text):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        new_content = content.replace(search_text, replace_text)
        new_content = new_content.replace(search_text.lower(), replace_text.lower())
        new_content = new_content.replace(search_text.capitalize(), replace_text.capitalize())
        new_content = new_content.replace(search_text.upper(), replace_text.upper())
        
        if content != new_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    root_dir = r"c:\Penelitian\ultralytics"
    exclude_dirs = {'.git', 'output', 'runs', '__pycache__', '.agent'}
    
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith(('.py', '.md', '.txt', '.yaml')):
                file_path = os.path.join(root, file)
                replace_in_file(file_path, "analytics", "analytics")

if __name__ == "__main__":
    main()
