import re

file_path = "mistralrs-web-chat/static/js/ui.js"
with open(file_path, "r") as f:
    content = f.read()

# Replace the incorrect line
old_line = "const maxH = parseFloat(getComputedStyle(input).lineHeight) * 15;"
new_line = "const maxH = 1.4 * parseFloat(getComputedStyle(input).fontSize) * 15;"

if old_line in content:
    content = content.replace(old_line, new_line)
    with open(file_path, "w") as f:
        f.write(content)
    print("Patch applied successfully.")
else:
    print("Target line not found in file.")
