import os

README_PATH = "readme.md"  # Ścieżka do README.md

# Przypisanie emoji do typów plików
ICONS = {
    "folder": "📁",
    "python": "🐍",
    "markdown": "📜",
    "text": "📝",
    "image": "🖼️",
    "code": "💻",
    "default": "📄"
}

# Funkcja do wyboru ikony na podstawie rozszerzenia pliku
def get_icon(filename):
    if os.path.isdir(filename):
        return ICONS["folder"]
    ext = filename.split(".")[-1].lower()
    return {
        "py": ICONS["python"],
        "md": ICONS["markdown"],
        "txt": ICONS["text"],
        "jpg": ICONS["image"],
        "png": ICONS["image"],
        "gif": ICONS["image"],
        "html": ICONS["code"],
        "css": ICONS["code"],
        "js": ICONS["code"]
    }.get(ext, ICONS["default"])

# Rekurencyjna funkcja do generowania drzewa katalogów
def generate_structure(directory, level=0):
    markdown = ""
    for item in sorted(os.listdir(directory)):
        if item.startswith("."):  # Pomijanie ukrytych plików (np. .git)
            continue
        path = os.path.join(directory, item)
        icon = get_icon(path)
        markdown += "  " * level + f"- {icon} `{item}`\n"
        if os.path.isdir(path):
            markdown += generate_structure(path, level + 1)
    return markdown

# Funkcja do aktualizacji README.md
def update_readme():
    structure = generate_structure(".")
    new_section = f"<!-- START_STRUCTURE -->\n\n# 📂 Struktura katalogów\n\n{structure}\n<!-- END_STRUCTURE -->"

    if os.path.exists(README_PATH):
        with open(README_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # Sprawdzamy, czy znacznik już istnieje
        if "<!-- START_STRUCTURE -->" in content and "<!-- END_STRUCTURE -->" in content:
            content = content.split("<!-- START_STRUCTURE -->")[0] + new_section + content.split("<!-- END_STRUCTURE -->")[1]
        else:
            content += "\n\n" + new_section  # Dodajemy strukturę na końcu

    else:
        content = new_section  # Tworzymy nowy plik README, jeśli nie istnieje

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Struktura katalogów została zaktualizowana w README.md")

# Uruchomienie skryptu
if __name__ == "__main__":
    update_readme()
