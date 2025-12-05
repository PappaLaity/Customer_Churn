#!/usr/bin/env python3
"""
Script pour remplacer automatiquement print() par logger
ATTENTION: Faire un backup avant d'exécuter !
"""
import re
from pathlib import Path
from typing import List


def replace_prints_in_file(file_path: Path) -> int:
    """
    Remplace les print() par logger.info() dans un fichier
    
    Returns:
        Nombre de remplacements effectués
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        original_content = content
        replacements = 0

        # Pattern pour détecter print()
        # Gère: print("..."), print(f"..."), print(variable)
        print_pattern = r'print\((.*?)\)'

        def replace_print(match):
            nonlocal replacements
            replacements += 1
            content = match.group(1)
            return f'logger.info({content})'

        # Effectuer les remplacements
        new_content = re.sub(print_pattern, replace_print, content)

        # Ajouter l'import du logger si des remplacements ont été faits
        if replacements > 0 and "from src.api.core.logger import" not in new_content:
            # Trouver la position après les imports existants
            import_section = re.search(r'((?:^import .*\n|^from .* import .*\n)+)', new_content, re.MULTILINE)
            if import_section:
                insert_pos = import_section.end()
                new_content = (
                    new_content[:insert_pos] +
                    "\nfrom src.api.core.logger import api_logger as logger\n" +
                    new_content[insert_pos:]
                )
            else:
                # Pas d'imports existants, ajouter au début
                new_content = "from src.api.core.logger import api_logger as logger\n\n" + new_content

        # Écrire seulement si changements
        if new_content != original_content:
            file_path.write_text(new_content, encoding="utf-8")
            print(f"✅ {file_path}: {replacements} remplacements")
            return replacements
        
        return 0

    except Exception as e:
        print(f"❌ Erreur avec {file_path}: {e}")
        return 0


def find_python_files(directories: List[Path]) -> List[Path]:
    """Trouve tous les fichiers .py dans les répertoires donnés"""
    files = []
    for directory in directories:
        if directory.exists():
            files.extend(directory.rglob("*.py"))
    return files


def main():
    """Point d'entrée principal"""
    print("🔄 Remplacement de print() par logger...")
    print("=" * 60)

    # Répertoires à traiter
    directories = [
        Path("src/api"),
        Path("src/etl"),
        Path("src/monitoring"),
        Path("src/training"),
        Path("dags"),
    ]

    # Fichiers à exclure
    exclude_patterns = ["__pycache__", "test_", "conftest.py", "__init__.py"]

    # Trouver les fichiers
    files = find_python_files(directories)
    files = [
        f for f in files 
        if not any(pattern in str(f) for pattern in exclude_patterns)
    ]

    print(f"📁 {len(files)} fichiers trouvés")
    print("")

    # Traiter chaque fichier
    total_replacements = 0
    for file_path in files:
        replacements = replace_prints_in_file(file_path)
        total_replacements += replacements

    print("")
    print("=" * 60)
    print(f"✅ Terminé! {total_replacements} remplacements au total")
    print("")
    print("⚠️  IMPORTANT: Vérifier manuellement que tout fonctionne:")
    print("   1. git diff src/")
    print("   2. pytest tests/ -v")
    print("   3. Si OK: git add . && git commit -m 'refactor: Replace print() with logger'")


if __name__ == "__main__":
    main()
