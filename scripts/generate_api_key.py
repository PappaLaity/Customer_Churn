#!/usr/bin/env python3
"""
Generate and rotate API keys safely.

Usage:
    python scripts/generate_api_key.py
    python scripts/generate_api_key.py --update-env
"""

import argparse
import secrets
import shutil
from pathlib import Path
from datetime import datetime


def generate_api_key(length=32):
    """Generate a cryptographically secure API key."""
    return secrets.token_urlsafe(length)


def backup_env_file(env_path):
    """Create a timestamped backup of the .env file."""
    if not env_path.exists():
        print(f"❌ {env_path} not found")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = env_path.parent / f".env.backup.{timestamp}"
    shutil.copy(env_path, backup_path)
    print(f"✅ Backup created: {backup_path}")
    return backup_path


def update_env_file(env_path, new_key):
    """Update the API_KEY_SECRET in .env file."""
    if not env_path.exists():
        print(f"❌ {env_path} not found. Please create it from .env.example first.")
        return False
    
    # Read current content
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update API_KEY_SECRET line
    updated = False
    for i, line in enumerate(lines):
        if line.startswith('API_KEY_SECRET'):
            old_line = line.strip()
            lines[i] = f'API_KEY_SECRET = "{new_key}"\n'
            updated = True
            print(f"📝 Updated: {old_line}")
            print(f"        → API_KEY_SECRET = \"{new_key}\"")
            break
    
    if not updated:
        print("⚠️  API_KEY_SECRET not found in .env, appending...")
        lines.append(f'\nAPI_KEY_SECRET = "{new_key}"\n')
    
    # Write back
    with open(env_path, 'w') as f:
        f.writelines(lines)
    
    print(f"✅ {env_path} updated successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate secure API keys')
    parser.add_argument(
        '--update-env',
        action='store_true',
        help='Automatically update .env file with new key'
    )
    parser.add_argument(
        '--length',
        type=int,
        default=32,
        help='Key length (default: 32)'
    )
    args = parser.parse_args()
    
    # Generate new key
    new_key = generate_api_key(args.length)
    
    print("=" * 70)
    print("🔐 API Key Generator")
    print("=" * 70)
    print(f"\n✨ Generated new API key:\n\n    {new_key}\n")
    
    if args.update_env:
        env_path = Path.cwd() / '.env'
        
        # Backup first
        backup_path = backup_env_file(env_path)
        if backup_path is None:
            return
        
        # Update .env
        if update_env_file(env_path, new_key):
            print("\n" + "=" * 70)
            print("✅ API KEY ROTATION COMPLETE")
            print("=" * 70)
            print("\n📋 Next steps:")
            print("   1. Restart your API: docker-compose restart fastapi")
            print("   2. All users must log in again (sessions invalidated)")
            print("   3. Update any external clients using this API")
            print(f"\n💾 Backup saved: {backup_path}")
        else:
            print("\n❌ Failed to update .env file")
    else:
        print("💡 To update .env automatically, run with --update-env flag:")
        print(f"   python {__file__} --update-env")
        print("\n📋 Manual steps:")
        print("   1. Copy the key above")
        print(f"   2. Update .env: API_KEY_SECRET = \"{new_key}\"")
        print("   3. Restart API: docker-compose restart fastapi")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
