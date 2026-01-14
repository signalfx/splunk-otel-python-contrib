#!/usr/bin/env python3
"""
Setup verification script for O11y AI Test Framework.
Run this to verify all dependencies are installed correctly.
"""
import sys
from pathlib import Path

def verify_imports():
    """Verify all required imports work."""
    print("🔍 Verifying imports...")
    
    try:
        import pytest
        print(f"  ✅ pytest {pytest.__version__}")
    except ImportError as e:
        print(f"  ❌ pytest: {e}")
        return False
    
    try:
        import playwright
        print(f"  ✅ playwright installed")
    except ImportError as e:
        print(f"  ❌ playwright: {e}")
        return False
    
    try:
        import requests
        print(f"  ✅ requests {requests.__version__}")
    except ImportError as e:
        print(f"  ❌ requests: {e}")
        return False
    
    try:
        import structlog
        print(f"  ✅ structlog {structlog.__version__}")
    except ImportError as e:
        print(f"  ❌ structlog: {e}")
        return False
    
    try:
        import yaml
        print(f"  ✅ pyyaml installed")
    except ImportError as e:
        print(f"  ❌ pyyaml: {e}")
        return False
    
    try:
        from tenacity import retry
        print(f"  ✅ tenacity installed")
    except ImportError as e:
        print(f"  ❌ tenacity: {e}")
        return False
    
    return True

def verify_framework_structure():
    """Verify framework directory structure."""
    print("\n🔍 Verifying framework structure...")
    
    required_dirs = [
        "config",
        "core",
        "clients",
        "page_objects",
        "validators",
        "fixtures",
        "tests",
        "utils",
        "test_data",
        "reports"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (missing)")
            all_exist = False
    
    return all_exist

def verify_core_components():
    """Verify core components can be imported."""
    print("\n🔍 Verifying core components...")
    
    try:
        from core.logger import get_logger
        logger = get_logger("test")
        print(f"  ✅ core.logger")
    except Exception as e:
        print(f"  ❌ core.logger: {e}")
        return False
    
    try:
        from core.retry_handler import retry_with_backoff
        print(f"  ✅ core.retry_handler")
    except Exception as e:
        print(f"  ❌ core.retry_handler: {e}")
        return False
    
    try:
        from core.api_client import APIClient
        print(f"  ✅ core.api_client")
    except Exception as e:
        print(f"  ❌ core.api_client: {e}")
        return False
    
    return True

def verify_config():
    """Verify configuration system."""
    print("\n🔍 Verifying configuration...")
    
    try:
        from config.base_config import BaseConfig
        print(f"  ✅ config.base_config")
        
        # Check if RC0 config exists
        rc0_config = Path("config/environments/rc0.yaml")
        if rc0_config.exists():
            print(f"  ✅ config/environments/rc0.yaml")
        else:
            print(f"  ❌ config/environments/rc0.yaml (missing)")
            return False
            
    except Exception as e:
        print(f"  ❌ config: {e}")
        return False
    
    return True

def verify_clients():
    """Verify API clients."""
    print("\n🔍 Verifying API clients...")
    
    try:
        from clients.apm_client import APMClient
        print(f"  ✅ clients.apm_client")
    except Exception as e:
        print(f"  ❌ clients.apm_client: {e}")
        return False
    
    return True

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("O11y AI Test Framework - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Dependencies", verify_imports),
        ("Framework Structure", verify_framework_structure),
        ("Core Components", verify_core_components),
        ("Configuration", verify_config),
        ("API Clients", verify_clients)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} check failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All checks passed! Framework is ready to use.")
        print("\nNext steps:")
        print("  1. Copy env.example to .env and configure credentials")
        print("  2. Continue implementing remaining components")
        print("  3. Write test cases")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
