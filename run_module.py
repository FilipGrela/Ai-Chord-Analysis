"""
Universal Script Launcher - runs modules from a specified folder with parameters
"""

import os
import sys
import argparse
import importlib.util
import inspect
from pathlib import Path
from typing import List, Dict, Callable


class ModuleLauncher:
    """Discovers and runs Python modules with support for parameters"""
    
    def __init__(self, modules_dir: str):
        self.modules_dir = Path(modules_dir).resolve()
        if not self.modules_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.modules_dir}")
        self.modules: Dict[str, Path] = {}
        self.discover_modules()
    
    def discover_modules(self) -> None:
        """Discover all Python modules in the target directory"""
        py_files = list(self.modules_dir.glob("*.py"))
        py_files = [f for f in py_files if not f.name.startswith("_")]
        
        for py_file in sorted(py_files):
            module_name = py_file.stem
            self.modules[module_name] = py_file
        
        if not self.modules:
            print(f"⚠️  No Python modules found in {self.modules_dir}")
        else:
            print(f"✓ Found {len(self.modules)} module(s)")
    
    def list_modules(self) -> None:
        """Print available modules"""
        if not self.modules:
            print("No modules available")
            return
        
        print("\n📚 Available modules:")
        for idx, name in enumerate(self.modules.keys(), 1):
            print(f"  {idx}. {name}")
    
    def get_module_info(self, module_name: str) -> Dict:
        """Extract main function and its parameters"""
        if module_name not in self.modules:
            return None
        
        module_path = self.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            return {"error": str(e)}
        
        main_func = getattr(module, "main", None)
        if not main_func or not callable(main_func):
            return {"has_main": False, "module": module}
        
        sig = inspect.signature(main_func)
        params = list(sig.parameters.values())
        
        return {
            "has_main": True,
            "main_func": main_func,
            "module": module,
            "params": params,
            "docstring": inspect.getdoc(main_func) or "No description"
        }
    
    def run_module(self, module_name: str, args: List[str] = None) -> None:
        """Run a module's main function with optional arguments"""
        print(f"\n🚀 Running module: {module_name}")
        print("-" * 50)
        
        info = self.get_module_info(module_name)
        
        if info is None:
            print(f"❌ Module '{module_name}' not found")
            return
        
        if "error" in info:
            print(f"❌ Error loading module: {info['error']}")
            return
        
        if not info["has_main"]:
            print("❌ Module doesn't have a main() function")
            return
        
        print(f"📝 {info['docstring']}")
        print()
        
        try:
            main_func = info["main_func"]
            params = info["params"]
            
            if not params:
                main_func()
            elif len(params) == 1 and params[0].name == "args":
                main_func(args or [])
            else:
                if args:
                    main_func(*args)
                else:
                    print(f"⚠️  Module expects {len(params)} argument(s), but none provided")
                    print(f"   Parameters: {', '.join(p.name for p in params)}")
            
            print("\n✅ Module completed successfully")
        
        except Exception as e:
            print(f"\n❌ Error running module: {e}")
            import traceback
            traceback.print_exc()
    
    def interactive_mode(self) -> None:
        """Interactive menu to select and run modules"""
        while True:
            print("\n" + "=" * 50)
            print("🎯 Module Launcher - Interactive Mode")
            print("=" * 50)
            self.list_modules()
            print("\n  0. Exit")
            print()
            
            try:
                choice = input("Select module (number): ").strip()
                
                if choice == "0":
                    print("Goodbye! 👋")
                    break
                
                module_names = list(self.modules.keys())
                idx = int(choice) - 1
                
                if 0 <= idx < len(module_names):
                    module_name = module_names[idx]
                    args_input = input("Enter arguments (space-separated, or press Enter for none): ").strip()
                    args = args_input.split() if args_input else []
                    
                    self.run_module(module_name, args)
                else:
                    print("❌ Invalid selection")
            
            except (ValueError, KeyboardInterrupt):
                print("\n⚠️  Invalid input or interrupted")
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Universal Script Launcher - runs Python modules with parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_modules.py backend/scripts                    # Interactive mode
  python run_modules.py backend/scripts -l                 # List modules
  python run_modules.py backend/scripts -m my_script       # Run specific module
  python run_modules.py backend/scripts -m script arg1 arg2 # With arguments
        """)
    
    parser.add_argument("modules_dir", help="Path to directory containing Python modules")
    parser.add_argument("-l", "--list", action="store_true", help="List available modules and exit")
    parser.add_argument("-m", "--module", help="Module name to run (without .py extension)")
    parser.add_argument("args", nargs="*", help="Arguments to pass to the module")
    
    args = parser.parse_args()
    
    try:
        launcher = ModuleLauncher(args.modules_dir)
        
        if args.list:
            launcher.list_modules()
        elif args.module:
            launcher.run_module(args.module, args.args if args.args else None)
        else:
            launcher.interactive_mode()
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
