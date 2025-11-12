import sys
from katabatic.cli import init_model


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [args...]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "init-model":
        if len(sys.argv) < 3:
            print(
                "Usage: python main.py init-model <model_name> [dependency1 dependency2 ...]")
            sys.exit(1)
        model_name = sys.argv[2]
        dependencies = sys.argv[3:] if len(sys.argv) > 3 else None
        try:
            from katabatic.cli.commands.model_init import init_model
            init_model(model_name, dependencies)
        except Exception as e:
            print(f"Error creating model: {e}")
            sys.exit(1)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
