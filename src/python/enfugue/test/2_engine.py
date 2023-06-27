"""
Uses the engine to create a simple image using default settings
"""
import os
from enfugue.diffusion.engine import DiffusionEngine
from pibble.util.log import DebugUnifiedLoggingContext

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-images", "base")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with DiffusionEngine() as engine:
            engine(prompt="A happy-looking puppy")["images"][0].save(os.path.join(save_dir, "./puppy-async.png"))

if __name__ == "__main__":
    main()
