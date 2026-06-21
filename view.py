import trimesh
import os
import sys

def view_latest_model():
    """Find and view the latest downloaded 3D model."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look in both root directory and models/ folder
    search_dirs = [
        base_dir,  # Root directory
        os.path.join(base_dir, "models"),  # models/ subfolder
    ]
    
    # Find all model directories
    model_dirs = []
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for item in os.listdir(search_dir):
            item_path = os.path.join(search_dir, item)
            if os.path.isdir(item_path) and item.startswith("model_"):
                model_dirs.append(item_path)
    
    if not model_dirs:
        print("No models found. Complete a scan and reconstruction first.")
        return
    
    # Sort by modification time (newest first)
    model_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = model_dirs[0]
    
    print(f"Latest model: {latest_dir}")
    
    # Find OBJ or STL file
    obj_files = []
    for f in os.listdir(latest_dir):
        if f.endswith('.obj') or f.endswith('.stl'):
            obj_files.append(os.path.join(latest_dir, f))
    
    if not obj_files:
        print(f"No OBJ/STL files found in {latest_dir}")
        return
    
    model_path = obj_files[0]
    print(f"Loading: {model_path}")
    
    try:
        mesh = trimesh.load(model_path)
        print(f"Vertices: {len(mesh.vertices)}")
        print(f"Faces: {len(mesh.faces)}")
        print("\nOpening 3D viewer...")
        print("Controls:")
        print("  - Left mouse: Rotate")
        print("  - Right mouse: Pan")
        print("  - Scroll: Zoom")
        print("  - Close window to exit")
        
        mesh.show()
        
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    view_latest_model()