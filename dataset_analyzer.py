import os
from PIL import Image

def analyze_dataset(data_dir="RealWaste"):
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.")
        return

    print(f"Scanning dataset in '{data_dir}'...")
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    total_images = 0
    corrupted_images = []
    class_distribution = {}

    for cat in categories:
        cat_path = os.path.join(data_dir, cat)
        files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Checking '{cat}' class ({len(files)} files)...")
        
        valid_count = 0
        for file in files:
            file_path = os.path.join(cat_path, file)
            try:
                # Attempt to open and verify the image file integrity
                with Image.open(file_path) as img:
                    img.verify()
                valid_count += 1
            except Exception as e:
                print(f"Corrupted image found: {file_path} (Error: {e})")
                corrupted_images.append(file_path)

        class_distribution[cat] = valid_count
        total_images += valid_count

    print("\n" + "="*40)
    print("        DATASET ANALYSIS REPORT")
    print("="*40)
    print(f"Total Valid Images: {total_images}")
    print(f"Total Corrupted Images: {len(corrupted_images)}")
    
    print("\nClass Distribution:")
    for cat, count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        print(f" - {cat:<20}: {count} ({percentage:.1f}%)")
    print("="*40)

    if corrupted_images:
        print("\nWould you like to delete the corrupted images? (Run script manually to clean)")
        for path in corrupted_images:
            print(f" - {path}")
    else:
        print("\nAll files are valid and healthy! Ready for training.")

if __name__ == "__main__":
    analyze_dataset()
