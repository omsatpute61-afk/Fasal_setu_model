import os
from IPython.display import FileLink, display

def generate_label_file(target_dir: str, output_filename: str) -> str | None:
    working_dir = "/kaggle/working/"
    output_path = os.path.join(working_dir, output_filename)
    
    if not os.path.exists(target_dir):
        print(f"Error: Target directory not found - {target_dir}")
        return None
        
    # List all sub-directories and filter out any rogue files
    subdirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    
    # Strictly sort alphabetically to match PyTorch class_to_idx indexing
    subdirs.sort()
    
    # Write exactly one class name per line
    with open(output_path, "w", encoding="utf-8") as f:
        for i, subdir in enumerate(subdirs):
            if i < len(subdirs) - 1:
                f.write(f"{subdir}\n")
            else:
                f.write(f"{subdir}")
                
    return output_filename

def main():
    disease_path = "/kaggle/input/datasets/shruthisindhura/pestopia/Datasets/Disease_Dataset"
    pest_path = "/kaggle/input/datasets/shruthisindhura/pestopia/Datasets/Pest_Dataset"
    
    disease_out = "disease_labels.txt"
    pest_out = "pest_labels.txt"
    
    disease_file = generate_label_file(disease_path, disease_out)
    pest_file = generate_label_file(pest_path, pest_out)
    
    if disease_file:
        print(f"Successfully generated {disease_file}")
        display(FileLink(disease_file))
        
    if pest_file:
        print(f"Successfully generated {pest_file}")
        display(FileLink(pest_file))

if __name__ == "__main__":
    main()
