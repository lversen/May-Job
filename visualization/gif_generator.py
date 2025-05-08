import os
import glob
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def create_training_gifs(results_dir):
    """
    Create GIFs from training visualizations.
    
    Parameters:
    - results_dir: Directory containing the visualization images
    
    Returns:
    - Dictionary mapping visualization type to GIF path
    """
    try:
        import os
        import glob
        import re
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from tqdm import tqdm
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "pillow", "numpy", "tqdm"])
        import os
        import glob
        import re
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from tqdm import tqdm
    
    # Create output directory for GIFs
    gif_dir = os.path.join(results_dir, 'gifs')
    os.makedirs(gif_dir, exist_ok=True)
    
    print(f"Searching for visualization images in {results_dir}...")
    
    # Find all PNG files in the results directory and its subdirectories
    all_png_files = glob.glob(os.path.join(results_dir, '**/*.png'), recursive=True)
    print(f"Found {len(all_png_files)} PNG files.")
    
    # Group files by visualization type
    visualization_groups = {}
    
    # Use tqdm to show progress while processing files
    for file_path in tqdm(all_png_files, desc="Grouping image files"):
        file_name = os.path.basename(file_path)
        
        # Extract epoch number if present
        epoch_match = re.search(r'epoch_(\d+)', file_name)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            
            # Determine visualization type
            if 'predictions_epoch' in file_name:
                viz_type = 'predictions'
            elif 'error_histogram' in file_name:
                viz_type = 'error_histogram'
            elif 'node_prediction_distribution' in file_name:
                viz_type = 'node_distribution'
            elif 'node_vs_target' in file_name:
                viz_type = 'node_vs_target'
            elif 'prediction_vs_target_by_molecule' in file_name:
                viz_type = 'prediction_by_molecule'
            else:
                # Skip files that don't match our patterns
                continue
                
            # Add to appropriate group with epoch information
            if viz_type not in visualization_groups:
                visualization_groups[viz_type] = []
            
            visualization_groups[viz_type].append((epoch, file_path))
    
    # Also check for non-epoch-specific visualizations (like loss curves and RMSE curves)
    for file_path in all_png_files:
        file_name = os.path.basename(file_path)
        
        if 'loss_curves' in file_name:
            if 'loss_curves' not in visualization_groups:
                visualization_groups['loss_curves'] = []
            visualization_groups['loss_curves'].append((0, file_path))  # Use 0 as placeholder epoch
            
        elif 'rmse_curves' in file_name:
            if 'rmse_curves' not in visualization_groups:
                visualization_groups['rmse_curves'] = []
            visualization_groups['rmse_curves'].append((0, file_path))  # Use 0 as placeholder epoch
    
    print(f"Grouped files into {len(visualization_groups)} visualization types.")
    
    # Try to find a font for adding text
    try:
        # Try different font options, falling back as needed
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
    except:
        font = None
    
    # Dictionary to store GIF paths
    gif_paths = {}
    
    # Create GIFs for each visualization type
    for viz_type, files in tqdm(visualization_groups.items(), desc="Creating GIFs for each visualization type"):
        # Sort by epoch
        files.sort(key=lambda x: x[0])
        
        print(f"Creating GIF for {viz_type} visualization ({len(files)} frames)...")
        
        # Create readable names for the visualization types
        readable_names = {
            'predictions': 'Predicted vs Actual Values',
            'error_histogram': 'Error Distribution',
            'node_distribution': 'Node Prediction Distribution',
            'node_vs_target': 'Node Predictions vs Molecule Targets',
            'prediction_by_molecule': 'Predictions by Molecule',
            'loss_curves': 'Loss Curves',
            'rmse_curves': 'RMSE Curves'
        }
        
        # Use the readable name for the GIF filename
        gif_name = viz_type.replace('_', '-')
        gif_path = os.path.join(gif_dir, f'{gif_name}.gif')
        gif_paths[viz_type] = gif_path
        
        # If the visualizations aren't epoch-specific, just use the latest one
        if viz_type in ['loss_curves', 'rmse_curves'] and len(files) > 0:
            # Just create a steady GIF of the latest plot
            latest_file = files[-1][1]
            img = Image.open(latest_file)
            
            # Create a GIF that shows the same image multiple times
            frames = [img] * 10  # 10 frames of the same image
            
            frames[0].save(gif_path, format='GIF', append_images=frames[1:], 
                         save_all=True, duration=500, loop=0)
            print(f"  Created GIF: {gif_path}")
            
        # For epoch-specific visualizations, create animated GIFs
        elif len(files) > 1:  # Need at least 2 images for animation
            frames = []
            
            # Get a list of all epochs in order
            epochs = [ep for ep, _ in files]
            
            # For better animated GIFs, ensure equal spacing between epochs
            # If we have too many epochs, sample some of them
            max_frames = 50  # Maximum number of frames in the GIF
            if len(epochs) > max_frames:
                # Sample epochs for smoother animation
                epochs_indices = np.linspace(0, len(epochs)-1, max_frames, dtype=int)
                selected_files = [files[i] for i in epochs_indices]
            else:
                selected_files = files
            
            # Process each file with tqdm progress bar
            for epoch, file_path in tqdm(selected_files, desc=f"Processing frames for {viz_type} GIF", leave=False):
                try:
                    img = Image.open(file_path)
                    
                    # Add epoch number as text overlay
                    if font is not None:
                        draw = ImageDraw.Draw(img)
                        draw.text((10, 10), f"Epoch: {epoch}", fill="black", font=font)
                    
                    frames.append(img)
                except Exception as e:
                    print(f"  Error processing {file_path}: {e}")
                    continue
            
            # Create GIF if we have frames
            if frames:
                try:
                    # Save with PIL, which gives better control over animation
                    frames[0].save(
                        gif_path, 
                        format='GIF',
                        append_images=frames[1:],
                        save_all=True,
                        duration=200,  # 200ms per frame
                        loop=0,  # Loop forever
                    )
                    print(f"  Created GIF: {gif_path}")
                except Exception as e:
                    print(f"  Error creating GIF for {viz_type}: {e}")
    
    print(f"All GIFs saved to {gif_dir}")
    return gif_paths


def create_training_animation(log_dir, results_dir):
    """
    Create animated GIFs from training visualizations.
    
    Parameters:
    - log_dir: Directory containing training logs
    - results_dir: Directory containing visualization results
    
    Returns:
    - Dictionary mapping visualization type to GIF path
    """
    gif_paths = create_training_gifs(results_dir)
    return gif_paths