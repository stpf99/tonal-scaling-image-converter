import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter, laplace


class ImageConverterApp(Gtk.Window):
    def __init__(self):
        super().__init__(title="Tonal Scaling Image Converter")
        self.set_border_width(10)
        self.set_default_size(600, 500)

        # Layout
        grid = Gtk.Grid()
        self.add(grid)

        # Widgets
        self.image_label = Gtk.Label(label="Select Image File:")
        self.image_path_entry = Gtk.Entry()
        self.browse_button = Gtk.Button(label="Browse")
        
        # Scaling dropdown
        self.scaling_label = Gtk.Label(label="Scaling Factor:")
        self.scaling_combo = Gtk.ComboBoxText()
        scaling_options = ["1x (Original)", "2x", "4x", "8x", "16x"]
        for option in scaling_options:
            self.scaling_combo.append_text(option)
        self.scaling_combo.set_active(0)  # Default to original size
        
        # Smoothing dropdown
        self.smoothing_label = Gtk.Label(label="Smoothing Level:")
        self.smoothing_combo = Gtk.ComboBoxText()
        smoothing_options = ["No Smoothing", "Light", "Medium", "Strong"]
        for option in smoothing_options:
            self.smoothing_combo.append_text(option)
        self.smoothing_combo.set_active(0)  # Default to no smoothing

        # Sharpening dropdown
        self.sharpening_label = Gtk.Label(label="Sharpening Level:")
        self.sharpening_combo = Gtk.ComboBoxText()
        sharpening_options = ["No Sharpening", "Light", "Medium", "Strong"]
        for option in sharpening_options:
            self.sharpening_combo.append_text(option)
        self.sharpening_combo.set_active(0)  # Default to no sharpening
        
        self.analyze_button = Gtk.Button(label="Analyze and Save as PNG")
        self.output_label = Gtk.Label(label="Output: Waiting for input...")

        # Layout Widgets
        grid.attach(self.image_label, 0, 0, 1, 1)
        grid.attach(self.image_path_entry, 1, 0, 2, 1)
        grid.attach(self.browse_button, 3, 0, 1, 1)
        grid.attach(self.scaling_label, 0, 1, 1, 1)
        grid.attach(self.scaling_combo, 1, 1, 1, 1)
        grid.attach(self.smoothing_label, 0, 2, 1, 1)
        grid.attach(self.smoothing_combo, 1, 2, 1, 1)
        grid.attach(self.sharpening_label, 0, 3, 1, 1)
        grid.attach(self.sharpening_combo, 1, 3, 1, 1)
        grid.attach(self.analyze_button, 0, 4, 4, 1)
        grid.attach(self.output_label, 0, 5, 4, 1)

        # Connect signals
        self.browse_button.connect("clicked", self.on_browse_clicked)
        self.analyze_button.connect("clicked", self.on_analyze_clicked)

        self.image_path = None
        self.raw_image = None

    def on_browse_clicked(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Select an Image File", parent=self, action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK
        )

        if dialog.run() == Gtk.ResponseType.OK:
            self.image_path = dialog.get_filename()
            self.image_path_entry.set_text(self.image_path)

        dialog.destroy()

    def on_analyze_clicked(self, widget):
        if not self.image_path:
            self.output_label.set_text("Error: No image file selected!")
            return

        try:
            image = Image.open(self.image_path).convert("RGB")
            self.raw_image = np.array(image)
            self.output_label.set_text("Analyzing tonal transitions...")

            # Get scaling factor
            scale_text = self.scaling_combo.get_active_text()
            scale_factor = int(scale_text.split('x')[0]) if 'x' in scale_text else 1

            # Get smoothing level
            smoothing_text = self.smoothing_combo.get_active_text()
            smoothing_sigma = self.get_smoothing_sigma(smoothing_text)

            # Get sharpening level
            sharpening_text = self.sharpening_combo.get_active_text()
            sharpening_level = self.get_sharpening_level(sharpening_text)

            # Generate tonal scaled image
            tonal_scaled_image = self.generate_tonal_scaled_image(
                self.raw_image, 
                scale_factor, 
                smoothing_sigma, 
                sharpening_level
            )
            
            tonal_scaled_image.save("tonal_scaled_output.png")
            self.output_label.set_text(
                f"Analysis complete. {scale_text} tonal scaled image "
                f"with {smoothing_text} smoothing and {sharpening_text} sharpening "
                f"saved as 'tonal_scaled_output.png'."
            )
        except Exception as e:
            self.output_label.set_text(f"Error: {str(e)}")

    def get_smoothing_sigma(self, smoothing_text):
        """Convert smoothing text to Gaussian filter sigma value"""
        smoothing_map = {
            "No Smoothing": 0.0,
            "Light": 0.5,
            "Medium": 1.0,
            "Strong": 2.0
        }
        return smoothing_map.get(smoothing_text, 0.0)

    def get_sharpening_level(self, sharpening_text):
        """Convert sharpening text to sharpening intensity"""
        sharpening_map = {
            "No Sharpening": 0.0,
            "Light": 0.5,
            "Medium": 1.0,
            "Strong": 2.0
        }
        return sharpening_map.get(sharpening_text, 0.0)

    def generate_tonal_scaled_image(self, image, scale_factor, smoothing_sigma=0.0, sharpening_level=0.0):
        """Generate a new image with advanced tonal transition preservation, smoothing, and sharpening"""
        height, width, _ = image.shape
        new_height = height * scale_factor
        new_width = width * scale_factor

        tonal_scaled_image = np.zeros((new_height, new_width, 3), dtype=np.float32)

        for channel in range(3):  # Process R, G, B channels separately
            channel_data = image[:, :, channel]
            scaled_channel = self.advanced_tonal_scale_channel(channel_data, scale_factor)
            
            # Convert to float for processing
            scaled_channel = scaled_channel.astype(np.float32)

            # Apply smoothing if sigma > 0
            if smoothing_sigma > 0:
                scaled_channel = gaussian_filter(scaled_channel, sigma=smoothing_sigma)

            # Apply sharpening if level > 0
            if sharpening_level > 0:
                # Laplacian edge detection
                laplace_filter = laplace(scaled_channel)
                # Unsharp masking technique
                scaled_channel = scaled_channel + sharpening_level * laplace_filter

            # Clip values to valid range and convert back to uint8
            tonal_scaled_image[:, :, channel] = np.clip(scaled_channel, 0, 255)

        return Image.fromarray(tonal_scaled_image.astype(np.uint8))

    @staticmethod
    def advanced_tonal_scale_channel(channel, scale_factor):
        """
        Advanced scaling method that preserves complex tonal transitions
        """
        height, width = channel.shape
        new_height = height * scale_factor
        new_width = width * scale_factor

        scaled_channel = np.zeros((new_height, new_width), dtype=np.uint8)

        # Detect local transitions and their characteristics
        def analyze_local_transitions(row):
            transitions = []
            start = 0
            for i in range(1, len(row)):
                if row[i] != row[start]:
                    # Record transition: start index, end index, start value, end value, direction
                    direction = 1 if row[i] > row[start] else -1
                    transitions.append({
                        'start_idx': start, 
                        'end_idx': i, 
                        'start_val': row[start], 
                        'end_val': row[i], 
                        'direction': direction
                    })
                    start = i
            return transitions

        def interpolate_transition(transition, target_length):
            """Interpolate a single tonal transition with preservation of local characteristics"""
            start_val = transition['start_val']
            end_val = transition['end_val']
            
            # Create interpolation that respects the original transition's direction and characteristics
            if target_length > 1:
                if transition['direction'] > 0:
                    # Ascending transition
                    interpolated = np.linspace(start_val, end_val, target_length, endpoint=False)
                else:
                    # Descending transition
                    interpolated = np.linspace(start_val, end_val, target_length, endpoint=False)
                
                return interpolated.astype(np.uint8)
            else:
                return np.array([start_val], dtype=np.uint8)

        # Process each row with advanced tonal preservation
        for y in range(new_height):
            source_row_idx = int(y / scale_factor)
            source_row = channel[source_row_idx]
            
            # Analyze transitions in the source row
            transitions = analyze_local_transitions(source_row)
            
            # Interpolate each transition in the scaled row
            current_pos = 0
            for transition in transitions:
                # Calculate scaled transition length
                scaled_transition_length = int((transition['end_idx'] - transition['start_idx']) * scale_factor)
                
                # Interpolate this transition segment
                interpolated_segment = interpolate_transition(transition, scaled_transition_length)
                
                # Place interpolated segment in scaled row
                end_pos = current_pos + scaled_transition_length
                scaled_channel[y, current_pos:end_pos] = interpolated_segment
                
                current_pos = end_pos

        return scaled_channel
# Run the app
if __name__ == "__main__":
    app = ImageConverterApp()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()
